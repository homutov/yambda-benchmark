import gc
import json
from pathlib import Path
from typing import Any

import click
import polars as pl
import torch

from yambda.constants import Constants
from yambda.evaluation.metrics import calc_metrics
from yambda.evaluation.ranking import Embeddings, Targets, rank_items
from yambda.processing import timesplit
from yambda.utils import argmax


DEFAULT_GRIDS = {
    "50m": {
        "likes": [0, 0.001, 0.25, 0.5, 1, 2, 3, 4, 6],
        "listens": [0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.5, 1.0, 2],
    },
    "500m": {
        "likes": [0, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.5, 1.0],
        "listens": [0, 0.001, 0.002, 0.004, 0.008],
    },
}


@click.command()
@click.option(
    '--data_dir', required=True, type=str, default="../../data/flat", show_default=True, help="Expects flat data"
)
@click.option(
    '--size',
    required=True,
    type=click.Choice(['50m', '500m']),  # 5b is not supported
    default="50m",
    multiple=False,
    show_default=True,
    help="5b is not supported due to (num_user, num_user) matrix",
)
@click.option(
    '--interaction',
    required=True,
    type=click.Choice(['likes', 'listens']),
    default="likes",
    multiple=False,
    show_default=True,
)
@click.option(
    '--hours',
    required=True,
    type=float,
    default=[-1],
    multiple=True,
    show_default=True,
    help="Hyperparameter. If -1 default grid will be used",
)
@click.option('--validation_metric', required=True, type=str, default="ndcg@100", show_default=True)
@click.option('--report_metrics', required=True, type=str, default=Constants.METRICS, multiple=True, show_default=True)
@click.option('--device', required=True, type=str, default="cuda:0", show_default=True)
def main(
    data_dir: str,
    size: str,
    interaction: str,
    hours: list[float],
    validation_metric: str,
    report_metrics: list[str],
    device: str,
):
    print(f"REPORT METRICS: {report_metrics}")
    print(f"SIZE {size}, INTERACTION {interaction}")
    result = item_knn(
        data_dir,
        size,
        interaction,
        device,
        hours=hours if hours[0] != -1 else DEFAULT_GRIDS[size][interaction],
        validation_metric=validation_metric,
        report_metrics=report_metrics,
    )
    print(json.dumps(result, indent=2))


def scan(path: str, dataset_size: str, dataset_name: str) -> pl.LazyFrame:
    path = Path(path) / dataset_size / dataset_name
    df = pl.scan_parquet(path.with_suffix(".parquet"))
    return df


def preprocess(
    df: pl.LazyFrame, interaction: str, val_size: int
) -> tuple[pl.LazyFrame, pl.LazyFrame | None, pl.LazyFrame]:
    if interaction == "listens":
        df = df.filter(pl.col("played_ratio_pct") >= Constants.TRACK_LISTEN_THRESHOLD)

    train, val, test = timesplit.flat_split_train_val_test(
        df, val_size=val_size, test_timestamp=Constants.TEST_TIMESTAMP
    )

    return (
        train,
        val.collect(engine="streaming").lazy() if val is not None else None,
        test.collect(engine="streaming").lazy(),
    )


def eliminate_zeros(x: torch.Tensor, threshold: float = 1e-9) -> torch.Tensor:
    mask = (x._values() > threshold).nonzero()
    nv = x._values().index_select(0, mask.view(-1))
    ni = x._indices().index_select(1, mask.view(-1))
    return torch.sparse_coo_tensor(ni, nv, x.shape)


def create_weighted_sparse_tensor(train: pl.LazyFrame, tau: float) -> torch.Tensor:
    uid_mapping = (
        train.select("uid").unique().with_columns(pl.col("uid").rank(method="dense").alias("uid_idx") - 1).collect()
    )

    item_mapping = (
        train.select("item_id")
        .unique()
        .with_columns(pl.col("item_id").rank(method="dense").alias("item_idx") - 1)
        .collect()
    )

    processed = (
        train.with_columns(pl.max("timestamp").over("uid").alias("max_timestamp"))
        .with_columns((pl.col("max_timestamp") - pl.col("timestamp")).alias("delta"))
        .with_columns((tau ** pl.col("delta")).alias("weight"))
        .join(uid_mapping.lazy(), on="uid", how="inner")
        .join(item_mapping.lazy(), on="item_id", how="inner")
    )

    coo_data = processed.group_by(["uid_idx", "item_idx"]).agg(pl.sum("weight").alias("total_weight")).collect()

    indices = torch.concat([coo_data["uid_idx"].to_torch()[None, :], coo_data["item_idx"].to_torch()[None, :]], dim=0)
    values = torch.tensor(coo_data["total_weight"].to_numpy(), dtype=torch.float)

    return eliminate_zeros(
        torch.sparse_coo_tensor(
            indices=indices, values=values, size=(uid_mapping["uid_idx"].max() + 1, item_mapping["item_idx"].max() + 1)
        )
    )


def sparse_normalize(sparse_tensor: torch.Tensor, dim=0, eps=1e-12):
    indices = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()

    unique_dim_indices, inverse = torch.unique(indices[dim], return_inverse=True)
    squared_values = values**2
    sum_squared = torch.zeros_like(unique_dim_indices, dtype=torch.float32)
    sum_squared.scatter_add_(0, inverse, squared_values)

    norms = torch.sqrt(sum_squared + eps)
    normalized_values = values / norms[inverse]

    return torch.sparse_coo_tensor(indices, normalized_values, sparse_tensor.size())


def training(
    train: pl.LazyFrame, hour: float, user_item: torch.Tensor, user_ids: torch.Tensor, device: str, decay: float = 0.9
) -> Embeddings:
    tau = 0.0 if hour == 0 else decay ** (1 / 24 / 60 / 60 / (hour / 24))

    user_item_with_tau = create_weighted_sparse_tensor(train, tau)
    user_embeddings = (user_item_with_tau @ user_item.T).to_dense()
    user_embeddings = torch.nn.functional.normalize(user_embeddings, dim=-1)

    return Embeddings(user_ids, user_embeddings.to(device))


def evaluation(
    train: pl.LazyFrame, val: pl.LazyFrame, device: str, hours: list[float], metrics: list[str]
) -> list[dict[str, Any]]:
    num_ranked_items = max([int(x.split("@")[1]) for x in metrics])

    unique_user_ids = train.select("uid").unique().sort("uid").collect(engine="streaming")["uid"].to_torch().to(device)
    unique_item_ids = (
        train.select("item_id").unique().sort("item_id").collect(engine="streaming")["item_id"].to_torch().to(device)
    )

    user_item = create_weighted_sparse_tensor(train, 1.0)
    item_embeddings = sparse_normalize(user_item.T.to(device), dim=-1)
    item_embeddings = Embeddings(unique_item_ids, item_embeddings)

    targets = Targets.from_sequential(
        val.group_by('uid', maintain_order=True).agg(pl.all().exclude('uid')).select(['uid', 'item_id']),
        device,
    )

    hour2metrics = []
    for hour in hours:
        user_embeddings = training(
            train=train,
            hour=hour,
            user_item=user_item,
            user_ids=unique_user_ids,
            device=device,
        )

        ranked = rank_items(
            users=user_embeddings,
            items=item_embeddings,
            num_items=num_ranked_items,
            batch_size=128,
        )

        del user_embeddings
        gc.collect()

        hour2metrics.append(calc_metrics(ranked, targets, metrics))

    del unique_user_ids
    del unique_item_ids
    del item_embeddings
    del targets
    gc.collect()

    return hour2metrics


def item_knn(
    data_dir: str,
    size: str,
    interaction: str,
    device: str,
    hours: list[float],
    validation_metric: str,
    report_metrics: list[str],
) -> dict[str, Any]:
    df = scan(data_dir, size, interaction)

    # hyperopt by validation
    train, val, _ = preprocess(df, interaction, val_size=Constants.VAL_SIZE)

    results = evaluation(train, val, device, hours, [validation_metric])

    metric_name, k = validation_metric.split('@')

    best_hour = hours[argmax(results, lambda x: x[metric_name][int(k)])]

    print(f"FINAL HYPERPARAMS {best_hour=}")

    # train final model
    train, _, test = preprocess(df, interaction, val_size=0)

    return evaluation(train, test, device, [best_hour], report_metrics)[0]


if __name__ == "__main__":
    main()
