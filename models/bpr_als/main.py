import json
from pathlib import Path
from typing import Any

import click
import numpy as np
import optuna
import polars as pl
import scipy.sparse as sp
import torch
from implicit.gpu.als import AlternatingLeastSquares
from implicit.gpu.bpr import BayesianPersonalizedRanking

from yambda.constants import Constants
from yambda.evaluation.metrics import calc_metrics
from yambda.evaluation.ranking import Embeddings, Targets, rank_items
from yambda.processing import timesplit
from yambda.utils import mean_dicts


@click.command()
@click.option(
    '--data_dir', required=True, type=str, default="../../data/flat", show_default=True, help="Expects flat data"
)
@click.option(
    '--size',
    required=True,
    type=click.Choice(['50m', '500m', "5b"]),
    default=["50m"],
    multiple=True,
    show_default=True,
)
@click.option(
    '--interaction',
    required=True,
    type=click.Choice(['likes', 'listens']),
    default=["likes"],
    multiple=True,
    show_default=True,
)
@click.option(
    "--model",
    required=True,
    type=click.Choice(['als', 'bpr']),
    multiple=False,
    show_default=True,
)
@click.option('--validation_metric', required=True, type=str, default="ndcg@100", show_default=True)
@click.option('--report_metrics', required=True, type=str, default=Constants.METRICS, multiple=True, show_default=True)
@click.option('--device', required=True, type=str, default="cuda:0", show_default=True)
def main(
    data_dir: str,
    size: list[str],
    interaction: list[str],
    model: str,
    validation_metric: str,
    report_metrics: list[str],
    device: str,
):
    print(f"REPORT METRICS: {report_metrics}")

    for s in size:
        for i in interaction:
            print(f"SIZE {s}, INTERACTION {i}")
            result = bpr_als(
                data_dir, s, i, device, model, validation_metric=validation_metric, report_metrics=report_metrics
            )
            print(json.dumps(result, indent=2))


def scan(path: str, dataset_size: str, dataset_name: str) -> pl.LazyFrame:
    path: Path = Path(path) / dataset_size / dataset_name
    df = pl.scan_parquet(path.with_suffix(".parquet"))
    return df


def preprocess(
    df: pl.LazyFrame, interaction: str, val_size: int
) -> tuple[pl.LazyFrame, pl.LazyFrame | None, pl.LazyFrame]:
    if interaction == "listens":
        df = df.filter(pl.col("played_ratio_pct") >= Constants.TRACK_LISTEN_THRESHOLD).select(
            "uid", "item_id", "timestamp"
        )

    train, val, test = timesplit.flat_split_train_val_test(
        df, val_size=val_size, test_timestamp=Constants.TEST_TIMESTAMP
    )

    return (
        train,
        val.collect(engine="streaming").lazy() if val is not None else None,
        test.collect(engine="streaming").lazy(),
    )


def process_train_data(df: pl.LazyFrame) -> tuple[pl.LazyFrame, list[int], list[int]]:
    unique_pairs = df.select("uid", "item_id").unique()

    unique_uids = df.select("uid").unique().sort("uid").collect(engine="streaming")["uid"].to_list()
    unique_item_ids = df.select("item_id").unique().sort("item_id").collect(engine="streaming")["item_id"].to_list()

    return unique_pairs, unique_uids, unique_item_ids


def build_csr_matrix(pairs: pl.LazyFrame, unique_uids: list[int], unique_item_ids: list[int]) -> sp.csr_matrix:
    uid_to_idx = {uid: i for i, uid in enumerate(unique_uids)}
    item_id_to_idx = {item_id: i for i, item_id in enumerate(unique_item_ids)}

    pairs = pairs.select(
        pl.col("uid").replace_strict(uid_to_idx, return_dtype=pl.UInt32),
        pl.col("item_id").replace_strict(item_id_to_idx, return_dtype=pl.UInt32),
    ).collect(engine="streaming")

    rows, cols = pairs["uid"].to_numpy(), pairs["item_id"].to_numpy()
    values = np.ones_like(rows, dtype=np.int32)

    return sp.coo_matrix(
        (values, (rows, cols)), dtype=np.float32, shape=(len(unique_uids), len(unique_item_ids))
    ).tocsr()


def train_embbedings_with_als(
    user_item_interactions: sp.csr_matrix,
    regularization: float = 0.01,
    iterations: int = 100,
    random_state: int = 42,
    factors: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    als = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
        calculate_training_loss=False,
    )
    als.fit(user_item_interactions, show_progress=False)
    return als.user_factors.to_numpy(), als.item_factors.to_numpy()


def train_embbedings_with_bpr(
    user_item_interactions: sp.csr_matrix,
    learning_rate: float = 0.01,
    regularization: float = 0.01,
    iterations: int = 100,
    random_state: int = 42,
    factors: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    bpr = BayesianPersonalizedRanking(
        factors=factors,
        learning_rate=learning_rate,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
        verify_negative_samples=True,
    )
    bpr.fit(user_item_interactions, show_progress=False)
    return bpr.user_factors.to_numpy(), bpr.item_factors.to_numpy()


def calc_embeddings_metrics(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    uid_tensor: torch.Tensor,
    item_id_tensor: torch.Tensor,
    targets: Targets,
    metrics: list[str],
    device: str,
) -> dict[str, dict[int, float]]:
    num_ranked_items = max([int(x.split("@")[1]) for x in metrics])
    user_emb = Embeddings(uid_tensor, torch.from_numpy(user_emb).to(device))
    item_emb = Embeddings(item_id_tensor, torch.from_numpy(item_emb).to(device))

    ranked_items = rank_items(user_emb, item_emb, num_ranked_items)
    return calc_metrics(ranked_items, targets, metrics)


def hyperopt(
    train: pl.LazyFrame,
    val: pl.LazyFrame,
    n_trials: int,
    model: str,
    validation_metric: str,
    device: str,
) -> dict[str, float | int]:
    pairs, unique_uids, unique_item_ids = process_train_data(train)

    user_item_interactions = build_csr_matrix(pairs, unique_uids, unique_item_ids)

    targets = Targets.from_sequential(val.group_by("uid").agg("item_id").collect(engine="streaming"), device)

    def objective(trial: optuna.Trial) -> float:
        iterations = trial.suggest_int(name="iterations", low=10, high=300, log=True)
        regularization = trial.suggest_float(name="regularization", low=1e-5, high=1, log=True)

        if model == "als":
            user_emb, item_emb = train_embbedings_with_als(user_item_interactions, regularization, iterations)
        else:
            learning_rate = trial.suggest_float(name="learning_rate", low=1e-5, high=0.1, log=True)
            user_emb, item_emb = train_embbedings_with_bpr(
                user_item_interactions, learning_rate, regularization, iterations
            )

        metrics = calc_embeddings_metrics(
            user_emb,
            item_emb,
            torch.tensor(unique_uids, device=device),
            torch.tensor(unique_item_ids, device=device),
            targets,
            [validation_metric],
            device,
        )

        t, k = validation_metric.split('@')
        return metrics[t][int(k)]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def evaluation(
    train: pl.LazyFrame,
    test: pl.LazyFrame,
    model: str,
    hp: dict[str, Any],
    random_seeds: list[int],
    report_metrics: list[str],
    device: str,
) -> dict[str, dict[int, float]]:
    train, unique_uids, unique_item_ids = process_train_data(train)

    user_item_interactions = build_csr_matrix(train, unique_uids, unique_item_ids)

    targets = Targets.from_sequential(
        test.group_by("uid").agg("item_id").collect(engine="streaming"),
        device,
    )

    metrics_list = []

    for seed in random_seeds:
        if model == "als":
            user_emb, item_emb = train_embbedings_with_als(
                user_item_interactions, hp["regularization"], hp["iterations"], seed
            )
        else:
            user_emb, item_emb = train_embbedings_with_bpr(
                user_item_interactions, hp["learning_rate"], hp["regularization"], hp["iterations"], seed
            )

        metrics = calc_embeddings_metrics(
            user_emb,
            item_emb,
            torch.tensor(unique_uids, device=device),
            torch.tensor(unique_item_ids, device=device),
            targets,
            report_metrics,
            device,
        )
        metrics_list.append(metrics)

    return mean_dicts(metrics_list)


def bpr_als(
    data_dir: str,
    size: str,
    interaction: str,
    device: str,
    model: str,
    validation_metric: str,
    report_metrics: list[str],
    n_trials: dict[str, int] = {"50m": 10, "500m": 2, "5b": 2},
):
    df = scan(data_dir, size, interaction)

    # hyperopt by validation
    train, val, _ = preprocess(df, interaction, val_size=Constants.VAL_SIZE)
    hp = hyperopt(train, val, n_trials[size], model, validation_metric, device)

    print("Best HP:")
    print(json.dumps(hp, indent=2))

    # final model
    train, _, test = preprocess(df, interaction, val_size=0)

    return evaluation(
        train,
        test,
        model,
        hp,
        random_seeds=[41, 42, 43],
        report_metrics=report_metrics,
        device=device,
    )


if __name__ == "__main__":
    main()
