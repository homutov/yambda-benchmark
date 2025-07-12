import json
from pathlib import Path

import click
import polars as pl
import torch

from yambda.constants import Constants
from yambda.evaluation.metrics import calc_metrics
from yambda.evaluation.ranking import Ranked, Targets
from yambda.processing import timesplit
from yambda.utils import mean_dicts


@click.command()
@click.option(
    '--data_dir',
    required=True,
    type=str,
    default="../../data/flat",
    show_default=True,
    help="Expects flat data",
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
@click.option('--device', required=True, type=str, default="cuda:0", show_default=True)
@click.option('--num_repeats', required=True, type=int, default=2, show_default=True)
def main(
    data_dir: str,
    size: list[str],
    interaction: list[str],
    device: str,
    num_repeats: int,
):
    print(f"calc metrics: {Constants.METRICS}")
    for s in size:
        for i in interaction:
            print(f"SIZE {s}, INTERACTION {i}")
            result = random_rec(data_dir, s, i, num_repeats, device)
            print(json.dumps(result, indent=2))


def scan(path: str, dataset_size: str, dataset_name: str) -> pl.LazyFrame:
    path: Path = Path(path) / dataset_size / dataset_name
    return pl.scan_parquet(path.with_suffix(".parquet"))


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


def random_rec(
    data_dir: str,
    size: str,
    interaction: str,
    num_repeats: int,
    device: str,
) -> dict[str, dict[int, float]]:
    df = scan(data_dir, size, interaction)

    train, _, test = preprocess(df, interaction, val_size=0)

    unique_user_ids = train.select("uid").unique().sort("uid").collect(engine="streaming")["uid"].to_torch().to(device)

    unique_item_ids = (
        train.select("item_id").unique().sort("item_id").collect(engine="streaming")["item_id"].to_torch().to(device)
    )

    print(f"NUM_USERS {unique_user_ids.shape[0]}, NUM_ITEMS {unique_item_ids.shape[0]}")

    targets = Targets.from_sequential(
        test.group_by('uid', maintain_order=True).agg("item_id"),
        device,
    )

    metrics_list = []

    for _ in range(num_repeats):
        ranked = Ranked(
            user_ids=unique_user_ids,
            item_ids=unique_item_ids[
                torch.randint(
                    0, unique_item_ids.shape[0] - 1, size=(unique_user_ids.shape[0], Constants.NUM_RANKED_ITEMS)
                )
            ],
            num_item_ids=unique_item_ids.shape[0],
        )

        metrics_list.append(
            calc_metrics(
                ranked,
                targets,
                metrics=Constants.METRICS,
            )
        )

    return mean_dicts(metrics_list)


if __name__ == "__main__":
    main()
