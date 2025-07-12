import numpy as np
import polars as pl

from yambda.processing.timesplit import flat_split_train_val_test, sequential_split_train_val_test


def create_dataframe(n: int = 1000) -> pl.DataFrame:
    uids = np.random.randint(1, int(n * 0.05), size=n)
    item_ids = np.random.randint(100, 200, size=n)

    timestamps = np.random.randint(0, 100_000, size=n)
    is_organic = np.random.choice([True, False], size=n)

    df = pl.DataFrame(
        {"uid": uids, "item_id": item_ids, "timestamp": timestamps, "is_organic": is_organic},
        schema={"uid": pl.UInt32, "item_id": pl.UInt32, "timestamp": pl.UInt32, "is_organic": pl.UInt8},
    )

    df = df.sort(["uid", "timestamp"])

    return df


def test_cross_check():
    df = create_dataframe(10000)

    q75_timestamp = int(df["timestamp"].quantile(0.75))

    print(q75_timestamp)

    flat_train, flat_val, flat_test = flat_split_train_val_test(
        df.lazy(), test_timestamp=q75_timestamp, gap_size=1000, val_size=1000
    )

    assert flat_val is not None

    df.group_by("uid", maintain_order=True).agg(pl.all().exclude("uid")).lazy()

    seq_train, seq_val, seq_test = sequential_split_train_val_test(
        df.group_by("uid", maintain_order=True).agg(pl.all().exclude("uid")).lazy(),
        test_timestamp=q75_timestamp,
        gap_size=1000,
        val_size=1000,
    )

    assert seq_val is not None

    assert seq_train.explode(pl.all().exclude("uid")).collect().equals(flat_train.collect())
    assert seq_val.explode(pl.all().exclude("uid")).collect().equals(flat_val.collect())
    assert seq_test.explode(pl.all().exclude("uid")).collect().equals(flat_test.collect())
