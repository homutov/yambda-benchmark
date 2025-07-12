import os

import click
import polars as pl


@click.command()
@click.option(
    "--src_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to the directory containing source parquet files, e.g., './50m'.",
)
@click.option(
    "--dst_dir",
    type=click.Path(file_okay=False),
    required=False,
    help="Path to the directory where Parquet files will be saved. "
    "If not specified, Parquet files are saved in 'src_dir'. e.g., './out'.",
)
@click.option(
    "--file_name",
    type=str,
    default="multi_event",
    help="Base name for the output Parquet file. Default is 'multi_event'.",
)
def cli(src_dir: str, dst_dir: str, file_name: str):
    if dst_dir is None:
        dst_dir = src_dir

    print(f"{src_dir=}, {dst_dir=}, {file_name=}")
    make_multievent_dataset(src_dir, dst_dir, file_name)


def make_multievent_dataset(src_dir: str, dst_dir: str, file_name: str):
    os.makedirs(dst_dir, exist_ok=True)

    dislikes = pl.scan_parquet(os.path.join(src_dir, "dislikes.parquet"))
    likes = pl.scan_parquet(os.path.join(src_dir, "likes.parquet"))
    listens = pl.scan_parquet(os.path.join(src_dir, "listens.parquet"))
    undislikes = pl.scan_parquet(os.path.join(src_dir, "undislikes.parquet"))
    unlikes = pl.scan_parquet(os.path.join(src_dir, "unlikes.parquet"))

    events = pl.Enum(["listen", "dislike", "like", "undislike", "unlike"])

    combined_df = pl.concat(
        [
            listens.with_columns(
                pl.lit("listen").cast(events).alias("event_type"),
            ),
            dislikes.with_columns(
                pl.lit(None).alias("played_ratio_pct"),
                pl.lit(None).alias("track_length_seconds"),
                pl.lit("dislike").cast(events).alias("event_type"),
            ),
            likes.with_columns(
                pl.lit(None).alias("played_ratio_pct"),
                pl.lit(None).alias("track_length_seconds"),
                pl.lit("like").cast(events).alias("event_type"),
            ),
            undislikes.with_columns(
                pl.lit(None).alias("played_ratio_pct"),
                pl.lit(None).alias("track_length_seconds"),
                pl.lit("undislike").cast(events).alias("event_type"),
            ),
            unlikes.with_columns(
                pl.lit(None).alias("played_ratio_pct"),
                pl.lit(None).alias("track_length_seconds"),
                pl.lit("unlike").cast(events).alias("event_type"),
            ),
        ]
    ).sort(
        by=[
            "uid",
            "timestamp",
        ],
        maintain_order=True,
    )

    combined_df.with_columns(pl.col("event_type").cast(events)).sink_parquet(
        os.path.join(dst_dir, file_name + ".parquet"),
        compression="lz4",
        statistics=True,
    )


if __name__ == "__main__":
    cli()
