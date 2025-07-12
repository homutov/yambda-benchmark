import os

import click
import polars as pl


@click.command()
@click.option(
    "--src_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to the directory containing source Parquet files, e.g., './50m'.",
)
@click.option(
    "--dst_dir",
    type=click.Path(file_okay=False),
    required=False,
    help="Path to the directory where Parquet files will be saved. "
    "If not specified, Parquet files are saved in 'src_dir' with prefix 'seq_'.",
)
@click.option(
    "--files",
    type=str,
    multiple=True,
    help="List of parquet filenames to convert. If not specified, all Parquet files in 'src_dir' will be converted. "
    "For example, '--files dislike.parquet --files like.parquet'.",
)
@click.option(
    "--aggregation",
    type=click.Choice(["structs", "columns"]),
    required=True,
    help="Agg method: 'structs' for a sequence of structs per 'uid', 'columns' for individual column aggregation.",
)
def cli(src_dir: str, dst_dir: str, files: list[str], aggregation: str):
    transform2sequential(src_dir, dst_dir, files, aggregation)


def transform2sequential(src_dir: str, dst_dir: str, files: list[str], aggregation: str):
    for file in files:
        print(f"Processing file: {file}")

        src_path = os.path.join(src_dir, file)

        parquet_path = os.path.join(dst_dir, file)

        if os.path.exists(parquet_path):
            parquet_path = os.path.join(dst_dir, f"{aggregation}_" + file)
            assert not os.path.exists(parquet_path)

        os.makedirs(dst_dir, exist_ok=True)

        df = pl.scan_parquet(src_path)

        if aggregation == "structs":
            seq_df = (
                df.select("uid", pl.struct(pl.all().exclude("uid")).alias("events"))
                .group_by("uid", maintain_order=True)
                .agg(pl.col("events"))
            )
            seq_df.sink_parquet(
                parquet_path,
                compression="lz4",
                statistics=True,
            )

        elif aggregation == "columns":
            col_agg_df = df.group_by("uid", maintain_order=True).agg(pl.all().exclude("uid"))
            col_agg_df.sink_parquet(
                parquet_path,
                compression="lz4",
                statistics=True,
            )


if __name__ == "__main__":
    cli()
