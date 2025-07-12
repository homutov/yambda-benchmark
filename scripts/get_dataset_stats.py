import enum
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import polars as pl


class ImageFormat(enum.Enum):
    PNG = "png"
    JPEG = "jpg"


@click.command()
@click.option(
    "--src_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to the directory containing dataset, e.g., './data'.",
)
@click.option(
    "--dst_dir",
    type=click.Path(file_okay=False),
    required=False,
    help="Path to the directory where statisics will be saved. "
    "If not specified, statistics is saved in '{src_dir}/stats'. e.g., './stats'.",
)
@click.option(
    "--file_name",
    type=str,
    default="multi_event",
    help="Base name for multi-event file. Default is 'multi_event'.",
)
def cli(src_dir: str, dst_dir: str, file_name: str):
    if dst_dir is None:
        dst_dir = f"{src_dir}/stats"
    generate_dataset_stats(src_dir, dst_dir, file_name)


def generate_dataset_stats(src_dir: str, dst_dir: str, file_name: str):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    src_dir_flat, src_dir_seq = src_dir / "flat", src_dir / "sequential"

    file_name = f"{file_name}.parquet"
    sizes = sorted(path.name for path in src_dir_flat.iterdir())
    for size in sizes:
        path_flat, path_seq = src_dir_flat / size, src_dir_seq / size
        assert path_seq.exists(), f"Cannot find sequential data in '{path_seq}'"
        assert (path_flat / file_name).exists(), (
            "Please, generate flat multi-event file using "
            "'make_multievent.py' or specify correct name using --file_name"
        )
        assert (path_seq / file_name).exists(), (
            "Please, generate sequential multi-event file "
            "using 'transform2sequential.py' script or specify correct name "
            "using --file_name"
        )

        print(f"Gathering stats for {size}...")
        dst_dir_size = dst_dir / size
        dst_dir_size.mkdir(parents=True, exist_ok=True)

        df = pl.scan_parquet(path_seq / file_name)
        generate_user_history_graph(df, dst_dir_size / "user_history_len.png")
        generate_log_user_history_graph(df, dst_dir_size / "user_history_log_len.png")

        df = pl.scan_parquet(path_flat / file_name)
        generate_item_interactions_graph(df, dst_dir_size / "item_interactions.png")
        get_recom_stats(df).write_csv(dst_dir_size / "recom_event_count.csv")
        get_history_len_stats(df).write_csv(dst_dir_size / "event_history_len.csv")
        get_dataset_stats(df).write_csv(dst_dir_size / "dataset_event_stats.csv")


def make_history_len_graph(
    df: pl.DataFrame,
    *,
    qs: tuple[float] | None = None,
    color: str = "lightskyblue",
    num_bins: int = 100,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    count, _, _ = ax.hist(df["value"], bins=num_bins, ec="k", lw=1.0, color=color)
    ylim = count.max() * 1.05

    xs = {"max": df["value"].max()}
    if qs is not None:
        xs.update((f"q{q * 100:.0f}", df["value"].quantile(q)) for q in qs)

    dx = 0.01 * (xs["max"] - df["value"].min())
    template = "{label}={x:.3f}" if xs["max"] <= 10 else "{label}={x:.0f}"
    for label, x in xs.items():
        ax.plot([x, x], [0, ylim], ls="--", c="k")
        text = template.format(label=label, x=x)
        ax.text(x + dx, ylim // 2, text, rotation=90, fontsize=16, bbox=dict(alpha=0.1, color="r"))

    if title is not None:
        ax.set_title(title, fontsize=24)
    ax.set_xlabel(xlabel or "Value", fontsize=22)
    ax.set_ylabel(ylabel or "Count", fontsize=22)

    ax.set_ylim([0, ylim])

    ax.tick_params(labelsize=16)
    ax.ticklabel_format(style="sci", useMathText=True)


def save_graph(output_path: os.PathLike, fmt: ImageFormat = ImageFormat.PNG):
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{fmt.value}")

    if output_path.exists():
        print(f"Rewriting file '{output_path}'")
    else:
        print(f"Saving to '{output_path}'")

    plt.savefig(output_path, dpi=300, format=fmt.value)


def generate_user_history_graph(df: pl.LazyFrame, out_path: os.PathLike):
    _, ax = plt.subplots(figsize=(12, 5))

    make_history_len_graph(
        df.select(value=pl.col("item_id").list.len()).collect(),
        num_bins=100,
        qs=(0.5, 0.9, 0.95),
        xlabel="Events",
        ylabel="Users",
        ax=ax,
    )
    plt.tight_layout()
    save_graph(out_path)


def generate_log_user_history_graph(df: pl.LazyFrame, out_path: os.PathLike):
    _, ax = plt.subplots(figsize=(12, 5))

    make_history_len_graph(
        df.select(value=pl.col("item_id").list.len().log10()).collect(),
        num_bins=40,
        xlabel="$Log_{10}$(Events)",
        ylabel="Users",
        color="lightgreen",
        ax=ax,
    )
    plt.tight_layout()
    save_graph(out_path)


def generate_item_interactions_graph(df: pl.LazyFrame, out_path: os.PathLike):
    _, ax = plt.subplots(figsize=(12, 5))

    make_history_len_graph(
        df.group_by("item_id").len().select(value=pl.col("len").log10()).collect(),
        num_bins=30,
        qs=(0.5, 0.9, 0.95),
        xlabel="$Log_{10}$(Events)",
        ylabel="Items",
        color="orange",
        ax=ax,
    )
    plt.tight_layout()
    save_graph(out_path)


def get_recom_stats(df: pl.LazyFrame) -> pl.DataFrame:
    print("Computing recom stats")
    df_cnt = df.group_by(("event_type", "is_organic")).len().collect()
    df_recom = df_cnt.filter(pl.col("is_organic").eq(0)).select(pl.col("event_type"), pl.col("len").alias("recom"))
    df_total = df_cnt.group_by("event_type").sum().select(pl.col("event_type"), pl.col("len").alias("total"))
    return df_total.join(df_recom, on="event_type").with_columns(ratio=pl.col("recom") / pl.col("total"))


def get_history_len_stats(df: pl.LazyFrame) -> pl.DataFrame:
    print("Computing event history length stats")
    return (
        df.group_by(("event_type", "uid"))
        .len()
        .group_by("event_type")
        .agg(
            median=pl.col("len").quantile(0.5).cast(pl.Int32),
            q90=pl.col("len").quantile(0.9).cast(pl.Int32),
            q95=pl.col("len").quantile(0.95).cast(pl.Int32),
        )
        .collect()
    )


def get_dataset_stats(df: pl.LazyFrame) -> pl.DataFrame:
    print("Computing dataset stats")
    return df.select(
        users=pl.col("uid").unique().len(),
        items=pl.col("item_id").unique().len(),
        listens=pl.col("event_type").filter(pl.col("event_type").eq("listen")).len(),
        likes=pl.col("event_type").filter(pl.col("event_type").eq("like")).len(),
        dislikes=pl.col("event_type").filter(pl.col("event_type").eq("dislike")).len(),
        unlikes=pl.col("event_type").filter(pl.col("event_type").eq("unlike")).len(),
        undislikes=pl.col("event_type").filter(pl.col("event_type").eq("undislike")).len(),
    ).collect()


if __name__ == "__main__":
    cli()
