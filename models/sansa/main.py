import heapq
import json
import os
import time
from typing import Any

import click
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from sansa import SANSA, ICFGramianFactorizerConfig, SANSAConfig, UMRUnitLowerTriangleInverterConfig
from tqdm import tqdm

from yambda.constants import Constants
from yambda.evaluation import metrics, ranking
from yambda.processing import timesplit


RANDOM_SEED = 42


@click.command()
@click.option(
    '--data_dir', required=True, type=str, default="../../data/flat", show_default=True, help="Expects flat data"
)
@click.option(
    '--size',
    required=True,
    type=click.Choice(['50m', '500m']),
    default="50m",
    multiple=False,
    show_default=True,
)
@click.option(
    '--interaction',
    required=True,
    type=click.Choice(['likes', 'listens']),
    default="likes",
    multiple=False,
    show_default=True,
)
@click.option('--report_metrics', required=True, type=str, default=Constants.METRICS, multiple=True, show_default=True)
@click.option('--device', required=True, type=str, default="cuda:0", show_default=True)
def main(
    data_dir: str,
    size: str,
    interaction: str,
    report_metrics: list[str],
    device: str,
):
    print(f"REPORT METRICS: {report_metrics}")
    print(f"SIZE {size}, INTERACTION {interaction}")
    result = train_sansa_model(
        data_dir,
        size=size,
        dataset_type=interaction,
        device=device,
        report_metrics=report_metrics,
    )
    print(json.dumps(result, indent=2))


def train_sansa_model(
    data_path: str,
    size: str,
    dataset_type: str,
    device: str,
    report_metrics: list[str],
) -> dict[str, Any]:
    np.random.seed(RANDOM_SEED)

    curr_time = time.time()
    print()
    print(curr_time)
    print(f"Size: {size}, Dataset: {dataset_type}")
    df, grouped_test, train, test = get_train_val_test_matrices(
        data_path=data_path,
        size=size,
        dataset_type=dataset_type,
    )
    data_finished = time.time()
    print(f"Data is loaded in {data_finished - curr_time} seconds")

    model = get_sansa_model()
    model.fit(train)
    train_finished = time.time()

    print(f"Model is trained in {train_finished - data_finished}")
    print(model)

    if report_metrics:
        calculated_metrics = evaluate_sansa(
            df=df,
            model=model,
            device=device,
            report_metrics=report_metrics,
            grouped_test=grouped_test,
            sparse_train=train,
            sparse_test=test,
        )

        print(f"Model is evaluated in {time.time() - train_finished}")

        return calculated_metrics

    return {}


def get_train_val_test_matrices(
    data_path: str,
    size: str = "50m",
    dataset_type: str = "likes",
) -> tuple[pl.LazyFrame, pl.LazyFrame, sp.csr_matrix, sp.csr_matrix]:
    df = pl.scan_parquet(os.path.join(os.path.join(data_path, size, f"{dataset_type}.parquet")))

    if dataset_type == "listens":
        df = df.filter(pl.col("played_ratio_pct") >= Constants.TRACK_LISTEN_THRESHOLD)

    flat_train, _, flat_test = timesplit.flat_split_train_val_test(
        df, val_size=0, test_timestamp=Constants.TEST_TIMESTAMP
    )

    all_uids = set(flat_train.collect().get_column("uid").to_list())
    all_items = set(flat_train.collect().get_column("item_id").to_list())

    print(f"Dataset, users_num: {len(all_uids)}, items_num: {len(all_items)}")

    # Create mapping to create sparse matrix
    uid_to_idx = {uid: i for i, uid in enumerate(all_uids)}
    item_id_to_idx = {item_id: i for i, item_id in enumerate(all_items)}

    sparse_train, _ = get_sparse_data(flat_train, uid_to_idx, item_id_to_idx)
    sparse_test, grouped_test = get_sparse_data(flat_test, uid_to_idx, item_id_to_idx)

    print(f"Sparse train shape: {sparse_train.shape}, test shape: {sparse_test.shape}")

    return df, grouped_test, sparse_train, sparse_test


def get_sparse_data(
    df: pl.LazyFrame, uid_to_idx: dict[int, int], item_id_to_idx: dict[int, int]
) -> tuple[sp.csr_matrix, pl.LazyFrame]:
    df = df.with_columns(
        pl.col("uid").replace_strict(uid_to_idx).alias("uid"),
        pl.col("item_id").replace_strict(item_id_to_idx, default=len(item_id_to_idx)).alias("item_id"),
        pl.lit(1).alias("action"),
    )

    grouped_df = df.group_by('uid', maintain_order=True).agg(
        [pl.col('item_id').alias('item_id'), pl.col('action').alias('actions')]
    )

    rows = []
    cols = []
    values = []

    for user_id, item_ids, actions in tqdm(grouped_df.select('uid', 'item_id', 'actions').collect().rows()):
        rows.extend([user_id] * len(item_ids))
        cols.extend(item_ids)
        values.extend(actions)

    user_item_data = sp.csr_matrix(
        (values, (rows, cols)),
        dtype=np.float32,
        shape=(len(uid_to_idx), len(item_id_to_idx) + 1),  # +1 for default unknown test items
    )

    return user_item_data, grouped_df


def get_sansa_model() -> SANSA:
    factorizer_config = ICFGramianFactorizerConfig(
        # reordering_use_long=True,
        factorization_shift_step=1e-3,  # initial diagonal shift if incomplete factorization fails
        factorization_shift_multiplier=2.0,  # multiplier for the shift for subsequent attempts
    )

    inverter_config = UMRUnitLowerTriangleInverterConfig(
        scans=1,  # number of scans through all columns of the matrix
        finetune_steps=15,  # number of finetuning steps, targeting worst columns
    )

    config = SANSAConfig(
        l2=10.0,  # regularization strength
        weight_matrix_density=5e-5,  # desired density of weights
        gramian_factorizer_config=factorizer_config,  # factorizer configuration
        lower_triangle_inverter_config=inverter_config,  # inverter configuration
    )

    print(config)

    model = SANSA(config)

    return model


def evaluate_sansa(
    df: pl.LazyFrame,
    model: SANSA,
    device: str,
    report_metrics: list[str],
    grouped_test: pl.LazyFrame,
    sparse_train: sp.csr_matrix,
    sparse_test: sp.csr_matrix,
) -> dict[str, Any]:
    num_items_for_metrics = len(set(df.collect().get_column("item_id").to_list()))
    print(num_items_for_metrics)

    test_targets = ranking.Targets.from_sequential(grouped_test, device=device)
    print(len(test_targets.user_ids))

    # to free some RAM
    del df, grouped_test

    train_pred_sparse = model.forward(sparse_train)
    print(f"Train prediction shape: {train_pred_sparse.shape}")

    A = train_pred_sparse
    num_users = A.shape[0]
    num_items_k = 150

    # 0 if there is no such item
    top_items_idx = np.full((num_users, num_items_k), 0, dtype=int)

    # -1 score if there is no such item
    top_items_score = np.full((num_users, num_items_k), -1, dtype=A.data.dtype)

    for row in tqdm(range(num_users)):
        start, end = A.indptr[row], A.indptr[row + 1]
        row_scores = A.data[start:end]
        row_cols = A.indices[start:end]

        if len(row_scores) == 0:
            continue

        k_here = min(num_items_k, len(row_scores))
        top_k = heapq.nlargest(k_here, zip(row_scores, row_cols), key=lambda x: x[0])

        # Fill in
        for i, (score, idx) in enumerate(top_k):
            top_items_idx[row, i] = idx
            top_items_score[row, i] = score

    user_ids = torch.arange(top_items_idx.shape[0], dtype=torch.int32, device="cpu")
    print(user_ids.shape)

    scores = torch.as_tensor(top_items_score, dtype=torch.float32, device="cpu")
    print(scores.shape)

    scores_indices = torch.as_tensor(top_items_idx, dtype=torch.long, device="cpu")
    print(scores_indices.shape)

    targets = torch.as_tensor(sparse_test.toarray(), dtype=torch.bool, device="cpu")
    print(targets.shape)

    targets = targets.to(dtype=torch.bool, device=device)
    not_zero_user_indices = targets.any(dim=1)
    print(torch.sum(not_zero_user_indices))

    not_zero_user_indices = not_zero_user_indices.to(dtype=torch.bool, device="cpu")

    user_ids = user_ids[not_zero_user_indices]
    scores = scores[not_zero_user_indices]
    print(f"After removing zero users scores shape: {scores.shape}, targets shape: {targets.shape}")

    scores_indices = scores_indices[not_zero_user_indices]
    print(scores_indices.shape)

    test_ranked = ranking.Ranked(
        user_ids=user_ids.to(device),
        scores=scores.to(device),
        item_ids=scores_indices.to(device),
        num_item_ids=num_items_for_metrics,
    )

    calculated_metrics = metrics.calc_metrics(test_ranked, test_targets, report_metrics)
    print(calculated_metrics)

    return calculated_metrics


if __name__ == "__main__":
    main()
