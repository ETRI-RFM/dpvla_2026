"""Fast initial-pose lookup for LeRobot v3.0 datasets.

Reads the chosen episode's first frame `observation.state` directly from
parquet files, bypassing LeRobotDataset's full-metadata scan (which can
take minutes on large datasets such as r_0123_w_latent: 15k+ episodes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_init_state_from_parquet(
    root: str,
    episode_idx: int,
) -> np.ndarray:
    """Return the first frame's full `observation.state` of `episode_idx`.

    Args:
        root:        LeRobot v3.0 dataset root (must contain meta/ and data/).
        episode_idx: Episode number.

    Returns:
        np.ndarray, dtype=float32, shape=(state_dim,)
        For the dual-arm G1 + brainco hand: 26 dims
        ([left_arm(7), right_arm(7), left_hand(6), right_hand(6)]).
    """
    root = Path(root)

    # 1) Find the episode's metadata row across the chunked meta files.
    ep_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No meta/episodes parquet files under {root}")

    ep_row = None
    for f in ep_files:
        t = pq.read_table(
            str(f),
            columns=[
                "episode_index",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "length",
            ],
            filters=[("episode_index", "=", episode_idx)],
        )
        if t.num_rows > 0:
            ep_row = t.to_pandas().iloc[0]
            break
    if ep_row is None:
        raise ValueError(
            f"episode {episode_idx} not found in {root}/meta/episodes"
        )

    chunk = int(ep_row["data/chunk_index"])
    file_idx = int(ep_row["data/file_index"])
    from_idx = int(ep_row["dataset_from_index"])

    # 2) Read just the single starting frame from the data parquet.
    data_path = root / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"
    if not data_path.is_file():
        raise FileNotFoundError(f"Expected data parquet not found: {data_path}")

    t = pq.read_table(
        str(data_path),
        columns=["observation.state", "index"],
        filters=[("index", "=", from_idx)],
    )
    if t.num_rows == 0:
        raise ValueError(
            f"frame index {from_idx} not found in {data_path} "
            f"(episode {episode_idx}, dataset_from_index={from_idx})"
        )

    return np.asarray(
        t.column("observation.state").to_pylist()[0], dtype=np.float32
    )


def load_init_arm_pose_from_parquet(
    root: str,
    episode_idx: int,
    arm_dof: int = 14,
) -> np.ndarray:
    """Return the first frame's arm pose of `episode_idx`.

    Thin wrapper around `load_init_state_from_parquet` that slices the
    leading `arm_dof` dimensions.
    """
    return load_init_state_from_parquet(root, episode_idx)[:arm_dof]
