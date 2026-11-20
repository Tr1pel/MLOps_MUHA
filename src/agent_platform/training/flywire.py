"""Pinned FlyWire v783 connectivity table and the small subgraph derived from it."""

import hashlib
import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)

REPO = "philshiu/Drosophila_brain_model"
COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
RAW_FILENAME = "Connectivity_783.parquet"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{RAW_FILENAME}"
FLYWIRE_VERSION = 783
DEFAULT_N_NEURONS = 400

EDGES_FILE = "subgraph_edges.parquet"
NEURONS_FILE = "subgraph_neurons.parquet"
METADATA_FILE = "metadata.json"

LICENSE_NOTE = (
    "Connectivity table from the FlyWire connectome (Dorkenwald et al. 2024, Nature; Schlegel "
    "et al. 2024, Nature), as packaged by Shiu et al. 2024 (Nature), repository "
    f"github.com/{REPO} (code under the MIT licence). The FlyWire data are subject to the "
    "FlyWire/Codex terms of use (flywire.ai); cite these papers when reusing the subgraph."
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_FILENAME
    if target.exists():
        logger.info("raw FlyWire table already present: %s", target)
        return target
    logger.info("downloading %s", RAW_URL)
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(RAW_URL, timeout=60) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.replace(target)
    return target


def select_subgraph(raw: pd.DataFrame, n_neurons: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-``n_neurons`` neurons by total synapse count (in + out), largest strongly connected
    component of their induced subgraph. Ranking by synaptic strength picks the hub neurons that
    are densely wired with each other, so activity injected anywhere keeps circulating.
    """
    pre_all = raw["Presynaptic_Index"].to_numpy()
    post_all = raw["Postsynaptic_Index"].to_numpy()
    n_all = int(max(pre_all.max(), post_all.max())) + 1
    syn = raw["Connectivity"].to_numpy(np.float64)
    strength = np.bincount(pre_all, weights=syn, minlength=n_all) + np.bincount(
        post_all, weights=syn, minlength=n_all
    )
    top = np.argsort(-strength, kind="stable")[:n_neurons]
    local = np.full(n_all, -1, dtype=np.int64)
    local[top] = np.arange(len(top))
    keep = (local[pre_all] >= 0) & (local[post_all] >= 0)
    pre, post = local[pre_all[keep]], local[post_all[keep]]
    graph = csr_matrix((np.ones(len(pre)), (pre, post)), shape=(len(top), len(top)))
    _, labels = connected_components(graph, directed=True, connection="strong")
    core = np.flatnonzero(labels == np.bincount(labels).argmax())
    remap = np.full(len(top), -1, dtype=np.int64)
    remap[core] = np.arange(len(core))
    in_core = (remap[pre] >= 0) & (remap[post] >= 0)

    root_id = np.zeros(n_all, dtype=np.int64)
    root_id[pre_all] = raw["Presynaptic_ID"].to_numpy()
    root_id[post_all] = raw["Postsynaptic_ID"].to_numpy()
    neurons = pd.DataFrame(
        {
            "index": np.arange(len(core), dtype=np.int32),
            "root_id": root_id[top[core]],
            "total_synapses": strength[top[core]].astype(np.int64),
        }
    )
    edges = pd.DataFrame(
        {
            "pre": remap[pre[in_core]].astype(np.int32),
            "post": remap[post[in_core]].astype(np.int32),
            "weight": raw["Excitatory x Connectivity"].to_numpy()[keep][in_core].astype(np.int32),
        }
    ).sort_values(["pre", "post"], ignore_index=True)
    return edges, neurons


def build_subgraph(
    data_dir: Path, n_neurons: int = DEFAULT_N_NEURONS, raw_dir: Path | None = None
) -> dict:
    """Download the pinned table (if needed), select the subgraph and write it to data_dir."""
    raw_path = download_raw(raw_dir or data_dir / "raw")
    raw_sha = sha256_of(raw_path)
    raw = pd.read_parquet(raw_path)
    edges, neurons = select_subgraph(raw, n_neurons)
    out_dir = data_dir / "flywire"
    out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(out_dir / EDGES_FILE, index=False, compression="zstd")
    neurons.to_parquet(out_dir / NEURONS_FILE, index=False, compression="zstd")
    n = len(neurons)
    raw_neurons = int(max(raw["Presynaptic_Index"].max(), raw["Postsynaptic_Index"].max()) + 1)
    metadata = {
        "source": "FlyWire connectome v783 connectivity table (Shiu et al. brain model)",
        "source_url": RAW_URL,
        "repository": f"https://github.com/{REPO}",
        "repository_commit": COMMIT,
        "raw_file": RAW_FILENAME,
        "raw_file_sha256": raw_sha,
        "raw_file_bytes": raw_path.stat().st_size,
        "raw_rows": len(raw),
        "raw_neurons": raw_neurons,
        "flywire_version": FLYWIRE_VERSION,
        "selection": {
            "method": "top neurons by total synapse count (in+out), largest strongly connected "
            "component of the induced subgraph",
            "n_candidates": n_neurons,
            "edge_weight": "signed synapse count: 'Excitatory x Connectivity' of the raw table",
        },
        "counts": {
            "neurons": n,
            "edges": len(edges),
            "excitatory_edges": int((edges["weight"] > 0).sum()),
            "inhibitory_edges": int((edges["weight"] < 0).sum()),
            "density": round(len(edges) / (n * n), 6),
            "synapses": int(edges["weight"].abs().sum()),
        },
        "files": {
            EDGES_FILE: sha256_of(out_dir / EDGES_FILE),
            NEURONS_FILE: sha256_of(out_dir / NEURONS_FILE),
        },
        "license_note": LICENSE_NOTE,
    }
    (out_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("subgraph written: %d neurons, %d edges", n, len(edges))
    return metadata


def subgraph_paths(data_dir: Path) -> tuple[Path, Path, Path]:
    base = data_dir / "flywire"
    return base / EDGES_FILE, base / NEURONS_FILE, base / METADATA_FILE


def load_metadata(data_dir: Path) -> dict:
    return json.loads(subgraph_paths(data_dir)[2].read_text(encoding="utf-8"))
