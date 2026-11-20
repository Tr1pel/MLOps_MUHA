"""EDA run: class balance, features, episodes and the FlyWire subgraph; logged to MLflow."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
from agent_platform.ml.features import ConnectomeFeatures, load_edges
from agent_platform.sim.env import ACTIONS, ENV_VERSION
from agent_platform.sim.episode import run_episode
from agent_platform.sim.teacher import Teacher
from agent_platform.training import dataset as ds
from agent_platform.training import flywire, tracking
from agent_platform.training.config import PipelineConfig
from agent_platform.training.plots import (
    ACTION_COLORS,
    INK_SECONDARY,
    SERIES,
    trajectory_grid,
)

logger = logging.getLogger(__name__)

ACTIVE_THRESHOLD = 0.05
SATURATED_THRESHOLD = 0.95
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])


def class_balance(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    counts = {}
    for split, frame in (("train", train), ("test", test)):
        c = frame[ds.TARGET].value_counts().reindex(range(len(ACTIONS)), fill_value=0)
        counts[split] = {name: int(c[i]) for i, name in enumerate(ACTIONS)}
    total = counts["train"]
    shares = {k: v / sum(total.values()) for k, v in total.items()}
    return {
        "counts": counts,
        "train_shares": shares,
        "imbalance_ratio": max(shares.values()) / max(min(shares.values()), 1e-12),
    }


def fig_class_balance(balance: dict):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    width = 0.38
    x = np.arange(len(ACTIONS))
    for k, (split, hatch) in enumerate((("train", None), ("test", "//"))):
        values = np.array([balance["counts"][split][a] for a in ACTIONS], dtype=float)
        shares = values / values.sum()
        bars = ax.bar(
            x + (k - 0.5) * width,
            shares,
            width,
            color=[ACTION_COLORS[a] for a in ACTIONS],
            hatch=hatch,
            edgecolor="white",
            label=split,
        )
        for bar, share in zip(bars, shares, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, share + 0.01, f"{share:.0%}", ha="center")
    ax.set_xticks(x, ACTIONS)
    ax.set_ylabel("share of rows")
    ax.set_title("Teacher action balance (solid = train, hatched = test)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_feature_distributions(train: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=(10, 7))
    for ax, name in zip(axes.ravel(), ds.FEATURE_NAMES, strict=True):
        bins = np.linspace(train[name].min(), train[name].max(), 31)
        for i, action in enumerate(ACTIONS):
            ax.hist(
                train.loc[train[ds.TARGET] == i, name],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=ACTION_COLORS[action],
                label=action,
            )
        ax.set_title(name)
        ax.set_yticks([])
    axes[0, 0].legend()
    fig.suptitle("Observation features by teacher action (density)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_correlation(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.grid(False)
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation (Pearson, train)")
    fig.tight_layout()
    return fig


def fig_episodes(episodes: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    bins = np.linspace(0, episodes["steps"].max() + 1, 31)
    ok = episodes[episodes["success"]]
    bad = episodes[~episodes["success"]]
    axes[0].hist(
        [ok["steps"], bad["steps"]],
        bins=bins,
        stacked=True,
        color=[SERIES[0], SERIES[1]],
        label=["reached food", "step limit"],
    )
    axes[0].set_xlabel("episode length (steps)")
    axes[0].set_ylabel("episodes")
    axes[0].legend()
    axes[0].set_title("Teacher episode length")
    counts = episodes["collisions"].clip(upper=10).value_counts().sort_index()
    axes[1].bar(counts.index, counts.to_numpy(), color=SERIES[0])
    axes[1].set_xlabel("blocked forward moves per episode (10 = 10 or more)")
    axes[1].set_title("Teacher collisions")
    fig.tight_layout()
    return fig


def fig_graph(edges: pd.DataFrame, n: int):
    out_deg = np.bincount(edges["pre"], minlength=n)
    in_deg = np.bincount(edges["post"], minlength=n)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    bins = np.linspace(0, max(out_deg.max(), in_deg.max()), 31)
    axes[0].hist(in_deg, bins=bins, histtype="step", linewidth=1.6, color=SERIES[0], label="in")
    axes[0].hist(out_deg, bins=bins, histtype="step", linewidth=1.6, color=SERIES[1], label="out")
    axes[0].set_xlabel("degree (distinct partners)")
    axes[0].set_ylabel("neurons")
    axes[0].legend()
    axes[0].set_title("Degree distribution")
    w = edges["weight"].to_numpy()
    axes[1].hist(
        np.sign(w) * np.log10(np.abs(w)),
        bins=40,
        color=INK_SECONDARY,
    )
    axes[1].set_xlabel("sign * log10(synapse count)")
    axes[1].set_title("Signed edge weights")
    axes[2].bar(
        ["excitatory", "inhibitory"],
        [int((w > 0).sum()), int((w < 0).sum())],
        color=[SERIES[0], SERIES[1]],
    )
    axes[2].set_title("Edge signs")
    axes[2].grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def propagation_profile(
    train: pd.DataFrame, edges: pd.DataFrame, cfg: PipelineConfig, edges_path: Path
) -> dict:
    """How far, how strongly and how usefully activity spreads over 1..MAX steps."""
    features = list(ds.FEATURE_NAMES)
    seeds = np.sort(train["episode_seed"].unique())
    half = set(seeds[::2].tolist())
    fit_rows = train[train["episode_seed"].isin(half)].head(cfg.eda_rows)
    val_rows = train[~train["episode_seed"].isin(half)].head(cfg.eda_rows // 2)
    conn = ConnectomeFeatures(
        graph_path=str(edges_path),
        n_steps=cfg.eda_max_steps,
        n_input=cfg.n_input,
        seed=cfg.model_seed,
    ).fit(fit_rows[features].to_numpy())
    history = conn.propagate(fit_rows[features].to_numpy())
    val_history = conn.propagate(val_rows[features].to_numpy())

    n = conn.weights_.shape[0]
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[edges["pre"], edges["post"]] = True
    reached = np.zeros(n, dtype=bool)
    reached[conn.input_neurons_] = True

    profile = []
    previous = np.zeros_like(history[0])
    for t in range(cfg.eda_max_steps):
        activity = history[t]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        clf.fit(activity, fit_rows[ds.TARGET])
        decoded = f1_score(
            val_rows[ds.TARGET], clf.predict(val_history[t]), average="macro", labels=[0, 1, 2]
        )
        profile.append(
            {
                "step": t + 1,
                "graph_reach_fraction": float(reached.mean()),
                "active_fraction": float((np.abs(activity) > ACTIVE_THRESHOLD).mean()),
                "saturated_fraction": float((np.abs(activity) > SATURATED_THRESHOLD).mean()),
                "mean_abs_activity": float(np.abs(activity).mean()),
                "change_from_previous": float(np.abs(activity - previous).mean()),
                "decoder_f1_macro_train_cv": float(decoded),
            }
        )
        previous = activity
        reached = reached | adjacency[reached].any(axis=0)
    return {"input_neurons": int(len(conn.input_neurons_)), "per_step": profile}


def fig_propagation(profile: dict):
    steps = [p["step"] for p in profile["per_step"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for key, label, color in (
        ("graph_reach_fraction", "reached by signal", SERIES[0]),
        ("active_fraction", f"|a| > {ACTIVE_THRESHOLD}", SERIES[1]),
        ("saturated_fraction", f"|a| > {SATURATED_THRESHOLD}", SERIES[2]),
    ):
        axes[0].plot(
            steps, [p[key] for p in profile["per_step"]], marker="o", color=color, label=label
        )
    axes[0].set_xlabel("propagation step")
    axes[0].set_ylabel("fraction of neurons")
    axes[0].legend()
    axes[0].set_title("Spread and saturation")
    axes[1].plot(
        steps, [p["change_from_previous"] for p in profile["per_step"]], marker="o", color=SERIES[0]
    )
    axes[1].set_xlabel("propagation step")
    axes[1].set_ylabel("mean |a_t - a_(t-1)|")
    axes[1].set_title("Convergence")
    axes[2].plot(
        steps,
        [p["decoder_f1_macro_train_cv"] for p in profile["per_step"]],
        marker="o",
        color=SERIES[0],
    )
    axes[2].set_xlabel("propagation step")
    axes[2].set_ylabel("macro F1 (held-out train episodes)")
    axes[2].set_title("Linear decoder on activity")
    fig.tight_layout()
    return fig


def graph_stats(edges: pd.DataFrame, n: int) -> dict:
    out_deg = np.bincount(edges["pre"], minlength=n)
    in_deg = np.bincount(edges["post"], minlength=n)
    w = edges["weight"].to_numpy()
    return {
        "neurons": n,
        "edges": len(edges),
        "density": len(edges) / (n * n),
        "mean_degree": float(out_deg.mean()),
        "max_in_degree": int(in_deg.max()),
        "max_out_degree": int(out_deg.max()),
        "excitatory_share": float((w > 0).mean()),
        "median_abs_weight": float(np.median(np.abs(w))),
        "max_abs_weight": int(np.abs(w).max()),
    }


def sample_trajectories(episodes: pd.DataFrame, n: int = 6) -> list:
    """Two failures (if any) and successes, re-simulated with the teacher for plotting."""
    failed = episodes.loc[~episodes["success"], "episode_seed"].head(n // 3).tolist()
    ok = episodes.loc[episodes["success"], "episode_seed"].head(n - len(failed)).tolist()
    teacher = Teacher()
    return [(run_episode(teacher, int(s)), None) for s in ok + failed]


def build_summary(balance, corr, episodes, gstats, profile, cfg) -> dict:
    feature_corr = corr.loc[list(ds.FEATURE_NAMES), list(ds.FEATURE_NAMES)].to_numpy()
    pairs = [
        (ds.FEATURE_NAMES[i], ds.FEATURE_NAMES[j], float(feature_corr[i, j]))
        for i in range(len(ds.FEATURE_NAMES))
        for j in range(i + 1, len(ds.FEATURE_NAMES))
        if abs(feature_corr[i, j]) >= 0.5
    ]
    failed = episodes[~episodes["success"]]
    steps = profile["per_step"]
    spread_step = next((p["step"] for p in steps if p["graph_reach_fraction"] >= 0.99), None)
    best = max(steps, key=lambda p: p["decoder_f1_macro_train_cv"])
    return {
        "env_version": ENV_VERSION,
        "class_balance": balance,
        "strong_feature_correlations": pairs,
        "episodes": {
            "n": len(episodes),
            "teacher_success_rate": float(episodes["success"].mean()),
            "mean_steps": float(episodes["steps"].mean()),
            "mean_collisions": float(episodes["collisions"].mean()),
            "failed_episodes": len(failed),
            "failed_with_collisions": int((failed["collisions"] > 0).sum()),
        },
        "flywire_graph": gstats,
        "propagation": {
            **profile,
            "spread_step_99pct": spread_step,
            "best_decoder_step": best["step"],
        },
        "configured_n_steps": list(cfg.n_steps),
    }


def summary_markdown(s: dict) -> str:
    bal, ep, g = s["class_balance"], s["episodes"], s["flywire_graph"]
    shares = bal["train_shares"]
    major = max(shares, key=shares.get)
    minor = min(shares, key=shares.get)
    prop = s["propagation"]
    steps = prop["per_step"]
    cfg_steps = ", ".join(str(k) for k in s["configured_n_steps"])
    lines = [
        "# EDA summary (generated)",
        "",
        "## Behaviour-cloning dataset",
        "- Action shares in train: "
        + ", ".join(f"{k} {v:.1%}" for k, v in shares.items())
        + f" (imbalance ratio {bal['imbalance_ratio']:.1f}). "
        f"'{major}' dominates and '{minor}' is rarest, so plain accuracy rewards ignoring the rare "
        "turns; the primary metric is macro F1 (every action counts equally).",
        f"- Teacher: success {ep['teacher_success_rate']:.1%} over {ep['n']} episodes, mean "
        f"{ep['mean_steps']:.1f} steps, {ep['mean_collisions']:.2f} blocked moves per episode. "
        f"{ep['failed_episodes']} episodes hit the step limit "
        f"({ep['failed_with_collisions']} of them with blocked moves), and they contribute "
        "up to the full step limit of rows each, so they weigh more in the data than their count "
        "suggests.",
    ]
    if s["strong_feature_correlations"]:
        pairs = "; ".join(f"{a}~{b} r={r:+.2f}" for a, b, r in s["strong_feature_correlations"])
        lines.append(f"- Feature pairs with |r| >= 0.5: {pairs}.")
    else:
        lines.append("- No feature pair has |r| >= 0.5.")
    lines += [
        "",
        "## FlyWire subgraph",
        f"- {g['neurons']} neurons, {g['edges']} directed edges (density {g['density']:.3f}), mean "
        f"degree {g['mean_degree']:.1f}, max in/out degree {g['max_in_degree']}/{g['max_out_degree']}.",
        f"- {g['excitatory_share']:.1%} of edges are excitatory; synapse counts are heavy-tailed "
        f"(median {g['median_abs_weight']:.0f}, max {g['max_abs_weight']}), so ConnectomeFeatures "
        "rescales the signed matrix to a fixed spectral radius before propagation.",
        f"- Signal injected into {prop['input_neurons']} input neurons reaches 99% of the graph by "
        f"step {prop['spread_step_99pct']}. Active fraction (|a|>{ACTIVE_THRESHOLD}) goes "
        f"{steps[0]['active_fraction']:.2f} -> {steps[1]['active_fraction']:.2f} -> "
        f"{steps[-1]['active_fraction']:.2f} (steps 1, 2, {steps[-1]['step']}); saturated fraction "
        f"at the last step is {steps[-1]['saturated_fraction']:.2f}.",
        f"- Step-to-step change is {steps[1]['change_from_previous']:.3f} at step 2 and "
        f"{steps[-1]['change_from_previous']:.3f} at step {steps[-1]['step']}.",
        f"- A linear decoder on activity (fit/validated on disjoint train episodes only) peaks at "
        f"step {prop['best_decoder_step']} with macro F1 "
        f"{max(p['decoder_f1_macro_train_cv'] for p in steps):.3f}.",
        f"- Compared propagation depths: n_steps = {cfg_steps}. Active fraction at those steps: "
        + ", ".join(f"{steps[k - 1]['active_fraction']:.2f}" for k in s["configured_n_steps"])
        + "; decoder macro F1: "
        + ", ".join(
            f"{steps[k - 1]['decoder_f1_macro_train_cv']:.3f}" for k in s["configured_n_steps"]
        )
        + ".",
    ]
    return "\n".join(lines) + "\n"


def run_eda(cfg: PipelineConfig) -> str:
    gen_run_id = tracking.read_generation_run_id(cfg.work_dir)
    train, test = ds.load_split(cfg.work_dir)
    episodes = pd.read_parquet(cfg.work_dir / ds.EPISODES_FILE)
    edges_path, _, _ = flywire.subgraph_paths(cfg.data_dir)
    edges = load_edges(edges_path)
    fly_meta = flywire.load_metadata(cfg.data_dir)
    n = fly_meta["counts"]["neurons"]

    balance = class_balance(train, test)
    corr = train[[*ds.FEATURE_NAMES, ds.TARGET]].corr()
    gstats = graph_stats(edges, n)
    profile = propagation_profile(train, edges, cfg, edges_path)
    summary = build_summary(balance, corr, episodes, gstats, profile, cfg)

    mlflow.set_experiment(cfg.data_experiment)
    with mlflow.start_run(run_name="eda") as run:
        mlflow.set_tags({"stage": "eda", "generation_run_id": gen_run_id})
        mlflow.log_input(tracking.generated_dataset(train, "train", gen_run_id), context="eda")
        mlflow.log_input(
            tracking.flywire_dataset(edges),
            context="eda",
            tags=tracking.flywire_input_tags(fly_meta),
        )
        mlflow.log_dict(fly_meta, "flywire/metadata.json")
        figures = {
            "class_balance.png": fig_class_balance(balance),
            "feature_distributions.png": fig_feature_distributions(train),
            "feature_correlation.png": fig_correlation(corr),
            "teacher_episodes.png": fig_episodes(episodes),
            "teacher_trajectories.png": trajectory_grid(
                sample_trajectories(episodes), "Teacher trajectories (successes, then failures)"
            ),
            "flywire_graph.png": fig_graph(edges, n),
            "flywire_propagation.png": fig_propagation(profile),
        }
        for name, fig in figures.items():
            mlflow.log_figure(fig, f"eda/{name}")
            plt.close(fig)
        mlflow.log_dict(summary, "eda/eda_summary.json")
        mlflow.log_text(summary_markdown(summary), "eda/eda_summary.md")
        mlflow.log_dict({"per_step": profile["per_step"]}, "eda/propagation_profile.json")
        mlflow.log_params({"propagation.max_steps": cfg.eda_max_steps, "env_version": ENV_VERSION})
        metrics = {f"share_{k}": v for k, v in balance["train_shares"].items()}
        metrics["class_imbalance_ratio"] = balance["imbalance_ratio"]
        metrics["teacher_success_rate"] = summary["episodes"]["teacher_success_rate"]
        metrics["teacher_mean_steps"] = summary["episodes"]["mean_steps"]
        metrics["graph_density"] = gstats["density"]
        metrics["graph_excitatory_share"] = gstats["excitatory_share"]
        mlflow.log_metrics(metrics)
        for p in profile["per_step"]:
            for key in (
                "active_fraction",
                "saturated_fraction",
                "graph_reach_fraction",
                "decoder_f1_macro_train_cv",
            ):
                mlflow.log_metric(f"propagation_{key}", p[key], step=p["step"])
        logger.info("EDA run %s", run.info.run_id)
        return run.info.run_id
