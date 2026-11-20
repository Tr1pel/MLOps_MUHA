"""Shared matplotlib helpers (non-interactive backend, one fixed colour per action)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from agent_platform.sim.env import ACTIONS, ARENA_SIZE  # noqa: E402
from agent_platform.sim.episode import EpisodeResult  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots 1-3 in fixed order: forward, left, right.
ACTION_COLORS = {"forward": "#2a78d6", "left": "#eb6834", "right": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQUENTIAL = "Blues"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "font.size": 9,
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.dpi": 130,
        }
    )


apply_style()


def draw_arena(ax, result: EpisodeResult) -> None:
    ax.set_xlim(0, ARENA_SIZE)
    ax.set_ylim(0, ARENA_SIZE)
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
    for x, y, r in result.obstacles:
        ax.add_patch(Circle((x, y), r, facecolor="#c9c8c2", edgecolor="none"))
    ax.add_patch(Circle(result.food, 0.4, facecolor="#eda100", edgecolor="none"))
    ax.set_xticks([])
    ax.set_yticks([])


def draw_trajectory(ax, result: EpisodeResult, color: str, label: str | None = None) -> None:
    xs, ys = zip(*result.trajectory, strict=True)
    ax.plot(xs, ys, color=color, linewidth=1.4, label=label)
    ax.plot(xs[0], ys[0], "o", color=color, markersize=5)


def trajectory_grid(episodes: list[tuple[EpisodeResult, EpisodeResult | None]], title: str):
    """Small multiples; each panel one seed: teacher (grey) and optionally a model (blue)."""
    n = len(episodes)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (first, second) in zip(axes.ravel(), episodes, strict=False):
        ax.axis("on")
        draw_arena(ax, first)
        if second is None:
            draw_trajectory(ax, first, SERIES[0])
            ax.set_title(
                f"seed {first.seed}: {'ok' if first.success else 'FAIL'}, {first.steps} steps",
                fontsize=8,
            )
        else:
            draw_trajectory(ax, first, INK_SECONDARY, "teacher")
            draw_trajectory(ax, second, SERIES[0], "model")
            ax.set_title(
                f"seed {first.seed}\nteacher {first.steps} steps | "
                f"model {'ok' if second.success else 'FAIL'} {second.steps} steps",
                fontsize=8,
            )
    handles = [
        plt.Line2D([], [], color=INK_SECONDARY, label="teacher" if episodes[0][1] else "agent"),
    ]
    if episodes[0][1] is not None:
        handles.append(plt.Line2D([], [], color=SERIES[0], label="model"))
    fig.legend(handles=handles, loc="upper right", ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def confusion_figure(matrix: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.grid(False)
    ax.imshow(matrix, cmap=SEQUENTIAL)
    ax.set_xticks(range(len(ACTIONS)), ACTIONS)
    ax.set_yticks(range(len(ACTIONS)), ACTIONS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("teacher action")
    top = matrix.max() if matrix.size else 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 0.55 * top else INK
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)
    ax.set_title(title)
    fig.tight_layout()
    return fig
