"""Plots and the summary table for the backpressure experiment.

Reads docs/backpressure/{A,B}_*.csv written by scripts/experiment.py, writes PNG files and
summary.md next to them.

    uv run --group train python scripts/plot_experiment.py
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR = Path(__file__).resolve().parent.parent / "docs" / "backpressure"
TITLES = {
    "A": "A: без механизмов (очередь без лимита)",
    "B": "B: с механизмами (лимит 50, reject-publish)",
}
WORKER_CONCURRENCY = 2
INFERENCE_SECONDS = 0.5
# Steady-state windows (seconds from the source start): 30 s are left after the start and after
# the second worker is launched, so model loading and the initial transient are excluded.
WINDOWS = {1: (30, 90), 2: (120, 180)}
COLORS = {"primary": "#2a6fbb", "secondary": "#d9822b", "third": "#4c9a5a", "grey": "#777777"}


def read(label: str, name: str) -> list[dict]:
    with (DIR / f"{label}_{name}.csv").open(encoding="utf-8") as handle:
        return [
            {k: float(v) if k != "container" else v for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def meta(label: str) -> dict:
    return json.loads((DIR / f"{label}_meta.json").read_text(encoding="utf-8"))


def per_second(finished: list[float], end: float) -> list[float]:
    counts = [0] * (int(end) + 1)
    for moment in finished:
        if 0 <= moment <= end:
            counts[int(moment)] += 1
    return counts


def rolling_median(
    xs: list[float], ys: list[float], width: float = 10.0
) -> tuple[list[float], list[float]]:
    out_x, out_y = [], []
    for center in range(0, int(max(xs)) + 1, 2):
        window = [
            y for x, y in zip(xs, ys, strict=True) if center - width / 2 <= x < center + width / 2
        ]
        if window:
            out_x.append(center)
            out_y.append(statistics.median(window))
    return out_x, out_y


def container_series(rows: list[dict], part: str) -> tuple[list[float], list[float]]:
    # Several workers are summed per timestamp bucket of 2 s (one docker stats call).
    buckets: dict[float, float] = {}
    for row in rows:
        if part in row["container"]:
            key = round(row["t"] / 2) * 2
            buckets[key] = buckets.get(key, 0) + row["mem_bytes"] / 2**20
    return list(buckets), list(buckets.values())


def figure(name: str, ylabel: str, draw, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for axis, label in zip(axes, "AB", strict=True):
        draw(axis, label)
        axis.axvline(meta(label)["scale_at_s"], color=COLORS["grey"], linestyle="--", linewidth=1.2)
        axis.annotate(
            "+1 воркер",
            (meta(label)["scale_at_s"], 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            color=COLORS["grey"],
            fontsize=9,
        )
        axis.set_title(TITLES[label], fontsize=11)
        axis.set_xlabel("время от старта источника, с")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.set_xlim(0, meta(label)["duration_s"])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(DIR / f"{name}.png", dpi=130)
    plt.close(fig)


def draw_depth(axis, label: str) -> None:
    rows = read(label, "queue")
    t = [r["t"] for r in rows]
    axis.plot(
        t, [r["messages_ready"] for r in rows], color=COLORS["primary"], label="ready (ожидают)"
    )
    axis.plot(
        t,
        [r["messages_unacknowledged"] for r in rows],
        color=COLORS["secondary"],
        label="unacked (в работе)",
    )
    if label == "B":
        axis.axhline(
            meta(label)["max_length"], color="#b03a3a", linestyle=":", label="x-max-length"
        )
    axis.legend(loc="best", fontsize=9)


def draw_memory(axis, label: str) -> None:
    rows = read(label, "queue")
    t = [r["t"] for r in rows]
    axis.plot(
        t,
        [r["node_mem_used_bytes"] / 2**20 for r in rows],
        color=COLORS["primary"],
        label="RabbitMQ: mem_used узла (Management API)",
    )
    axis.plot(
        t,
        [r["queue_memory_bytes"] / 2**20 for r in rows],
        color=COLORS["secondary"],
        label="RabbitMQ: память очереди",
    )
    containers = read(label, "containers")
    for part, color, text in (
        ("rabbitmq", COLORS["third"], "контейнер rabbitmq (docker stats)"),
        ("source", COLORS["grey"], "контейнер source"),
        ("worker", "#8a5fb0", "контейнеры worker, сумма"),
    ):
        x, y = container_series(containers, part)
        axis.plot(x, y, color=color, linestyle="--", label=text)
    axis.legend(loc="best", fontsize=8)


def draw_rate(axis, label: str) -> None:
    source = read(label, "source")
    tasks = read(label, "tasks")
    axis.plot(
        [r["t"] for r in source],
        [r["published_per_s"] for r in source],
        color=COLORS["primary"],
        label="источник: принято брокером, задач/с",
    )
    axis.plot(
        [r["t"] for r in source],
        [r["rejected_per_s"] for r in source],
        color=COLORS["secondary"],
        alpha=0.6,
        label="источник: отказов (nack), 1/с",
    )
    end = int(meta(label)["duration_s"])
    done = per_second([r["finished_t"] for r in tasks], end)
    axis.plot(range(len(done)), done, color=COLORS["third"], label="воркеры: DONE, задач/с")
    axis.set_yscale("symlog", linthresh=10)
    axis.legend(loc="best", fontsize=8)


def draw_time(axis, label: str) -> None:
    tasks = read(label, "tasks")
    x = [r["finished_t"] for r in tasks]
    system = [r["in_system_s"] for r in tasks]
    inference = [r["inference_s"] for r in tasks]
    axis.scatter(x, system, s=3, color=COLORS["primary"], alpha=0.25)
    axis.scatter(x, inference, s=3, color=COLORS["third"], alpha=0.25)
    axis.plot(
        *rolling_median(x, system),
        color=COLORS["primary"],
        label="в системе (finished - enqueued), медиана 10 с",
    )
    axis.plot(
        *rolling_median(x, inference),
        color=COLORS["third"],
        label="inference (finished - started), медиана 10 с",
    )
    if label == "A":
        axis.set_yscale("log")
    axis.legend(loc="best", fontsize=8)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def summary() -> str:
    lines = [
        "| Сценарий | Воркеры | Теория, задач/с | Источник, задач/с | Воркеры (DONE), задач/с | Макс. ready (лимит B: 50) | Макс. unacked | RabbitMQ mem_used, МиБ (мин–макс) | В системе, медиана, с | В системе, p95, с | inference, медиана, с |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label in "AB":
        queue, source, tasks = read(label, "queue"), read(label, "source"), read(label, "tasks")
        for workers, (start, end) in WINDOWS.items():
            q = [r for r in queue if start <= r["t"] <= end]
            published = [r for r in source if start <= r["t"] <= end]
            rate = (published[-1]["published_total"] - published[0]["published_total"]) / (
                published[-1]["t"] - published[0]["t"]
            )
            done = [r for r in tasks if start <= r["finished_t"] < end]
            waits = [r["in_system_s"] for r in done]
            ready = max(r["messages_ready"] for r in q)
            unacked = max(r["messages_unacknowledged"] for r in q)
            mem = [r["node_mem_used_bytes"] / 2**20 for r in q]
            lines.append(
                f"| {label} | {workers} | {workers * WORKER_CONCURRENCY / INFERENCE_SECONDS:.0f} | {rate:.1f} | "
                f"{len(done) / (end - start):.2f} | {ready:.0f} | {unacked:.0f} | {min(mem):.0f}–{max(mem):.0f} | "
                f"{statistics.median(waits):.2f} | {percentile(waits, 0.95):.2f} | "
                f"{statistics.median(r['inference_s'] for r in done):.2f} |"
            )
    totals = []
    for label in "AB":
        m, queue = meta(label), read(label, "queue")
        totals.append(
            f"- {label}: принято брокером {m['source_published_total']}, отказов {m['source_rejected_total']}, "
            f"обработано (DONE) за {m['duration_s']:.0f} с {m['tasks_done']}, "
            f"макс. ready за весь прогон {max(r['messages_ready'] for r in queue):.0f}, "
            f"макс. unacked {max(r['messages_unacknowledged'] for r in queue):.0f}, "
            f"в конце: ready={queue[-1]['messages_ready']:.0f}, unacked={queue[-1]['messages_unacknowledged']:.0f}"
        )
    return "\n".join(lines) + "\n\n" + "\n".join(totals) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    figure("queue_depth", "сообщений в очереди", draw_depth, "Глубина очереди")
    figure("memory", "МиБ", draw_memory, "Память")
    figure(
        "rates",
        "задач/с (symlog)",
        draw_rate,
        "Скорость источника и пропускная способность воркеров",
    )
    figure("task_time", "секунды", draw_time, "Время выполнения задачи")
    text = summary()
    (DIR / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
