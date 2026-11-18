import asyncio
import itertools
from datetime import UTC, datetime
from uuid import UUID

import pytest
from conftest import log_records, metric_total
from opentelemetry.trace import StatusCode

import agent_platform.source as source
from agent_platform.core.config import Settings
from agent_platform.core.logging import task_id_context
from agent_platform.messaging import BrokerUnavailableError, QueueFullError


class Recorder:
    """Publish stand-in that fails with the queued errors, then succeeds; logs every call."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.attempts = []
        self.enqueued_at = []

    async def publish(self, task):
        self.attempts.append(task.seed)
        self.enqueued_at.append(task.enqueued_at)
        if self.errors:
            raise self.errors.pop(0)


async def run(tasks, recorder, sleeps, **kwargs):
    async def sleep(delay):
        sleeps.append(delay)

    return await source.run_source(iter(tasks), recorder.publish, sleep=sleep, **kwargs)


def tasks(count):
    return list(source.generate_tasks(count, 0, "b", 10))


def test_task_ids_are_deterministic_per_batch_and_seed():
    assert source.task_id_for("b", 1) == source.task_id_for("b", 1)
    assert source.task_id_for("b", 1) != source.task_id_for("b", 2)
    assert source.task_id_for("b", 1) != source.task_id_for("other", 1)
    assert isinstance(source.task_id_for("b", 1), UUID)
    first = [t.task_id for t in source.generate_tasks(3, 5, "b", 10)]
    again = [t.task_id for t in source.generate_tasks(3, 5, "b", 10)]
    assert first == again


def test_records_are_generated_lazily():
    generator = source.generate_tasks(10**9, 0, "b", 10)
    first = next(generator)
    assert (first.seed, first.max_steps) == (0, 10)
    assert next(generator).seed == 1


def test_backoff_grows_exponentially_with_cap_and_jitter():
    for failures, base in [(1, 0.05), (2, 0.1), (3, 0.2), (4, 0.4), (5, 0.8), (6, 1.0), (50, 1.0)]:
        samples = [source.backoff_delay(failures) for _ in range(50)]
        assert all(base <= s <= base * 1.2 for s in samples)
    assert len({source.backoff_delay(3) for _ in range(20)}) > 1


async def test_full_queue_retries_the_same_task_with_growing_backoff():
    recorder = Recorder([QueueFullError()] * 3)
    sleeps = []
    published, rejected = await run(tasks(2), recorder, sleeps)

    assert recorder.attempts == [0, 0, 0, 0, 1]  # same task until accepted, then the next
    assert (published, rejected) == (2, 3)
    assert len(sleeps) == 3
    assert sleeps[0] < sleeps[1] < sleeps[2]


async def test_enqueued_at_is_set_on_every_attempt_not_at_generation():
    old = datetime(2000, 1, 1, tzinfo=UTC)
    stale = [t.model_copy(update={"enqueued_at": old}) for t in tasks(1)]
    recorder = Recorder([QueueFullError()] * 2)
    await run(stale, recorder, [])

    assert len(recorder.enqueued_at) == 3
    assert all(moment > old for moment in recorder.enqueued_at)
    assert recorder.enqueued_at == sorted(recorder.enqueued_at)


async def test_backoff_is_capped_and_resets_after_a_success():
    recorder = Recorder()
    sleeps = []
    calls = itertools.count()

    async def publish(task):
        # Task 0 is rejected 8 times; task 1 is rejected once.
        recorder.attempts.append(task.seed)
        index = next(calls)
        if index < 8 or index == 9:
            raise QueueFullError()

    async def sleep(delay):
        sleeps.append(delay)

    await source.run_source(iter(tasks(2)), publish, sleep=sleep)

    assert max(sleeps) <= source.BACKOFF_CAP_SECONDS * 1.2
    assert sleeps[7] >= source.BACKOFF_CAP_SECONDS
    assert sleeps[8] < 0.07  # back to the base delay for the next task


async def test_unavailable_broker_is_retried_like_a_full_queue():
    recorder = Recorder([BrokerUnavailableError(), QueueFullError()])
    sleeps = []
    published, rejected = await run(tasks(1), recorder, sleeps)
    assert (published, rejected) == (1, 2)
    assert len(sleeps) == 2


async def test_no_sleep_when_every_publish_succeeds():
    recorder = Recorder()
    sleeps = []
    assert await run(tasks(5), recorder, sleeps) == (5, 0)
    assert sleeps == []


async def test_unexpected_errors_are_not_swallowed():
    recorder = Recorder([ValueError("bug")])
    with pytest.raises(ValueError):
        await run(tasks(1), recorder, [])


async def test_progress_is_logged_with_totals_and_rate_as_fields(log_output):
    await run(tasks(3), Recorder([QueueFullError()]), [], report_interval=0.0)

    progress = [r for r in log_records(log_output) if r["event"] == "source_progress"]
    assert progress
    last = progress[-1]
    assert (last["published_total"], last["rejected_total"]) == (3, 1)
    assert last["rate_per_s"] >= 0
    assert last["level"] == "INFO"


async def test_each_rejection_is_a_warning_with_attempt_and_delay(log_output):
    sleeps = []
    await run(tasks(1), Recorder([QueueFullError(), BrokerUnavailableError()]), sleeps)

    rejected = [r for r in log_records(log_output) if r["event"] == "publish_rejected"]
    assert [r["level"] for r in rejected] == ["WARNING", "WARNING"]
    assert [r["attempt"] for r in rejected] == [1, 2]
    assert [r["reason"] for r in rejected] == ["queue_full", "unavailable"]
    assert [r["delay_s"] for r in rejected] == [round(d, 3) for d in sleeps]
    assert all(r["task_id"] and r["trace_id"] for r in rejected)


async def test_one_trace_per_task_with_rejection_and_backoff_events(spans):
    sleeps = []
    await run(tasks(3), Recorder([QueueFullError(), QueueFullError()]), sleeps)

    roots = [s for s in spans.get_finished_spans() if s.name == "source.publish_task"]
    assert len(roots) == 3
    assert len({s.context.trace_id for s in roots}) == 3  # never one trace for the batch
    assert all(s.parent is None for s in roots)
    first = roots[0]
    assert first.attributes["task.seed"] == 0
    assert first.attributes["task.id"] == str(source.task_id_for("b", 0))
    events = [(e.name, dict(e.attributes)) for e in first.events]
    assert [name for name, _ in events] == ["broker_rejected", "backoff"] * 2
    assert events[0][1] == {"attempt": 1, "reason": "queue_full"}
    assert events[1][1]["attempt"] == 1
    assert events[1][1]["delay_s"] == pytest.approx(sleeps[0])
    assert events[3][1]["attempt"] == 2
    assert not roots[1].events
    assert task_id_context.get() is None


async def test_metrics_count_published_rejected_and_backoff_seconds():
    names = ("source.tasks.published", "source.publish.rejected", "source.backoff.duration")
    before = [metric_total(name) for name in names]
    sleeps = []
    await run(tasks(2), Recorder([QueueFullError(), BrokerUnavailableError()]), sleeps)
    after = [metric_total(name) for name in names]

    assert [a - b for a, b in zip(after, before, strict=True)] == [
        2,
        2,
        pytest.approx(sum(sleeps)),
    ]


async def test_a_failed_publish_that_is_not_backpressure_ends_the_span_with_error(spans):
    with pytest.raises(ValueError):
        await run(tasks(1), Recorder([ValueError("bug")]), [])
    [root] = [s for s in spans.get_finished_spans() if s.name == "source.publish_task"]
    assert root.status.status_code is StatusCode.ERROR


def test_parse_args_defaults_and_validation():
    args = source.parse_args(["--count", "30"])
    assert (args.count, args.start_seed, args.batch_id, args.max_steps) == (30, 0, "default", 200)
    with pytest.raises(SystemExit):
        source.parse_args(["--count", "1", "--max-steps", "5000"])
    with pytest.raises(SystemExit):
        source.parse_args(["--count", "5", "--start-seed", str(2**32 - 2)])
    with pytest.raises(SystemExit):
        source.parse_args([])


async def test_run_declares_topology_publishes_everything_and_closes_the_broker(
    monkeypatch, log_output
):
    events = []
    published = []

    class FakeBroker:
        async def start(self):
            events.append("start")

        async def stop(self):
            events.append("stop")

    async def fake_topology(broker, settings):
        events.append("topology")

    async def fake_publish(broker, settings, message):
        published.append(message.seed)

    monkeypatch.setattr(source, "make_broker", lambda settings: FakeBroker())
    monkeypatch.setattr(source, "declare_topology", fake_topology)
    monkeypatch.setattr(source, "publish_task", fake_publish)

    code = await source.run(
        Settings(_env_file=None), count=4, start_seed=10, batch_id="b", max_steps=5
    )

    assert code == 0
    assert published == [10, 11, 12, 13]
    assert events == ["start", "topology", "stop"]
    done = [r for r in log_records(log_output) if r["event"] == "source_done"]
    assert (done[0]["published_total"], done[0]["rejected_total"]) == (4, 0)


async def test_run_closes_the_broker_when_interrupted(monkeypatch):
    events = []

    class FakeBroker:
        async def start(self):
            raise asyncio.CancelledError

        async def stop(self):
            events.append("stop")

    monkeypatch.setattr(source, "make_broker", lambda settings: FakeBroker())
    code = await source.run(
        Settings(_env_file=None), count=1, start_seed=0, batch_id="b", max_steps=5
    )
    assert code == 130
    assert events == ["stop"]


def test_main_passes_arguments_to_run(monkeypatch):
    seen = {}

    async def fake_run(settings, count, start_seed, batch_id, max_steps):
        seen.update(count=count, start_seed=start_seed, batch_id=batch_id, max_steps=max_steps)
        return 0

    flushed = []
    monkeypatch.setattr(source, "run", fake_run)
    monkeypatch.setattr(source, "shutdown_telemetry", lambda: flushed.append(True))
    code = source.main(["--count", "7", "--start-seed", "3", "--batch-id", "x", "--max-steps", "9"])
    assert code == 0
    assert flushed == [True]  # buffered spans and metrics leave before the process exits
    assert seen == {"count": 7, "start_seed": 3, "batch_id": "x", "max_steps": 9}
