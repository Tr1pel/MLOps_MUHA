import asyncio
import itertools
import logging
from datetime import UTC, datetime
from uuid import UUID

import pytest

import agent_platform.source as source
from agent_platform.core.config import Settings
from agent_platform.core.logging import logger
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


async def test_progress_is_logged_with_totals_and_rate(monkeypatch):
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture()
    monkeypatch.setattr(logger, "handlers", [handler])
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        await run(tasks(3), Recorder([QueueFullError()]), [], report_interval=0.0)
    finally:
        logger.setLevel(previous_level)

    assert records
    assert all(line.startswith("source_progress published_total=") for line in records)
    assert "rejected_total=1" in records[-1]
    assert "published_total=3" in records[-1]
    assert "rate_per_s=" in records[-1]


def test_parse_args_defaults_and_validation():
    args = source.parse_args(["--count", "30"])
    assert (args.count, args.start_seed, args.batch_id, args.max_steps) == (30, 0, "default", 200)
    with pytest.raises(SystemExit):
        source.parse_args(["--count", "1", "--max-steps", "5000"])
    with pytest.raises(SystemExit):
        source.parse_args(["--count", "5", "--start-seed", str(2**32 - 2)])
    with pytest.raises(SystemExit):
        source.parse_args([])


async def test_run_declares_topology_publishes_everything_and_closes_the_broker(monkeypatch):
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

    monkeypatch.setattr(source, "run", fake_run)
    code = source.main(["--count", "7", "--start-seed", "3", "--batch-id", "x", "--max-steps", "9"])
    assert code == 0
    assert seen == {"count": 7, "start_seed": 3, "batch_id": "x", "max_steps": 9}
