"""Custom vLLM V1 schedulers for MLS Lab 3.

Use with vLLM >= 0.8:

    vllm serve models/Qwen3-0.6B \
      --scheduler-cls my_scheduler.StepExclusivePrefillFirstScheduler

The implementations deliberately delegate KV-cache allocation, prefix caching,
preemption, and SchedulerOutput construction to vLLM's default scheduler.
Each strategy only changes which queues the default scheduler can see, or the
order of the waiting queue, before calling ``super().schedule()``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler


logger = logging.getLogger(__name__)


def _num_prompt_tokens(request: Any) -> int:
    if hasattr(request, "num_prompt_tokens"):
        return int(request.num_prompt_tokens)
    prompt_ids = getattr(request, "prompt_token_ids", None)
    if prompt_ids is not None:
        return len(prompt_ids)
    return int(getattr(request, "num_tokens", 0))


def _queue_items(queue: Any) -> list[Any]:
    return list(queue)


def _replace_queue_order(queue: Any, requests: Iterable[Any]) -> bool:
    """Best-effort queue reorder for vLLM FCFS/deque request queues.

    vLLM's FCFS queue subclasses ``deque`` and supports ``clear`` + ``extend``.
    Priority queues intentionally order by priority/arrival time; overriding
    their heap key externally would be brittle, so this helper leaves them
    untouched and returns False.
    """

    ordered = list(requests)
    if hasattr(queue, "clear") and hasattr(queue, "extend"):
        queue.clear()
        queue.extend(ordered)
        return True
    if hasattr(queue, "remove_requests") and hasattr(queue, "add_request"):
        try:
            existing = _queue_items(queue)
            queue.remove_requests(existing)
            for request in ordered:
                queue.add_request(request)
            return True
        except Exception:
            logger.exception("Failed to reorder waiting queue.")
    return False


def _scheduled_token_count(output: SchedulerOutput) -> int:
    return int(sum(output.num_scheduled_tokens.values()))


class PeakKVTraceMixin:
    """Mixin that writes a lightweight scheduler-side KV trace as JSONL.

    Set ``MLS_LAB3_KV_TRACE=/path/to/trace.jsonl`` before starting vLLM. Each
    scheduling step records the number of computed/runnable tokens in the
    running set and the largest value observed so far. The value is a token
    count; multiply by the model-specific per-token KV bytes in the report.
    """

    _lab3_step_id: int
    _lab3_peak_running_tokens: int

    def _init_trace_state(self) -> None:
        if not hasattr(self, "_lab3_step_id"):
            self._lab3_step_id = 0
            self._lab3_peak_running_tokens = 0

    def _record_kv_trace(self, output: SchedulerOutput, strategy: str) -> None:
        self._init_trace_state()
        self._lab3_step_id += 1
        running_tokens = 0
        for request in getattr(self, "running", []):
            running_tokens += int(getattr(request, "num_computed_tokens", 0))
        self._lab3_peak_running_tokens = max(
            self._lab3_peak_running_tokens, running_tokens
        )

        trace_path = os.environ.get("MLS_LAB3_KV_TRACE")
        if not trace_path:
            return
        record = {
            "ts": time.time(),
            "strategy": strategy,
            "step": self._lab3_step_id,
            "num_running": len(getattr(self, "running", [])),
            "num_waiting": len(getattr(self, "waiting", [])),
            "scheduled_tokens": _scheduled_token_count(output),
            "running_computed_tokens": running_tokens,
            "peak_running_computed_tokens": self._lab3_peak_running_tokens,
            "num_scheduled_tokens": dict(output.num_scheduled_tokens),
        }
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class StepExclusivePrefillFirstScheduler(PeakKVTraceMixin, Scheduler):
    """Strategy A: step-exclusive prefill-first.

    If the waiting queue is non-empty, hide all running decode requests for this
    step and let vLLM's parent scheduler schedule only waiting prefill requests.
    Otherwise, fall back to the default scheduler, which means the step is pure
    decode because no prefill is waiting.
    """

    def schedule(self) -> SchedulerOutput:
        if len(self.waiting) > 0:
            saved_running = self.running
            self.running = []
            output = super().schedule()
            if _scheduled_token_count(output) > 0:
                self.running = saved_running + self.running
                logger.info(
                    "[Lab3 Strategy A] prefill-only step scheduled %s tokens",
                    _scheduled_token_count(output),
                )
                self._record_kv_trace(output, "step_exclusive_prefill_first")
                return output

            # If prefill cannot make progress because the token/KV budget is
            # exhausted, restore running requests and let decode drain. Without
            # this fallback, waiting stays non-empty and the scheduler can spin
            # forever on zero-token prefill-only steps.
            self.running = saved_running
            output = super().schedule()
            logger.info(
                "[Lab3 Strategy A] decode fallback scheduled %s tokens",
                _scheduled_token_count(output),
            )
            self._record_kv_trace(output, "step_exclusive_prefill_first")
            return output

        output = super().schedule()
        logger.info(
            "[Lab3 Strategy A] decode-only step scheduled %s tokens",
            _scheduled_token_count(output),
        )
        self._record_kv_trace(output, "step_exclusive_prefill_first")
        return output


class SJFScheduler(PeakKVTraceMixin, Scheduler):
    """Strategy B1: shortest-job-first over the waiting prefill queue.

    Before every scheduling step, sort waiting requests by prompt length while
    preserving default vLLM behavior for running decode requests and KV-cache
    allocation. This improves TTFT for short prompts but can hurt long-prompt
    fairness under sustained load.
    """

    def schedule(self) -> SchedulerOutput:
        if len(self.waiting) > 1:
            sorted_waiting = sorted(
                _queue_items(self.waiting),
                key=lambda req: (_num_prompt_tokens(req), getattr(req, "arrival_time", 0.0)),
            )
            if not _replace_queue_order(self.waiting, sorted_waiting):
                logger.warning(
                    "[Lab3 SJF] waiting queue type %s could not be reordered; "
                    "falling back to default order.",
                    type(self.waiting),
                )

        output = super().schedule()
        logger.info(
            "[Lab3 SJF] scheduled %s tokens; waiting=%s running=%s",
            _scheduled_token_count(output),
            len(self.waiting),
            len(self.running),
        )
        self._record_kv_trace(output, "sjf")
        return output


class PureDecodeFirstScheduler(PeakKVTraceMixin, Scheduler):
    """Optional baseline: if decode is active, schedule only running requests."""

    def schedule(self) -> SchedulerOutput:
        if len(self.running) > 0:
            saved_waiting = self.waiting
            try:
                from vllm.v1.core.sched.request_queue import create_request_queue

                self.waiting = create_request_queue(self.policy)
                output = super().schedule()
            finally:
                self.waiting = saved_waiting
            logger.info(
                "[Lab3 PureDecodeFirst] decode-only step scheduled %s tokens",
                _scheduled_token_count(output),
            )
            self._record_kv_trace(output, "pure_decode_first")
            return output

        output = super().schedule()
        logger.info(
            "[Lab3 PureDecodeFirst] prefill step scheduled %s tokens",
            _scheduled_token_count(output),
        )
        self._record_kv_trace(output, "pure_decode_first")
        return output


class PeakKVTrackingScheduler(PeakKVTraceMixin, Scheduler):
    """Default vLLM scheduler plus JSONL KV tracing for Task 1 measurements."""

    def schedule(self) -> SchedulerOutput:
        output = super().schedule()
        self._record_kv_trace(output, "default_with_peak_kv_trace")
        return output
