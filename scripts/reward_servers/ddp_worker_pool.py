# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small DDP worker pool for reward HTTP servers."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import socket
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist

WorkerFactory = Callable[..., Any]


class DDPWorkerPool:
    """Run one long-lived DDP worker process per visible CUDA device."""

    def __init__(
        self,
        *,
        worker_factory: WorkerFactory,
        worker_kwargs: dict[str, Any],
        world_size: int,
        batch_keys: Iterable[str],
        logger: logging.Logger,
        startup_timeout_s: float = 600.0,
        compute_timeout_s: float = 1800.0,
        shutdown_timeout_s: float = 30.0,
    ) -> None:
        if world_size <= 1:
            raise ValueError(f"DDP worker pool requires world_size > 1, got {world_size}.")

        self.worker_factory = worker_factory
        self.worker_kwargs = worker_kwargs
        self.world_size = world_size
        self.batch_keys = set(batch_keys)
        self.logger = logger
        self.startup_timeout_s = startup_timeout_s
        self.compute_timeout_s = compute_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self.device_ids = list(range(world_size))

        self._ctx = mp.get_context("spawn")
        self._request_queues: list[Any] = []
        self._response_queue: Any = None
        self._processes: list[mp.Process] = []
        self._lifecycle_lock = threading.Lock()
        self._compute_lock = threading.Lock()
        self._started = False

    @property
    def started(self) -> bool:
        """Whether the worker processes are currently running."""
        return self._started and all(process.is_alive() for process in self._processes)

    def start(self) -> None:
        """Start workers and wait until all ranks have loaded their models."""
        with self._lifecycle_lock:
            if self.started:
                return
            self._shutdown_locked(terminate_only=True)

            init_method = f"tcp://127.0.0.1:{_find_free_port()}"
            self._response_queue = self._ctx.Queue()
            self._request_queues = [self._ctx.Queue() for _ in range(self.world_size)]
            self._processes = []

            for local_rank, request_queue in enumerate(self._request_queues):
                process = self._ctx.Process(
                    target=_worker_loop,
                    kwargs={
                        "local_rank": local_rank,
                        "world_size": self.world_size,
                        "init_method": init_method,
                        "request_queue": request_queue,
                        "response_queue": self._response_queue,
                        "worker_factory": self.worker_factory,
                        "worker_kwargs": self.worker_kwargs,
                    },
                    daemon=True,
                )
                process.start()
                self._processes.append(process)

            ready_ranks: set[int] = set()
            deadline = time.monotonic() + self.startup_timeout_s
            while len(ready_ranks) < self.world_size:
                if time.monotonic() > deadline:
                    self._shutdown_locked(terminate_only=True)
                    raise TimeoutError(
                        f"Timed out waiting for DDP workers to start: "
                        f"ready={sorted(ready_ranks)} world_size={self.world_size}."
                    )

                for rank, process in enumerate(self._processes):
                    if rank not in ready_ranks and not process.is_alive():
                        self._shutdown_locked(terminate_only=True)
                        raise RuntimeError(
                            f"DDP worker rank {rank} exited during startup "
                            f"with code {process.exitcode}."
                        )

                try:
                    message = self._response_queue.get(timeout=0.25)
                except queue.Empty:
                    continue

                kind = message.get("kind")
                rank = int(message.get("rank", -1))
                if kind == "ready":
                    ready_ranks.add(rank)
                elif kind == "startup_error":
                    self._shutdown_locked(terminate_only=True)
                    raise RuntimeError(
                        f"DDP worker rank {rank} failed to start: "
                        f"{message.get('error')}\n{message.get('traceback')}"
                    )
                else:
                    self.logger.warning("Ignoring unexpected DDP startup message: %s", message)

            self._started = True
            self.logger.info(
                "DDP workers ready: world_size=%s device_ids=%s",
                self.world_size,
                self.device_ids,
            )

    def shutdown(self) -> None:
        """Stop all workers and release their CUDA contexts."""
        with self._lifecycle_lock:
            self._shutdown_locked(terminate_only=False)

    def close(self) -> None:
        """Alias for shutdown for server teardown paths."""
        self.shutdown()

    def compute(self, *, payload: dict[str, Any], total_size: int) -> list[float]:
        """Compute a full request by splitting it across DDP workers."""
        if total_size <= 0:
            return []

        self.start()
        request_id = uuid.uuid4().hex
        shards = _build_shards(total_size=total_size, world_size=self.world_size)
        rewards: list[Optional[float]] = [None] * total_size

        with self._compute_lock:
            for rank, source_indices, result_indices in shards:
                self._request_queues[rank].put(
                    {
                        "cmd": "compute",
                        "request_id": request_id,
                        "payload": _slice_payload(payload, self.batch_keys, source_indices),
                        "result_indices": result_indices,
                    }
                )

            remaining = self.world_size
            first_error: Optional[RuntimeError] = None
            deadline = time.monotonic() + self.compute_timeout_s
            while remaining > 0:
                try:
                    message = self._response_queue.get(timeout=0.5)
                except queue.Empty:
                    for rank, process in enumerate(self._processes):
                        if not process.is_alive():
                            raise RuntimeError(
                                f"DDP worker rank {rank} exited during request {request_id} "
                                f"with code {process.exitcode}."
                            )
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"Timed out waiting for DDP request {request_id}: "
                            f"remaining_workers={remaining}."
                        )
                    continue

                if message.get("kind") != "result" or message.get("request_id") != request_id:
                    self.logger.warning(
                        "Ignoring unexpected DDP compute message for request %s: %s",
                        request_id,
                        message,
                    )
                    continue

                remaining -= 1
                rank = int(message["rank"])
                error = message.get("error")
                if error is not None:
                    if first_error is None:
                        first_error = RuntimeError(
                            f"DDP worker rank {rank} failed request {request_id}: "
                            f"{error}\n{message.get('traceback')}"
                        )
                    continue

                result_indices = list(message["result_indices"])
                values = [float(value) for value in message["values"]]
                if result_indices and len(result_indices) != len(values):
                    raise RuntimeError(
                        f"DDP worker rank {rank} returned {len(values)} value(s) for "
                        f"{len(result_indices)} result index(es)."
                    )
                for index, value in zip(result_indices, values):
                    rewards[index] = value

        if first_error is not None:
            raise first_error
        missing = [index for index, value in enumerate(rewards) if value is None]
        if missing:
            raise RuntimeError(f"DDP request {request_id} missed rewards for indices {missing}.")
        return [float(value) for value in rewards]

    def _shutdown_locked(self, *, terminate_only: bool) -> None:
        """Stop workers while holding the lifecycle lock."""
        processes = self._processes
        request_queues = self._request_queues

        if not terminate_only:
            for request_queue in request_queues:
                try:
                    request_queue.put({"cmd": "shutdown"})
                except Exception:  # noqa: BLE001
                    pass

            deadline = time.monotonic() + self.shutdown_timeout_s
            for process in processes:
                timeout = max(0.0, deadline - time.monotonic())
                process.join(timeout=timeout)

        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.is_alive():
                process.join(timeout=5.0)

        self._request_queues = []
        self._response_queue = None
        self._processes = []
        self._started = False


def _worker_loop(
    *,
    local_rank: int,
    world_size: int,
    init_method: str,
    request_queue: Any,
    response_queue: Any,
    worker_factory: WorkerFactory,
    worker_kwargs: dict[str, Any],
) -> None:
    """Worker process entrypoint."""
    worker = None
    try:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=local_rank,
            world_size=world_size,
            timeout=timedelta(minutes=30),
        )
        worker = worker_factory(
            local_rank=local_rank,
            world_size=world_size,
            **worker_kwargs,
        )
        dist.barrier()
        response_queue.put({"kind": "ready", "rank": local_rank})
    except Exception as exc:  # noqa: BLE001
        response_queue.put(
            {
                "kind": "startup_error",
                "rank": local_rank,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _destroy_process_group()
        return

    try:
        while True:
            message = request_queue.get()
            command = message.get("cmd")
            if command == "shutdown":
                break
            if command != "compute":
                continue

            try:
                values = worker.compute(message["payload"])
                response_queue.put(
                    {
                        "kind": "result",
                        "request_id": message["request_id"],
                        "rank": local_rank,
                        "result_indices": message["result_indices"],
                        "values": values,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                response_queue.put(
                    {
                        "kind": "result",
                        "request_id": message["request_id"],
                        "rank": local_rank,
                        "result_indices": message["result_indices"],
                        "values": [],
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        if worker is not None and hasattr(worker, "close"):
            worker.close()
        _destroy_process_group()
        if torch.cuda.is_available():
            with torch.cuda.device(local_rank):
                torch.cuda.empty_cache()


def _destroy_process_group() -> None:
    """Destroy the current process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _find_free_port() -> int:
    """Find a free localhost TCP port for DDP process-group rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_shards(total_size: int, world_size: int) -> list[tuple[int, list[int], list[int]]]:
    """Build per-rank source/result index shards.

    Empty result shards still receive sample zero so every rank participates in
    the same compute step.
    """
    base, remainder = divmod(total_size, world_size)
    shards: list[tuple[int, list[int], list[int]]] = []
    start = 0
    for rank in range(world_size):
        count = base + (1 if rank < remainder else 0)
        result_indices = list(range(start, start + count))
        start += count
        source_indices = result_indices or [0]
        shards.append((rank, source_indices, result_indices))
    return shards


def _slice_payload(
    payload: dict[str, Any],
    batch_keys: set[str],
    source_indices: list[int],
) -> dict[str, Any]:
    """Slice list-valued batch fields and keep scalar metadata unchanged."""
    sliced: dict[str, Any] = {}
    for key, value in payload.items():
        if key in batch_keys and isinstance(value, list):
            sliced[key] = [value[index] for index in source_indices]
        else:
            sliced[key] = value
    return sliced
