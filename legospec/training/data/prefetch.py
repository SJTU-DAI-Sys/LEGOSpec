"""Ordered CPU prefetch and one-batch CUDA staging without committing cursors."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass


@dataclass
class _Item:
    value: object = None
    error: BaseException | None = None
    done: bool = False


class OrderedCPUPrefetch:
    def __init__(self, source, capacity):
        if int(capacity) < 1:
            raise ValueError("CPU prefetch capacity must be positive")
        self.source = iter(source)
        self.queue = queue.Queue(maxsize=int(capacity))
        self.stop = threading.Event()
        self.thread = None
        self.closed = False
        try:
            self.first = next(self.source)
        except StopIteration:
            self.closed = True
            self.first = _Item(done=True)
            return
        self.first = _Item(value=self.first)
        self.thread = threading.Thread(
            target=self._produce, name="offline-cpu-prefetch", daemon=True
        )
        self.thread.start()

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self):
        try:
            while not self.stop.is_set():
                try:
                    value = next(self.source)
                except StopIteration:
                    self._put(_Item(done=True))
                    return
                if not self._put(_Item(value=value)):
                    return
        except BaseException as error:
            self._put(_Item(error=error))
        finally:
            close = getattr(self.source, "close", None)
            if close is not None:
                close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.first is not None:
            item, self.first = self.first, None
        elif self.closed:
            raise StopIteration
        else:
            item = self.queue.get()
        if item.error is not None:
            self.close()
            raise item.error
        if item.done:
            self.close()
            raise StopIteration
        return item.value

    def close(self):
        self.closed = True
        self.stop.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=10)


def ordered_cpu_batches(source, capacity):
    if int(capacity) == 0:
        yield from source
        return
    prefetched = OrderedCPUPrefetch(source, capacity)
    try:
        yield from prefetched
    finally:
        prefetched.close()


def cuda_prefetched_batches(source, target_layer_ids, require_last):
    """Stage one offline batch ahead while retaining CPU planning metadata."""
    import torch

    source = iter(source)
    stream = torch.cuda.Stream()
    keys = ("input_ids", "attention_mask", "loss_mask")

    def stage(data):
        with torch.cuda.stream(stream):
            inputs = {key: data[key].cuda(non_blocking=True) for key in keys}
            inputs["hidden_states"] = {
                layer: data["hidden_states"][layer].cuda(non_blocking=True)
                for layer in target_layer_ids
            }
            if require_last or data.get("final_hidden_state") is not None:
                inputs["final_hidden_state"] = data["final_hidden_state"].cuda(
                    non_blocking=True
                )
            ready = stream.record_event()
        return dict(data, _cuda_inputs=inputs), ready

    pending = next(source, None)
    if pending is None:
        return
    pending, ready = stage(pending)
    try:
        while pending is not None:
            current, event = pending, ready
            compute = torch.cuda.current_stream()
            compute.wait_event(event)
            for value in current["_cuda_inputs"].values():
                tensors = value.values() if isinstance(value, dict) else (value,)
                for tensor in tensors:
                    tensor.record_stream(compute)
            following = next(source, None)
            if following is None:
                pending = None
            else:
                pending, ready = stage(following)
            yield current
    finally:
        # Also covers early iterator close: queued DMA must release CPU buffers.
        stream.synchronize()
        close = getattr(source, "close", None)
        if close is not None:
            close()
