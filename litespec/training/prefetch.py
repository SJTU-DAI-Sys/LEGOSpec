"""Bounded, ordered CPU input delivery without advancing training cursors.

The first next() is performed by the caller before starting a thread. This
keeps DataLoader worker creation on the main thread; subsequent retrieval,
collation delivery and sampler replenishment can overlap GPU work.
"""
from dataclasses import dataclass
import queue
import threading


@dataclass
class _Item:
    value: object = None
    error: BaseException | None = None
    done: bool = False


class OrderedCPUPrefetch:
    def __init__(self, source, capacity):
        if int(capacity) < 1:
            raise ValueError('CPU prefetch capacity must be positive')
        self.source=iter(source)
        self.queue=queue.Queue(maxsize=int(capacity))
        self.stop=threading.Event()
        self.thread=None
        self.closed=False
        try:self.first=next(self.source)
        except StopIteration:
            self.closed=True;self.first=_Item(done=True);return
        self.first=_Item(value=self.first)
        self.thread=threading.Thread(target=self._produce,name='offline-cpu-prefetch',daemon=True)
        self.thread.start()

    def _put(self,item):
        while not self.stop.is_set():
            try:self.queue.put(item,timeout=.05);return True
            except queue.Full:pass
        return False

    def _produce(self):
        try:
            while not self.stop.is_set():
                try:value=next(self.source)
                except StopIteration:
                    self._put(_Item(done=True));return
                if not self._put(_Item(value=value)):return
        except BaseException as error:
            self._put(_Item(error=error))
        finally:
            close=getattr(self.source,'close',None)
            if close is not None:close()

    def __iter__(self):return self

    def __next__(self):
        if self.first is not None:
            item,self.first=self.first,None
        elif self.closed:raise StopIteration
        else:item=self.queue.get()
        if item.error is not None:
            self.close();raise item.error
        if item.done:
            self.close();raise StopIteration
        return item.value

    def close(self):
        self.closed=True;self.stop.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=10)
        # The producer owns source.close(), avoiding concurrent generator close.


def ordered_cpu_batches(source,capacity):
    if int(capacity)==0:
        yield from source;return
    prefetched=OrderedCPUPrefetch(source,capacity)
    try:yield from prefetched
    finally:prefetched.close()
