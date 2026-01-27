import torch
import threading
import queue
import time


class ThreadedScheduler():
    """
    Usage:

    with ThreadedScheduler(scheduler) as data_iterator:
        for data in data_iterator:
            # do something with data

    """

    def __init__(self, scheduler, queue_size: int = 4):
        self.data_queue = queue.Queue(maxsize=queue_size)
        self.scheduler_thread = threading.Thread(target=self.loadData)
        self.scheduler = scheduler
        self._sentinel_done = object()
        self._sentinel_error = object()
        self._error: Exception | None = None

    def loadData(self):
        try:
            for data_tuple in self.scheduler:
                data_copy = []
                for each in data_tuple:
                    if isinstance(each, torch.Tensor):
                        data_copy.append(each.clone())
                    elif isinstance(each, (list, tuple)):
                        copied = []
                        for item in each:
                            copied.append(item.clone() if isinstance(item, torch.Tensor) else item)
                        data_copy.append(copied)
                    else:
                        data_copy.append(each)

                self.data_queue.put(data_copy)
        except Exception as e:
            self._error = e
            self.data_queue.put(self._sentinel_error)
        finally:
            self.data_queue.put(self._sentinel_done)

    def __len__(self):
        return len(self.scheduler)

    def __enter__(self):
        self.scheduler_thread.start()
        return self

    def __iter__(self):
        remaining = len(self.scheduler)
        while remaining > 0:
            # If the loader thread crashed before producing any data, avoid blocking forever.
            try:
                item = self.data_queue.get(timeout=60)
            except queue.Empty:
                if not self.scheduler_thread.is_alive():
                    raise RuntimeError(
                        "ThreadedScheduler stalled: loader thread exited without producing data. "
                        f"error={self._error}"
                    ) from self._error
                raise RuntimeError("ThreadedScheduler stalled: no data produced for 60s (loader still alive).")

            if item is self._sentinel_error:
                raise RuntimeError(f"ThreadedScheduler loader failed: {self._error}") from self._error
            if item is self._sentinel_done:
                return
            yield item
            remaining -= 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        # error handle
        self.scheduler_thread.join()
