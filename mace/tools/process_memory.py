"""Optional NVML process-memory sampling, independent of CUDA allocators."""
import atexit
import os
import threading


class ProcessMemory:
    def __init__(self, interval=0.02):
        self.interval = interval
        self.peak = 0
        self.samples = 0
        self.error = None
        self.initialized = False
        self.thread = None
        self.done = threading.Event()
        self.gpu_uuids = set()
        self.other_compute_pids = set()
        try:
            import pynvml

            self.nvml = pynvml
            pynvml.nvmlInit()
            self.initialized = True
            self.handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
            self.sample()
            self.thread = threading.Thread(target=self.poll, daemon=True)
            self.thread.start()
        except Exception as exc:
            self.error = str(exc)
        atexit.register(self.stop)

    def sample(self):
        values = []
        for handle in self.handles:
            processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for process in processes:
                if process.pid == os.getpid():
                    uuid = self.nvml.nvmlDeviceGetUUID(handle)
                    self.gpu_uuids.add(uuid.decode() if isinstance(uuid, bytes) else uuid)
                    self.other_compute_pids.update(p.pid for p in processes if p.pid != os.getpid())
                    value = process.usedGpuMemory
                    if value is None or value >= (1 << 63):
                        raise RuntimeError("NVML process memory is unavailable")
                    values.append(value)
        if not values:
            raise RuntimeError(
                "Current process is not visible in NVML compute processes"
            )
        self.peak = max(self.peak, sum(values))
        self.samples += 1

    def poll(self):
        while not self.done.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.error = str(exc)
                return

    def stop(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=1)
        if self.initialized:
            self.nvml.nvmlShutdown()
            self.initialized = False
        return {
            "status": ("partial" if self.samples else "unavailable")
            if self.error
            else "ok",
            "sampled_peak_bytes": self.peak if self.samples else None,
            "samples": self.samples,
            "poll_interval_ms": 1000 * self.interval,
            "scope": "current process across physical GPUs; includes all resident implementations",
            "gpu_uuids": sorted(self.gpu_uuids),
            "other_compute_pids_on_process_gpus": sorted(self.other_compute_pids),
            "error": self.error,
        }
