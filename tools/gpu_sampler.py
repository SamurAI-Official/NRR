#!/usr/bin/env python3
"""gpu_sampler.py - ask the driver what the GPU was actually doing.

A launched kernel is not a used GPU, and only nvidia-smi can say which one happened. Shared by
tools/train_nrr.py (training) and tools/measure_model.py (inference), so the two report the same
kind of number from the same source rather than each having their own idea of "utilization".
"""

import subprocess
import threading


class GpuSampler(threading.Thread):
    """Samples utilization.gpu and memory.used on an interval while a workload runs."""

    def __init__(self, interval=0.4):
        super().__init__(daemon=True)
        self.interval = interval
        # Not named _stop: threading.Thread has an internal _stop() method, and an attribute of that name
        # shadows it, so Thread.join() fails with "TypeError: 'Event' object is not callable".
        self._halt = threading.Event()
        self.samples = []

    def run(self):
        while not self._halt.is_set():
            try:
                lines = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                for line in lines:
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) == 2:
                        self.samples.append((float(parts[0]), float(parts[1])))
            except Exception:
                pass  # a sampling gap is not a workload failure
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)

    def summary(self):
        if not self.samples:
            return {"samples": 0,
                    "note": "nvidia-smi produced no samples, so utilization was not measured"}
        usage = [sample[0] for sample in self.samples]
        memory = [sample[1] for sample in self.samples]
        return {"samples": len(self.samples),
                "utilization_percent": {"mean": round(sum(usage) / len(usage), 1), "max": max(usage)},
                "memory_used_mb": {"mean": round(sum(memory) / len(memory)), "max": max(memory)}}