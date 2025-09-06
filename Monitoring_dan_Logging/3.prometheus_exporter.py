import time
import argparse
import logging
from prometheus_client import Gauge, start_http_server
import psutil

CPU_GAUGE = Gauge("system_cpu_percent", "System CPU percent")
MEM_GAUGE = Gauge("system_memory_percent", "System memory percent")
DISK_GAUGE = Gauge("system_disk_percent", "System disk percent")
PROC_COUNT = Gauge("system_process_count", "Number of processes")
UPTIME = Gauge("system_uptime_seconds", "System uptime seconds")

def update_metrics():
    CPU_GAUGE.set(psutil.cpu_percent(interval=None))
    MEM_GAUGE.set(psutil.virtual_memory().percent)
    DISK_GAUGE.set(psutil.disk_usage("/").percent)
    PROC_COUNT.set(len(psutil.pids()))
    UPTIME.set(time.time() - psutil.boot_time())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting exporter on port {args.port}")
    start_http_server(args.port)

    while True:
        update_metrics()
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
