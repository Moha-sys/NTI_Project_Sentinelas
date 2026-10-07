#!/usr/bin/env python3
"""
Automated Telemetry Dataset Generator for AIOps Machine Learning
Generates high-fidelity labeled telemetry datasets for:
1. Unsupervised Anomaly Detection (One-Class / Isolation Forest / Autoencoder)
   -> data/normal_baseline_dataset.csv (100% normal operations)
2. Root Cause Analysis & Multi-Class Failure Classification (RCA)
   -> data/chaos_labeled_dataset.csv (6 labeled fault types + normal windows)
3. Combined Telemetry Dataset
   -> data/combined_telemetry_dataset.csv (Unified master dataset)

Features Collected (every 3 seconds per microservice):
- timestamp: Unix epoch timestamp
- service_name: (frontend, cart-service, payment-service, productcatalog-service, redis-cart)
- cpu_percent: CPU utilization % (0.0 to 100.0)
- memory_mb: Memory utilization in MB
- latency_ms: Response latency in ms
- error_rate: HTTP 5xx error ratio (0.0 to 1.0)
- service_up: Binary availability (1 = alive, 0 = dead)

Labels:
- is_anomaly: (0 = normal, 1 = active fault)
- fault_type: (normal, pod_failure, cpu_stress, network_latency, memory_leak, packet_loss, cascade_redis_drop)
- root_cause_service: Culprit microservice ('none' if normal)
"""

import os
import sys
import time
import json
import csv
import signal
import random
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_SERVICES = [
    "frontend",
    "cart-service",
    "payment-service",
    "productcatalog-service",
    "redis-cart"
]

FAULT_TYPES = [
    "pod_failure",
    "cpu_stress",
    "network_latency",
    "memory_leak",
    "packet_loss",
    "cascade_redis_drop"
]

CSV_FIELDNAMES = [
    "timestamp",
    "service_name",
    "cpu_percent",
    "memory_mb",
    "latency_ms",
    "error_rate",
    "service_up",
    "is_anomaly",
    "fault_type",
    "root_cause_service"
]

# Global reference for clean signal exit
ACTIVE_API_URL = DEFAULT_API_URL

def handle_sigint(signum, frame):
    print("\n\n[!] Interrupt received (SIGINT). Restoring cluster to healthy state...")
    try:
        remediate_all(ACTIVE_API_URL)
    except Exception:
        pass
    print("[+] Remediation complete. Exiting cleanly.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

def check_backend_health(api_url: str) -> bool:
    """Verifies that the FastAPI AIOps platform backend is active and responsive."""
    try:
        req = urllib.request.Request(f"{api_url}/api/v1/health", headers={"User-Agent": "DatasetGenerator/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass
    return False

def http_post_json(url: str, payload: dict) -> dict:
    """Helper to perform HTTP POST with JSON body."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "DatasetGenerator/1.0"}
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))

def fetch_mesh_telemetry(api_url: str) -> List[Dict[str, Any]]:
    """Fetches real-time telemetry from all 5 supervised microservices."""
    req = urllib.request.Request(
        f"{api_url}/api/v1/mesh/status",
        headers={"User-Agent": "DatasetGenerator/1.0"}
    )
    with urllib.request.urlopen(req, timeout=4.0) as resp:
        return json.loads(resp.read().decode("utf-8"))

def inject_chaos(api_url: str, service: str, fault_type: str) -> dict:
    """Injects a precision fault into the specified microservice."""
    url = f"{api_url}/api/v1/mesh/chaos/inject"
    payload = {
        "service": service,
        "experiment_type": fault_type
    }
    return http_post_json(url, payload)

def remediate_all(api_url: str) -> dict:
    """Heals all injected chaos and restores all microservices to normal operations."""
    url = f"{api_url}/api/v1/mesh/remediate"
    payload = {"service": "all"}
    try:
        return http_post_json(url, payload)
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def record_sample(
    service_status: Dict[str, Any],
    is_anomaly: int,
    fault_type: str,
    root_cause_service: str
) -> Dict[str, Any]:
    """Formats raw telemetry into a labeled dataset observation row."""
    ts = int(time.time())
    name = service_status.get("service_name") or service_status.get("service", "unknown")
    
    cpu = float(service_status.get("cpu_percent", 0.0))
    mem = float(service_status.get("memory_mb", 0.0))
    lat = float(service_status.get("latency_ms", 15.0))
    err = float(service_status.get("error_rate", 0.0))
    up = int(service_status.get("service_up", 1 if service_status.get("status") == "HEALTHY" else 0))

    return {
        "timestamp": ts,
        "service_name": name,
        "cpu_percent": round(cpu, 2),
        "memory_mb": round(mem, 2),
        "latency_ms": round(lat, 2),
        "error_rate": round(err, 4),
        "service_up": up,
        "is_anomaly": is_anomaly,
        "fault_type": fault_type,
        "root_cause_service": root_cause_service
    }

def append_rows_to_csv(filepath: Path, rows: List[Dict[str, Any]], init_file: bool = False):
    """Safely appends rows to target CSV file, creating header if needed."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    write_header = init_file or not filepath.exists() or filepath.stat().st_size == 0
    mode = "w" if init_file else "a"
    
    with open(filepath, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def run_normal_baseline_collection(
    api_url: str,
    duration_sec: int,
    interval_sec: int,
    output_path: Path
) -> int:
    """
    Collects clean normal baseline dataset (100% healthy operations).
    Guarantees no chaos active, is_anomaly=0, fault_type='normal', root_cause_service='none'.
    """
    print("=" * 72)
    print("  [MODE 1/2] Normal Baseline Telemetry Collection (Unsupervised ML)")
    print(f"  Target Duration: {duration_sec}s | Sampling Interval: {interval_sec}s")
    print(f"  Output File:     {output_path}")
    print("=" * 72)

    # Ensure all microservices are healthy before starting
    remediate_all(api_url)
    time.sleep(2)

    append_rows_to_csv(output_path, [], init_file=True)
    total_samples = 0
    start_time = time.time()
    next_sample_time = start_time

    while time.time() - start_time < duration_sec:
        now = time.time()
        if now >= next_sample_time:
            try:
                mesh_data = fetch_mesh_telemetry(api_url)
                rows = []
                for s in mesh_data:
                    row = record_sample(
                        service_status=s,
                        is_anomaly=0,
                        fault_type="normal",
                        root_cause_service="none"
                    )
                    rows.append(row)
                
                append_rows_to_csv(output_path, rows)
                total_samples += len(rows)
                
                elapsed = int(time.time() - start_time)
                remaining = max(0, duration_sec - elapsed)
                pct = int((elapsed / max(1, duration_sec)) * 100)
                print(f"\r  [Normal Baseline] {elapsed}s/{duration_sec}s ({pct}%) | Collected: {total_samples} samples | Remaining: {remaining}s", end="", flush=True)
            except Exception as e:
                print(f"\n  [!] Warning collecting telemetry: {e}")

            next_sample_time += interval_sec
        time.sleep(0.2)

    print(f"\n  [✓] Normal baseline collection completed: {total_samples} samples saved to {output_path}.\n")
    return total_samples

def run_chaos_rca_collection(
    api_url: str,
    cycles: int,
    normal_window: int,
    fault_duration: int,
    heal_duration: int,
    interval_sec: int,
    output_path: Path
) -> int:
    """
    Collects labeled chaos failure dataset for Root Cause Analysis (RCA).
    Cycles through:
    1. Pre-fault normal stabilization window (is_anomaly=0)
    2. Active precision chaos fault injection (is_anomaly=1, labeled fault_type & root_cause_service)
    3. Self-healing remediation recovery window
    """
    print("=" * 72)
    print("  [MODE 2/2] Chaos & Root Cause Analysis (RCA) Telemetry Collection")
    print(f"  Chaos Cycles:   {cycles} cycles")
    print(f"  Cycle Anatomy:  {normal_window}s Normal -> {fault_duration}s Fault -> {heal_duration}s Heal")
    print(f"  Output File:    {output_path}")
    print("=" * 72)

    append_rows_to_csv(output_path, [], init_file=True)
    total_samples = 0

    # Build balanced schedule of fault experiments across microservices
    experiments = []
    # Microservices for single-target failures
    single_target_services = ["cart-service", "frontend", "payment-service", "productcatalog-service", "redis-cart"]

    for c in range(cycles):
        # Round-robin through the 6 fault types
        f_type = FAULT_TYPES[c % len(FAULT_TYPES)]
        if f_type == "cascade_redis_drop":
            target = "redis-cart"
        else:
            # Rotate services so all services get exposed to various faults
            target = single_target_services[(c // len(FAULT_TYPES) + c) % len(single_target_services)]
        experiments.append((f_type, target))

    for idx, (f_type, target_svc) in enumerate(experiments, start=1):
        print(f"\n--- [Cycle {idx}/{cycles}] Experiment: {f_type.upper()} on {target_svc} ---")
        
        # 1. Normal pre-fault window
        print(f"  [1/3] Normal Pre-Fault Stabilization ({normal_window}s)...")
        remediate_all(api_url)
        w_start = time.time()
        next_t = w_start
        while time.time() - w_start < normal_window:
            if time.time() >= next_t:
                try:
                    mesh_data = fetch_mesh_telemetry(api_url)
                    rows = [
                        record_sample(s, is_anomaly=0, fault_type="normal", root_cause_service="none")
                        for s in mesh_data
                    ]
                    append_rows_to_csv(output_path, rows)
                    total_samples += len(rows)
                except Exception as e:
                    print(f"  [!] Telemetry error: {e}")
                next_t += interval_sec
            time.sleep(0.2)

        # 2. Inject fault window
        print(f"  [2/3] Injecting Chaos: {f_type} -> {target_svc} ({fault_duration}s active)...")
        try:
            res = inject_chaos(api_url, target_svc, f_type)
            action = res.get("action", f_type)
            print(f"        -> Chaos Mesh Engine: {action} active on {target_svc}")
        except Exception as e:
            print(f"  [!] Failed to inject chaos: {e}")

        w_start = time.time()
        next_t = w_start
        while time.time() - w_start < fault_duration:
            if time.time() >= next_t:
                try:
                    mesh_data = fetch_mesh_telemetry(api_url)
                    rows = [
                        record_sample(s, is_anomaly=1, fault_type=f_type, root_cause_service=target_svc)
                        for s in mesh_data
                    ]
                    append_rows_to_csv(output_path, rows)
                    total_samples += len(rows)
                except Exception as e:
                    print(f"  [!] Telemetry error: {e}")
                next_t += interval_sec
            time.sleep(0.2)

        # 3. Heal / recovery window
        print(f"  [3/3] Executing Self-Healing Remediation ({heal_duration}s recovery)...")
        remediate_all(api_url)
        w_start = time.time()
        next_t = w_start
        while time.time() - w_start < heal_duration:
            if time.time() >= next_t:
                try:
                    mesh_data = fetch_mesh_telemetry(api_url)
                    rows = [
                        record_sample(s, is_anomaly=0, fault_type="normal", root_cause_service="none")
                        for s in mesh_data
                    ]
                    append_rows_to_csv(output_path, rows)
                    total_samples += len(rows)
                except Exception as e:
                    print(f"  [!] Telemetry error: {e}")
                next_t += interval_sec
            time.sleep(0.2)

    print(f"\n  [✓] Chaos & RCA collection completed: {total_samples} samples saved to {output_path}.\n")
    return total_samples

def create_combined_dataset(normal_path: Path, chaos_path: Path, combined_path: Path) -> int:
    """Merges normal baseline and chaos labeled datasets into a unified master training dataset."""
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    with open(combined_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for src in [normal_path, chaos_path]:
            if src.exists() and src.stat().st_size > 0:
                with open(src, "r", newline="", encoding="utf-8") as in_f:
                    reader = csv.DictReader(in_f)
                    for row in reader:
                        writer.writerow(row)
                        total_rows += 1

    print(f"  [✓] Combined dataset created: {total_rows} total rows saved to {combined_path}.")
    return total_rows

def main():
    parser = argparse.ArgumentParser(
        description="Automated Telemetry Dataset Generator for Unsupervised Anomaly Detection & RCA"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "normal_only", "chaos_only"],
        default="all",
        help="Collection mode: 'all' (default), 'normal_only', or 'chaos_only'"
    )
    parser.add_argument(
        "--normal-duration",
        type=int,
        default=300,
        help="Duration in seconds for normal baseline collection (default: 300)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=12,
        help="Number of chaos cycles to run (default: 12)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3,
        help="Sampling interval in seconds per microservice (default: 3)"
    )
    parser.add_argument(
        "--fault-duration",
        type=int,
        default=35,
        help="Duration of fault injection window per cycle in seconds (default: 35)"
    )
    parser.add_argument(
        "--normal-window",
        type=int,
        default=30,
        help="Duration of pre-fault normal window per cycle in seconds (default: 30)"
    )
    parser.add_argument(
        "--heal-duration",
        type=int,
        default=15,
        help="Duration of recovery heal window per cycle in seconds (default: 15)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"Base URL for AIOps platform FastAPI backend (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save CSV datasets (default: data)"
    )

    args = parser.parse_args()

    global ACTIVE_API_URL
    ACTIVE_API_URL = args.api_url

    print("=" * 72)
    print("  Chaos-Grounded Autonomous AIOps Platform - Dataset Generator")
    print(f"  Backend URL: {args.api_url}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Mode Selected: {args.mode}")
    print("=" * 72)

    # 1. Health check
    if not check_backend_health(args.api_url):
        print(f"\n[!] Error: Unable to connect to AIOps engine at {args.api_url}.")
        print("    Please start the engine first using:")
        print("    ./run.sh dev-backend\n")
        sys.exit(1)
    
    print("[✓] Backend engine confirmed healthy. Microservice Mesh active.\n")

    out_dir = Path(args.output_dir).resolve()
    normal_csv = out_dir / "normal_baseline_dataset.csv"
    chaos_csv = out_dir / "chaos_labeled_dataset.csv"
    combined_csv = out_dir / "combined_telemetry_dataset.csv"

    # 2. Run selected collection modes
    if args.mode in ["all", "normal_only"]:
        run_normal_baseline_collection(
            api_url=args.api_url,
            duration_sec=args.normal_duration,
            interval_sec=args.interval,
            output_path=normal_csv
        )

    if args.mode in ["all", "chaos_only"]:
        run_chaos_rca_collection(
            api_url=args.api_url,
            cycles=args.cycles,
            normal_window=args.normal_window,
            fault_duration=args.fault_duration,
            heal_duration=args.heal_duration,
            interval_sec=args.interval,
            output_path=chaos_csv
        )

    # 3. Combine datasets
    if args.mode == "all":
        create_combined_dataset(normal_csv, chaos_csv, combined_csv)
    elif args.mode == "normal_only" and normal_csv.exists():
        create_combined_dataset(normal_csv, chaos_csv, combined_csv)
    elif args.mode == "chaos_only" and chaos_csv.exists():
        create_combined_dataset(normal_csv, chaos_csv, combined_csv)

    print("\n" + "=" * 72)
    print("  Dataset Generation Complete!")
    print("=" * 72)
    for p in [normal_csv, chaos_csv, combined_csv]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f) - 1
            print(f"  • {p.name:<32}: {lines:>6} samples ({p})")
    print("=" * 72 + "\n")

if __name__ == "__main__":
    main()
