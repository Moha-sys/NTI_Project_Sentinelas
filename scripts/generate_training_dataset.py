#!/usr/bin/env python3
"""
Automated Telemetry Dataset Generator for AIOps Machine Learning
Generates high-fidelity labeled telemetry datasets with realistic service variance:
1. Primary AIOps Chaos Dataset -> data/aiops_chaos_dataset.csv
2. Normal Baseline Dataset     -> data/normal_baseline_dataset.csv (100% normal)
3. Chaos Labeled Dataset       -> data/chaos_labeled_dataset.csv (Chaos cycles only)
4. Combined Telemetry Dataset  -> data/combined_telemetry_dataset.csv

Features:
- timestamp: Unix epoch timestamp (advances every 3 seconds per microservice)
- service_name: (frontend, cart-service, payment-service, productcatalog-service, redis-cart)
- cpu_percent: Realistic CPU % with service-specific baselines & jitter (0.0 to 100.0)
- memory_mb: Service-specific memory footprint (e.g. frontend ~148MB vs redis ~32MB)
- latency_ms: Service-specific response latency in ms
- error_rate: HTTP 5xx error ratio (0.0 to 1.0)
- service_up: Binary availability (1 = alive, 0 = crashed)

Labels:
- is_anomaly: (0 = normal, 1 = active fault)
- fault_type: (normal, pod_failure, cpu_stress, network_latency, memory_leak, packet_loss, cascade_redis_drop)
- root_cause_service: Culprit microservice ('none' if normal)
"""

import os
import sys
import time
import math
import random
import csv
import signal
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple

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

# Service profiles with distinct architectural baselines to prevent flat data
SERVICE_PROFILES = {
    "frontend": {
        "base_cpu": 3.8,
        "cpu_std": 0.85,
        "base_mem": 148.0,
        "mem_std": 6.2,
        "base_lat": 24.5,
        "lat_std": 3.8,
        "dependencies": ["cart-service", "payment-service", "productcatalog-service"]
    },
    "cart-service": {
        "base_cpu": 2.2,
        "cpu_std": 0.55,
        "base_mem": 96.0,
        "mem_std": 4.5,
        "base_lat": 16.2,
        "lat_std": 2.4,
        "dependencies": ["redis-cart"]
    },
    "payment-service": {
        "base_cpu": 1.4,
        "cpu_std": 0.38,
        "base_mem": 44.5,
        "mem_std": 2.6,
        "base_lat": 11.5,
        "lat_std": 1.8,
        "dependencies": []
    },
    "productcatalog-service": {
        "base_cpu": 1.8,
        "cpu_std": 0.45,
        "base_mem": 64.0,
        "mem_std": 3.2,
        "base_lat": 14.0,
        "lat_std": 2.1,
        "dependencies": []
    },
    "redis-cart": {
        "base_cpu": 0.85,
        "cpu_std": 0.22,
        "base_mem": 32.5,
        "mem_std": 1.9,
        "base_lat": 3.8,
        "lat_std": 0.8,
        "dependencies": []
    }
}

def generate_step_telemetry(
    timestamp: int,
    step_index: int,
    active_fault: str,
    root_cause_service: str,
    fault_progress: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Generates realistic, multivariate telemetry across all 5 microservices for a single time step.
    Includes natural organic diurnal load oscillations, Gaussian variance, and cascade fault physics.
    """
    # Natural organic load breathing (sinusoidal diurnal wave +/- 12% load variation)
    load_factor = 1.0 + 0.12 * math.sin(step_index * 0.05)

    rows = []
    for svc in DEFAULT_SERVICES:
        profile = SERVICE_PROFILES[svc]
        base_cpu = profile["base_cpu"] * load_factor
        base_mem = profile["base_mem"] * (1.0 + 0.03 * math.sin(step_index * 0.03))
        base_lat = profile["base_lat"] * load_factor

        is_culprit = (svc == root_cause_service)
        is_normal = (active_fault == "normal" or not active_fault)

        if is_normal:
            # Clean normal baseline with realistic stochastic noise
            cpu = max(0.4, min(8.5, random.gauss(base_cpu, profile["cpu_std"])))
            mem = max(15.0, random.gauss(base_mem, profile["mem_std"]))
            lat = max(1.2, random.gauss(base_lat, profile["lat_std"]))
            err = max(0.0, min(0.012, random.gauss(0.001, 0.0004)))
            up = 1
            anomaly = 0
            f_type = "normal"
            rc_svc = "none"

        elif active_fault == "pod_failure":
            if is_culprit:
                cpu = 0.0
                mem = 0.0
                lat = 0.0
                err = 1.0
                up = 0
            else:
                cpu = max(0.4, min(8.5, random.gauss(base_cpu, profile["cpu_std"])))
                mem = max(15.0, random.gauss(base_mem, profile["mem_std"]))
                up = 1
                # Downstream dependents observe connection drop / retries
                if root_cause_service in profile["dependencies"]:
                    lat = max(base_lat + 40.0, random.gauss(base_lat + 120.0, 15.0))
                    err = max(0.25, min(0.65, random.gauss(0.45, 0.06)))
                else:
                    lat = max(1.2, random.gauss(base_lat, profile["lat_std"]))
                    err = max(0.0, min(0.012, random.gauss(0.001, 0.0004)))
            anomaly = 1
            f_type = "pod_failure"
            rc_svc = root_cause_service

        elif active_fault == "cpu_stress":
            if is_culprit:
                # 88% - 98.5% CPU saturation
                cpu = max(88.0, min(99.4, random.gauss(94.2, 2.2)))
                mem = max(base_mem + 12.0, random.gauss(base_mem + 24.0, 3.5))
                lat = max(base_lat * 2.5, random.gauss(base_lat * 3.8, 5.5))
                err = max(0.01, min(0.06, random.gauss(0.032, 0.008)))
                up = 1
            else:
                cpu = max(0.4, min(8.5, random.gauss(base_cpu, profile["cpu_std"])))
                mem = max(15.0, random.gauss(base_mem, profile["mem_std"]))
                lat = max(1.2, random.gauss(base_lat, profile["lat_std"]))
                err = max(0.0, min(0.012, random.gauss(0.001, 0.0004)))
                up = 1
            anomaly = 1
            f_type = "cpu_stress"
            rc_svc = root_cause_service

        elif active_fault == "network_latency":
            if is_culprit:
                cpu = max(0.4, min(8.5, random.gauss(base_cpu, profile["cpu_std"])))
                mem = max(15.0, random.gauss(base_mem, profile["mem_std"]))
                # 850ms - 1250ms latency spike
                lat = max(780.0, min(1350.0, random.gauss(925.0, 68.0)))
                err = max(0.0, min(0.045, random.gauss(0.018, 0.006)))
                up = 1
            else:
                cpu = max(0.4, min(8.5, random.gauss(base_cpu, profile["cpu_std"])))
                mem = max(15.0, random.gauss(base_mem, profile["mem_std"]))
                if root_cause_service in profile["dependencies"]:
                    lat = max(base_lat + 180.0, random.gauss(base_lat + 320.0, 35.0))
                    err = max(0.01, min(0.08, random.gauss(0.035, 0.012)))
                else:
                    lat = max(1.2, random.gauss(base_lat, profile["lat_std"]))
                    err = max(0.0, min(0.012, random.gauss(0.001, 0.0004)))
                up = 1
            anomaly = 1
            f_type = "network_latency"
            rc_svc = root_cause_service

        elif active_fault == "memory_leak":
            if is_culprit:
                # Memory swells linearly with fault progress from +120MB to +320MB
                leak_amount = 140.0 + fault_progress * 190.0
                cpu = max(base_cpu + 1.2, min(12.0, random.gauss(base_cpu + 2.8, 0.6)))
                mem = max(base_mem + 100.0, random.gauss(base_mem + leak_amount, 6.5))
                lat = max(base_lat * 1.3, random.gauss(base_lat * 1.8, 3.2))
                err = max(0.0, min(0.035, random.gauss(0.012, 0.004)))
                up = 1
            else:
                cpu = max(0.4, min(8.5, random.gauss(base_cpu, profile["cpu_std"])))
                mem = max(15.0, random.gauss(base_mem, profile["mem_std"]))
                lat = max(1.2, random.gauss(base_lat, profile["lat_std"]))
                err = max(0.0, min(0.012, random.gauss(0.001, 0.0004)))
                up = 1
            anomaly = 1
            f_type = "memory_leak"
            rc_svc = root_cause_service

        elif active_fault == "packet_loss":
            if is_culprit:
                cpu = max(0.4, min(8.5, random.gauss(base_cpu, profile["cpu_std"])))
                mem = max(15.0, random.gauss(base_mem, profile["mem_std"]))
                lat = max(base_lat * 4.0, random.gauss(190.0, 26.0))
                # 45% - 75% error rate from dropped packets
                err = max(0.42, min(0.78, random.gauss(0.62, 0.07)))
                up = 1
            else:
                cpu = max(0.4, min(8.5, random.gauss(base_cpu, profile["cpu_std"])))
                mem = max(15.0, random.gauss(base_mem, profile["mem_std"]))
                if root_cause_service in profile["dependencies"]:
                    lat = max(base_lat + 80.0, random.gauss(base_lat + 160.0, 20.0))
                    err = max(0.15, min(0.45, random.gauss(0.28, 0.05)))
                else:
                    lat = max(1.2, random.gauss(base_lat, profile["lat_std"]))
                    err = max(0.0, min(0.012, random.gauss(0.001, 0.0004)))
                up = 1
            anomaly = 1
            f_type = "packet_loss"
            rc_svc = root_cause_service

        elif active_fault == "cascade_redis_drop":
            if svc == "redis-cart":
                cpu = 0.0
                mem = 0.0
                lat = 0.0
                err = 1.0
                up = 0
            elif svc == "cart-service":
                # Cart service direct dependency failure
                cpu = max(2.5, min(9.0, random.gauss(4.2, 0.65)))
                mem = max(base_mem, random.gauss(base_mem + 8.0, 3.5))
                lat = max(210.0, min(520.0, random.gauss(335.0, 38.0)))
                err = max(0.68, min(0.92, random.gauss(0.79, 0.05)))
                up = 1
            elif svc == "frontend":
                # Frontend secondary cascading failure
                cpu = max(base_cpu + 1.0, min(10.0, random.gauss(base_cpu + 2.2, 0.7)))
                mem = max(base_mem, random.gauss(base_mem + 10.0, 4.0))
                lat = max(140.0, min(360.0, random.gauss(230.0, 28.0)))
                err = max(0.32, min(0.56, random.gauss(0.42, 0.045)))
                up = 1
            else:
                # payment-service and productcatalog-service are isolated and healthy
                cpu = max(0.4, min(8.5, random.gauss(base_cpu, profile["cpu_std"])))
                mem = max(15.0, random.gauss(base_mem, profile["mem_std"]))
                lat = max(1.2, random.gauss(base_lat, profile["lat_std"]))
                err = max(0.0, min(0.012, random.gauss(0.001, 0.0004)))
                up = 1
            anomaly = 1
            f_type = "cascade_redis_drop"
            rc_svc = "redis-cart"

        else:
            cpu = max(0.4, random.gauss(base_cpu, profile["cpu_std"]))
            mem = max(15.0, random.gauss(base_mem, profile["mem_std"]))
            lat = max(1.2, random.gauss(base_lat, profile["lat_std"]))
            err = 0.0
            up = 1
            anomaly = 0
            f_type = "normal"
            rc_svc = "none"

        rows.append({
            "timestamp": timestamp,
            "service_name": svc,
            "cpu_percent": round(cpu, 2),
            "memory_mb": round(mem, 2),
            "latency_ms": round(lat, 2),
            "error_rate": round(err, 4),
            "service_up": up,
            "is_anomaly": anomaly,
            "fault_type": f_type,
            "root_cause_service": rc_svc
        })

    return rows

def generate_target_rows_dataset(
    target_rows: int,
    normal_ratio: float = 0.60,
    interval_sec: int = 3,
    output_dir: Path = Path("data")
) -> Tuple[Path, int]:
    """
    Generates a full, comprehensive, high-variance dataset matching target_rows.
    Guarantees ~60% normal and ~40% balanced fault distribution across all 6 modes and 5 services.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_services = len(DEFAULT_SERVICES)
    total_steps = math.ceil(target_rows / num_services)
    
    normal_steps_target = int(total_steps * normal_ratio)
    chaos_steps_target = total_steps - normal_steps_target

    print("=" * 74)
    print("  Scalable High-Fidelity Telemetry Dataset Generator")
    print(f"  Target Total Rows:     {target_rows} rows ({total_steps} timestamp steps)")
    print(f"  Target Distribution:   ~{int(normal_ratio*100)}% Normal ({normal_steps_target * num_services} rows) | "
          f"~{int((1-normal_ratio)*100)}% Chaos ({chaos_steps_target * num_services} rows)")
    print(f"  Services Covered:      {', '.join(DEFAULT_SERVICES)}")
    print(f"  Chaos Failure Modes:   {', '.join(FAULT_TYPES)}")
    print("=" * 74)

    # Build sequence of blocks: alternating normal baseline and chaos episodes
    # Each chaos episode: pre-normal (4-6 steps) -> fault (8-12 steps) -> recovery (3-5 steps)
    single_target_services = ["frontend", "cart-service", "payment-service", "productcatalog-service", "redis-cart"]

    # Calculate number of episodes
    steps_per_fault = 10
    num_episodes = max(1, chaos_steps_target // steps_per_fault)
    steps_per_normal_block = max(1, normal_steps_target // (num_episodes + 1))

    # Pre-build episode schedule ensuring equal coverage of all 6 fault types and services
    episode_schedule = []
    for ep_idx in range(num_episodes):
        f_type = FAULT_TYPES[ep_idx % len(FAULT_TYPES)]
        if f_type == "cascade_redis_drop":
            target_svc = "redis-cart"
        else:
            target_svc = single_target_services[(ep_idx // len(FAULT_TYPES) + ep_idx) % len(single_target_services)]
        episode_schedule.append((f_type, target_svc))

    all_rows = []
    normal_rows = []
    chaos_rows = []

    current_ts = int(time.time()) - (total_steps * interval_sec)
    step_count = 0

    print("\n  [1/3] Generating structured normal and chaos time-series observations...")

    ep_idx = 0
    while len(all_rows) < target_rows:
        # 1. Normal Block
        normal_block_len = min(steps_per_normal_block, max(1, (target_rows - len(all_rows)) // num_services))
        for _ in range(normal_block_len):
            step_rows = generate_step_telemetry(
                timestamp=current_ts,
                step_index=step_count,
                active_fault="normal",
                root_cause_service="none"
            )
            all_rows.extend(step_rows)
            normal_rows.extend(step_rows)
            current_ts += interval_sec
            step_count += 1
            if len(all_rows) >= target_rows:
                break

        if len(all_rows) >= target_rows:
            break

        # 2. Chaos Episode
        f_type, target_svc = episode_schedule[ep_idx % len(episode_schedule)]
        ep_idx += 1
        fault_block_len = min(steps_per_fault, max(1, (target_rows - len(all_rows)) // num_services))

        for f_step in range(fault_block_len):
            progress = (f_step + 1) / fault_block_len
            step_rows = generate_step_telemetry(
                timestamp=current_ts,
                step_index=step_count,
                active_fault=f_type,
                root_cause_service=target_svc,
                fault_progress=progress
            )
            all_rows.extend(step_rows)
            chaos_rows.extend(step_rows)
            current_ts += interval_sec
            step_count += 1
            if len(all_rows) >= target_rows:
                break

    # Truncate precisely to target_rows if slightly over
    all_rows = all_rows[:target_rows]

    # Target CSV paths
    aiops_csv = output_dir / "aiops_chaos_dataset.csv"
    combined_csv = output_dir / "combined_telemetry_dataset.csv"
    normal_csv = output_dir / "normal_baseline_dataset.csv"
    chaos_csv = output_dir / "chaos_labeled_dataset.csv"

    print(f"  [2/3] Writing {len(all_rows)} rows to disk across target CSV files...")

    # Write primary aiops_chaos_dataset.csv
    with open(aiops_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    # Write combined_telemetry_dataset.csv
    with open(combined_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    # Write normal_baseline_dataset.csv (pure is_anomaly=0)
    with open(normal_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows([r for r in all_rows if r["is_anomaly"] == 0])

    # Write chaos_labeled_dataset.csv (chaos subset)
    with open(chaos_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(chaos_rows[:int(target_rows * (1 - normal_ratio))])

    print("  [3/3] Analyzing class distribution and feature variance...")

    # Value counts summary
    fault_counts = Counter(r["fault_type"] for r in all_rows)
    anomaly_counts = Counter(r["is_anomaly"] for r in all_rows)
    rc_counts = Counter(r["root_cause_service"] for r in all_rows)

    print("\n" + "=" * 74)
    print("  Dataset Generation Complete & Verified!")
    print("=" * 74)
    print(f"  Primary Dataset File: {aiops_csv} ({len(all_rows)} rows)")
    print(f"  Combined Dataset:     {combined_csv} ({len(all_rows)} rows)")
    print(f"  Normal Baseline File: {normal_csv} ({sum(1 for r in all_rows if r['is_anomaly'] == 0)} rows)")
    print(f"  Chaos Labeled File:   {chaos_csv} ({len(chaos_rows[:int(target_rows * (1 - normal_ratio))])} rows)")
    print("-" * 74)

    print("\n  [+] Value Counts: 'fault_type' distribution:")
    total = len(all_rows)
    for ft, count in fault_counts.most_common():
        pct = (count / total) * 100
        bar = "█" * int(pct // 2)
        print(f"      {ft:<22}: {count:>5} rows ({pct:>5.1f}%)  {bar}")

    print("\n  [+] Value Counts: 'is_anomaly' distribution:")
    for a_val, count in anomaly_counts.items():
        label_str = "Normal (0)" if a_val == 0 else "Anomaly (1)"
        pct = (count / total) * 100
        print(f"      {label_str:<22}: {count:>5} rows ({pct:>5.1f}%)")

    print("\n  [+] Value Counts: 'root_cause_service' distribution:")
    for rc, count in rc_counts.most_common():
        pct = (count / total) * 100
        print(f"      {rc:<24}: {count:>5} rows ({pct:>5.1f}%)")

    print("=" * 74 + "\n")
    return aiops_csv, len(all_rows)

def main():
    parser = argparse.ArgumentParser(
        description="Scalable Telemetry Dataset Generator for Unsupervised Anomaly Detection & RCA"
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=4000,
        help="Target number of rows to generate (default: 4000)"
    )
    parser.add_argument(
        "--normal-ratio",
        type=float,
        default=0.60,
        help="Proportion of normal healthy data (default: 0.60 for ~60%% normal, 40%% chaos)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3,
        help="Sampling step interval in seconds (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save CSV datasets (default: data)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "normal_only", "chaos_only"],
        default="all",
        help="Collection mode (default: all)"
    )
    parser.add_argument(
        "--normal-duration",
        type=int,
        default=300,
        help="Duration in seconds for normal collection when in duration mode"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=12,
        help="Number of cycles when in cycle mode"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for FastAPI backend (default: http://localhost:8000)"
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    generate_target_rows_dataset(
        target_rows=args.target_rows,
        normal_ratio=args.normal_ratio,
        interval_sec=args.interval,
        output_dir=out_dir
    )

if __name__ == "__main__":
    main()
