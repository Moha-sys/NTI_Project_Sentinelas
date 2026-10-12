#!/usr/bin/env python3
"""
Sentinelas — n8n anomaly webhook simulation script.

Sends three representative anomaly payloads directly to the configured n8n
webhook URL, bypassing the FastAPI backend entirely.  Use this to verify that
the n8n workflow, Telegram bot, and email channel are all working end-to-end
without needing the full Sentinelas stack to be running.

Usage
-----
    # From the repo root:
    python3 n8n/simulate_anomaly.py

    # Override the target URL for a single run:
    N8N_WEBHOOK_URL=http://localhost:4555/webhook/sentinelas-anomaly python3 n8n/simulate_anomaly.py
"""
import asyncio
import json
import os
import sys
import time

try:
    import httpx
except ImportError:
    sys.exit(
        "ERROR: httpx is not installed.\n"
        "Install it with:  pip install httpx\n"
        "Or activate the project venv first."
    )

# ── Configuration ──────────────────────────────────────────────────────────────

N8N_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "http://localhost:4555/webhook/sentinelas-anomaly",
)
TIMEOUT = 10.0   # seconds per request
DELAY   = 1.2    # seconds between payloads

# ── Simulated anomaly scenarios ────────────────────────────────────────────────

SCENARIOS = [
    {
        "_label": "Scenario 1 — CPU throttling (cart-service, high confidence)",
        "service_name":     "cart-service",
        "anomaly_type":     "cpu_throttling@cart-service",
        "confidence_score": 0.91,
        "timestamp":        time.time(),
        "status":           "ANOMALY_DETECTED",
    },
    {
        "_label": "Scenario 2 — Pod OOM killed (payment-service, critical)",
        "service_name":     "payment-service",
        "anomaly_type":     "pod_oom_killed@payment-service",
        "confidence_score": 0.97,
        "timestamp":        time.time() + 2,
        "status":           "ANOMALY_DETECTED",
    },
    {
        "_label": "Scenario 3 — Network latency spike (frontend, medium confidence)",
        "service_name":     "frontend",
        "anomaly_type":     "network_latency@frontend",
        "confidence_score": 0.74,
        "timestamp":        time.time() + 4,
        "status":           "ANOMALY_DETECTED",
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"

def _banner(text: str) -> None:
    width = 66
    print(f"\n{BOLD}{'─' * width}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'─' * width}{RESET}")

def _strip_internal(payload: dict) -> dict:
    """Remove simulation-only keys before sending."""
    return {k: v for k, v in payload.items() if not k.startswith("_")}

# ── Core sender ────────────────────────────────────────────────────────────────

async def send_one(client: httpx.AsyncClient, scenario: dict, index: int) -> bool:
    label   = scenario.get("_label", f"Scenario {index}")
    payload = _strip_internal(scenario)

    print(f"\n{CYAN}[{index}] {label}{RESET}")
    print(f"    {DIM}Target : {N8N_URL}{RESET}")
    print(f"    {DIM}Payload:{RESET}")
    for k, v in payload.items():
        if k == "timestamp":
            v = f"{v}  ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(v))})"
        print(f"           {k}: {v}")

    try:
        resp = await client.post(N8N_URL, json=payload, timeout=TIMEOUT)
        if resp.is_success:
            print(f"    {GREEN}✓  HTTP {resp.status_code} — n8n acknowledged{RESET}")
            try:
                body = resp.json()
                print(f"    {DIM}Response: {json.dumps(body, indent=None)[:160]}{RESET}")
            except Exception:
                pass
            return True
        else:
            print(f"    {RED}✗  HTTP {resp.status_code} — {resp.text[:200]}{RESET}")
            return False
    except httpx.ConnectError:
        print(
            f"    {RED}✗  Connection refused — is n8n running at {N8N_URL}?{RESET}\n"
            f"    {YELLOW}   Start n8n:  docker run -p 5678:5678 n8nio/n8n{RESET}"
        )
        return False
    except httpx.TimeoutException:
        print(f"    {RED}✗  Request timed out after {TIMEOUT}s{RESET}")
        return False
    except Exception as exc:
        print(f"    {RED}✗  Unexpected error: {exc}{RESET}")
        return False


async def main() -> None:
    _banner("Sentinelas × n8n  —  Anomaly Webhook Simulation")
    print(f"  Webhook target : {BOLD}{N8N_URL}{RESET}")
    print(f"  Scenarios      : {len(SCENARIOS)}")
    print(f"  Delay between  : {DELAY}s\n")

    results = []
    async with httpx.AsyncClient() as client:
        for i, scenario in enumerate(SCENARIOS, start=1):
            ok = await send_one(client, scenario, i)
            results.append(ok)
            if i < len(SCENARIOS):
                await asyncio.sleep(DELAY)

    passed = sum(results)
    failed = len(results) - passed

    _banner("Results")
    for i, (scenario, ok) in enumerate(zip(SCENARIOS, results), start=1):
        icon = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{icon}]  {scenario.get('_label', f'Scenario {i}')}")

    print()
    if failed == 0:
        print(
            f"  {GREEN}{BOLD}All {passed}/{len(results)} scenarios delivered successfully.{RESET}\n"
            f"  {GREEN}Check your Telegram chat — alerts should appear momentarily.{RESET}"
        )
    else:
        print(
            f"  {YELLOW}{passed}/{len(results)} scenarios succeeded, "
            f"{failed} failed.{RESET}\n"
            f"  {YELLOW}See messages above for details.{RESET}"
        )

    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
