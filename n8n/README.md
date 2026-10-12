# Sentinelas × n8n — Anomaly Webhook Integration

This directory contains the n8n workflow JSON for routing Sentinelas anomaly
alerts to **Telegram** and **Email** channels.

---

## Architecture

```
Sentinelas Backend
  ML Pipeline (IsolationForest + LightGBM)
        │ anomaly confirmed
        ▼
  POST /api/v1/webhooks/anomaly/trigger   ←── telemetry rows in
        │ background task
        ▼
  n8n Webhook Trigger  (sentinelas-anomaly)   ←  n8n_main : host port 4555
        │
        ├─ confidence ≥ 0.7 ──► Format Message  (HTML + plain text bodies)
        │                             │
        │                             ▼
        │                       Telegram Alert  (parse_mode: HTML)
        │                             │
        │                             ▼
        │                       Email Alert     (Gmail SMTP, App Password)
        │                             │
        │                             └─► Respond 200 OK
        │
        └─ confidence < 0.7 ──► Respond 200 Skipped
```

**Why serial?** `Telegram Alert → Email Alert → Respond 200 OK` is a single
chain. The original graph fanned *both* senders into the same `Respond 200 OK`
node, which makes n8n fire the response twice (`"Response already sent"` error).

---

## Live instance

| Item | Value |
|------|-------|
| Container | `n8n_main` (`n8nio/n8n:latest`, v2.39.10) |
| URL | `http://localhost:4555` (host `4555` → container `5678`) |
| Volume | `/home/nourz/n8n_backup` → `/home/node/.n8n` |
| Workflow id | `sentinelas-anomaly-wf1` |
| Webhook URL | `http://localhost:4555/webhook/sentinelas-anomaly` |
| Executions | `docker logs n8n_main` |

> **Note:** port `4555` and path `sentinelas-anomaly` must match
> `N8N_WEBHOOK_URL` in `backend/.env`. A stale UUID path returns
> `Received request for unknown webhook`.

---

## Payload Schema

Every POST from Sentinelas to n8n carries this JSON body:

| Field                  | Type    | Description                                      |
|------------------------|---------|--------------------------------------------------|
| `service_name`         | string  | Originating microservice (e.g. `cart-service`)   |
| `anomaly_type`         | string  | Fault classification (e.g. `cpu_throttling@cart-service`) |
| `confidence_score`     | float   | Anomaly probability in [0, 1]                    |
| `timestamp`            | float   | Unix epoch of the detection event                |
| `status`               | string  | Alert lifecycle — `ANOMALY_DETECTED` by default  |
| `anomaly_score`        | float   | Raw IsolationForest decision score (negative = anomaly) |
| `root_cause_service`   | string  | ML-predicted root-cause service                  |
| `root_cause_fault_type`| string  | ML-predicted fault class                         |

Example payload:
```json
{
  "service_name": "cart-service",
  "anomaly_type": "cpu_throttling@cart-service",
  "confidence_score": 0.87,
  "timestamp": 1718000000.0,
  "status": "ANOMALY_DETECTED",
  "anomaly_score": -0.1234,
  "root_cause_service": "cart-service",
  "root_cause_fault_type": "cpu_throttling"
}
```

---

## Credentials (secrets live here, never in git)

Both credentials are stored **encrypted** in the n8n database. They were created
with `n8n import:credentials`, which encrypts plaintext with the instance key in
`/home/nourz/n8n_backup/config`.

| Credential | id | type | Contents |
|------------|----|------|----------|
| Sentinelas Alert Bot | `sentinelas-telegram-bot` | `telegramApi` | BotFather access token |
| Gmail SMTP | `sentinelas-gmail-smtp` | `smtp` | `smtp.gmail.com:587`, user + **16-char Gmail App Password** |

Create/replace one (n8n must be stopped):

```bash
cat > /tmp/cred.json <<'JSON'
[ { "id": "sentinelas-gmail-smtp", "name": "Gmail SMTP", "type": "smtp",
    "data": { "user": "nourhanelshieikh71@gmail.com", "password": "xxxx xxxx xxxx xxxx",
              "host": "smtp.gmail.com", "port": 587, "secure": false,
              "starttls": true, "disableStartTls": false } } ]
JSON

docker stop n8n_main
docker run --rm -v /home/nourz/n8n_backup:/home/node/.n8n -v /tmp:/input \
  --entrypoint n8n n8nio/n8n:latest import:credentials --input=/input/cred.json
docker start n8n_main
```

> **Why not `$vars`?** The `variables` table in this instance is empty, so the
> original `{{ $vars.TELEGRAM_CHAT_ID }}` / `$vars.ALERT_*_EMAIL` expressions
> resolved to nothing. Chat ID and mail addresses are set inline on the nodes;
> the bot token stays only in the encrypted credential.

---

## Deploying / updating the workflow

CLI changes need a restart before the webhook is registered:

```bash
# 1. stop, import, activate, start
docker stop n8n_main

docker run --rm -v /home/nourz/n8n_backup:/home/node/.n8n \
  -v "$PWD/n8n:/input" --entrypoint n8n n8nio/n8n:latest \
  import:workflow --input=/input/sentinelas_n8n_workflow.json

docker run --rm -v /home/nourz/n8n_backup:/home/node/.n8n \
  --entrypoint n8n n8nio/n8n:latest \
  update:workflow --id=sentinelas-anomaly-wf1 --active=true

docker run --rm -v /home/nourz/n8n_backup:/home/node/.n8n \
  --entrypoint n8n n8nio/n8n:latest \
  publish:workflow --id=sentinelas-anomaly-wf1

docker start n8n_main
```

Confirm the route exists:

```bash
docker logs n8n_main 2>&1 | grep -i webhook
```

---

## Configure Sentinelas

`backend/.env` (gitignored):

```
N8N_WEBHOOK_URL=http://localhost:4555/webhook/sentinelas-anomaly
```

---

## API Endpoints

### `POST /api/v1/webhooks/anomaly/trigger`

Runs the full ML pipeline on a telemetry snapshot.  
Fires the n8n webhook **only when an anomaly is confirmed**.

```bash
curl -s -X POST http://localhost:8000/api/v1/webhooks/anomaly/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {
        "service_name": "cart-service",
        "cpu_percent": 95.0,
        "memory_mb": 1800.0,
        "latency_ms": 2400.0,
        "error_rate": 0.45,
        "service_up": 1
      }
    ]
  }' | python3 -m json.tool
```

**Response (anomaly found):**
```json
{
  "triggered": true,
  "n8n_notified": true,
  "payload": {
    "service_name": "cart-service",
    "anomaly_type": "network_latency@cart-service",
    "confidence_score": 0.6721,
    "timestamp": 1791827919.19,
    "status": "ANOMALY_DETECTED",
    "anomaly_score": -0.1794,
    "root_cause_service": "cart-service",
    "root_cause_fault_type": "network_latency"
  },
  "message": "Anomaly detected — n8n webhook notification dispatched."
}
```

> ⚠️ `n8n_notified: true` only means n8n accepted the POST. n8n still applies the
> **confidence ≥ 0.7** gate: the run above produced `0.6721`, so it was
> **deliberately suppressed** (`"routed": false`) and no Telegram message was sent.

---

### `POST /api/v1/webhooks/anomaly/notify`

Bypasses the ML pipeline — forwards a pre-formed payload directly to n8n.  
Useful for integration testing or external alert sources.

```bash
curl -s -X POST http://localhost:8000/api/v1/webhooks/anomaly/notify \
  -H "Content-Type: application/json" \
  -d '{
    "service_name": "payment-service",
    "anomaly_type": "pod_oom_killed",
    "confidence_score": 0.92,
    "status": "ANOMALY_DETECTED"
  }' | python3 -m json.tool
```

---

## End-to-end test (no backend needed)

```bash
python3 n8n/simulate_anomaly.py
```

Fires three scenarios straight at n8n:

| # | Scenario | Confidence | Result |
|---|----------|-----------|--------|
| 1 | CPU throttling — cart-service | 0.91 | sent ✅ |
| 2 | Pod OOM killed — payment-service | 0.97 | sent ✅ |
| 3 | Network latency — frontend | 0.74 | sent ✅ |

Direct webhook check:

```bash
# below threshold → suppressed
curl -s -X POST http://localhost:4555/webhook/sentinelas-anomaly \
  -H "Content-Type: application/json" \
  -d '{"service_name":"cart-service","anomaly_type":"network_latency@cart-service","confidence_score":0.6721}'
# → {"received":true,"routed":false,"reason":"confidence_score below threshold"}

# above threshold → Telegram
curl -s -X POST http://localhost:4555/webhook/sentinelas-anomaly \
  -H "Content-Type: application/json" \
  -d '{"service_name":"payment-service","anomaly_type":"pod_oom_killed@payment-service","confidence_score":0.92}'
# → {"received":true,"routed":true,...,"telegram_sent":true}
```

---

## Telegram message format

`parse_mode` is **HTML**, not Markdown. Markdown (`_`) breaks on fault labels
like `cpu_throttling` and Telegram returns
`can't parse entities: Can't find end of the entity`.

```
🚨 Sentinelas Anomaly Alert          ← bold

Service: cart-service                ← labels bold
Type: cpu_throttling@cart-service
Confidence: 91.0%
Status: ANOMALY_DETECTED
Time: 2026-10-12T17:10:24.000Z
Root Cause: cart-service — cpu_throttling
```

---

## Extending the workflow

The n8n workflow is intentionally minimal.  
Fork it inside n8n to add extra branches, e.g.:

- **PagerDuty** — use the HTTP Request node to call PagerDuty Events API v2.
- **Slack** — use the built-in Slack node with `#incidents` channel.
- **Jira** — auto-create an incident ticket using the Jira Software node.
- **Suppress low-severity** — add an extra IF node filtering `anomaly_type`.
- **Tune the gate** — the `Confidence ≥ 0.7 ?` node; the current LightGBM model
  emits probabilities around 0.67 for clear anomalies, so 0.7 suppresses them.
