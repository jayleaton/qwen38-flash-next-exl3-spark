# The spark image standard

Every model image we build follows this contract so the dashboard can discover,
report and control it uniformly, on any number of machines.

## What a model image provides

1. **An OpenAI-compatible API** on a known port (e.g. `5000`).
2. **A metrics endpoint** on a known port (e.g. `8787`): `GET /api/stats`, served
   by the bundled `spark-stats` process. It reports the machine (CPU, memory,
   GPU, disk) and this model (loaded name, context, slots, tokens/sec, draft
   acceptance, request counts).
3. **Docker labels** so a host agent can find and control it.
4. **A predictable log location** so `spark-stats` can derive tokens/sec without
   the server exposing Prometheus metrics. TabbyAPI writes `logs/*.log`; for
   vLLM/SGLang, point `spark-stats` at `/metrics` instead.

## Required Docker labels

```yaml
labels:
  spark.model: "qwen38-flash-next"                 # required: enables discovery
  spark.name: "Qwen3.8-Flash-Next (uncensored EXL3)"
  spark.stats_port: "8787"                         # in-container /api/stats
  spark.inference_port: "5000"                     # OpenAI port
  # optional, for compose-based control instead of `docker start/stop`:
  # spark.compose_file: "/abs/path/docker-compose.yml"
  # spark.service: "qwen"
```

## Required ports (convention)

| Port | Purpose | Publish as |
| --- | --- | --- |
| `5000` | OpenAI-compatible API | `127.0.0.1:5000:5000` (loopback; expose via `tailscale serve`) |
| `8787` | `spark-stats` metrics | `127.0.0.1:8787:8787` |

Loopback by default; the tailnet provides transport and identity. Do not bind
`0.0.0.0` on the host unless you also firewall it.

## The stats payload (`GET /api/stats`)

```json
{
  "identity": {"hostname": "dgx-01", "tailscale": "dgx-01.<tailnet>.ts.net", "version": "0.2.0"},
  "host": {"cpu": {...}, "mem": {...}, "disk": {...}},
  "gpu": {"name": "NVIDIA GB10", "util_pct": 93.0, "temp_c": 77, "power_w": 67},
  "targets": [
    {
      "name": "Qwen3.8-Flash-Next (uncensored EXL3)",
      "up": true,
      "models": ["qwen38-flash-next-unc-3bpw"],
      "ctx": 262144, "slots": 4,
      "tps": 79.9, "draft_accept_pct": 71.6,
      "requests_window": 3, "tokens_window": 1200
    }
  ],
  "ts": 1789709814
}
```

`spark-host` (per machine) merges this into:

```json
{"identity": {...}, "host": {...}, "gpu": {...},
 "models": [{"id": "...", "name": "...", "container": "...", "state": "running",
             "stats_port": 8787, "metrics": { ...the target above... }}]}
```

and adds control endpoints `POST /api/models/<id>/{start,stop,restart}`.

## Why labels + a host agent, not in-container control

A container cannot start or stop its siblings without the Docker socket, and
machine metrics belong to the host. So:

- **in the image**: model/server metrics (`spark-stats`).
- **on the host**: machine metrics + discovery + control (`spark-host`).
- **on your main machine**: the aggregate UI (`dashboard/`).

## Adding a new model image

1. Copy `spark_stats.py` in and run it in the entrypoint on `8787` with a config
   that points at the local server (see `stats/container.json`).
2. Add the labels above to the compose file.
3. Publish both ports on loopback.
4. Point it at the server's log file (or `/metrics`) for token accounting.
5. `spark-host` picks it up automatically — no dashboard change needed.
