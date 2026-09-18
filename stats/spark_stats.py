#!/usr/bin/env python3
"""
spark-stats - a tiny metrics endpoint for a machine that serves models.

Designed to run **inside the model container** (or standalone on a bare box) and
report:

  * identity  - hostname, tailscale name, uptime, version
  * host      - CPU %, memory/swap, load, disk
  * GPU       - utilisation, memory, temperature, power (NVIDIA, optional)
  * inference - for each configured target: liveness, loaded models, context,
                slots, and tokens/sec, draft acceptance, TTFT and request
                counts parsed from the server's own logs (or Prometheus
                `/metrics` when the server provides it).

Standard library only, one process, a few MB of RSS. It polls in a background
thread so HTTP requests are served from cache instantly.

Standalone:
    python3 spark_stats.py --config config/stats.standalone.json

Inside the model image (defaults suit a TabbyAPI container):
    python3 spark_stats.py --config /opt/spark-stats.json &

The dashboard (dashboard/server.py) scrapes GET /api/stats from every machine.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "0.2.0"
_STARTED = time.time()

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _read(path: str, default: str = "") -> str:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return default


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def _num(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# host collectors
# --------------------------------------------------------------------------- #


class CpuSampler:
    """CPU % from two /proc/stat samples."""

    def __init__(self) -> None:
        self._prev = None

    def sample(self) -> float:
        parts = _read("/proc/stat").split("\n", 1)[0].split()
        if len(parts) < 5:
            return 0.0
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        if self._prev is None:
            pct = 0.0
        else:
            dt, di = total - self._prev[0], idle - self._prev[1]
            pct = 100.0 * (dt - di) / dt if dt > 0 else 0.0
        self._prev = (total, idle)
        return round(pct, 1)


def collect_mem() -> dict:
    info = {}
    for line in _read("/proc/meminfo").splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if fields:
            info[key] = int(fields[0])
    gib = lambda kb: round(kb / 1048576, 1)  # noqa: E731
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", 0)
    used = max(total - avail, 0)
    swap_total = info.get("SwapTotal", 0)
    swap_free = info.get("SwapFree", 0)
    return {
        "total_gb": gib(total),
        "used_gb": gib(used),
        "available_gb": gib(avail),
        "pct": round(100 * used / total, 1) if total else 0.0,
        "swap_total_gb": gib(swap_total),
        "swap_used_gb": gib(max(swap_total - swap_free, 0)),
    }


def collect_load() -> dict:
    parts = _read("/proc/loadavg").split()
    up = _read("/proc/uptime").split()
    return {
        "load1": _num(parts[0]) if parts else None,
        "load5": _num(parts[1]) if len(parts) > 1 else None,
        "load15": _num(parts[2]) if len(parts) > 2 else None,
        "uptime_s": int(float(up[0])) if up else None,
        "cores": os.cpu_count(),
    }


def collect_gpu() -> dict | None:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout=4.0,
    ).strip()
    if not out:
        return None
    fields = [f.strip() for f in out.splitlines()[0].split(",")]
    if len(fields) < 6:
        return None
    mem_used, mem_total = _num(fields[2]), _num(fields[3])
    return {
        "name": fields[0],
        "util_pct": _num(fields[1]),
        "mem_used_gb": round(mem_used / 1024, 1) if mem_used is not None else None,
        "mem_total_gb": round(mem_total / 1024, 1) if mem_total is not None else None,
        "temp_c": _num(fields[4]),
        "power_w": _num(fields[5]),
    }


def collect_disk(path: str = "/") -> dict | None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return {
        "path": path,
        "total_gb": round(usage.total / 2**30, 1),
        "used_gb": round(usage.used / 2**30, 1),
        "pct": round(100 * usage.used / usage.total, 1),
    }


class TailscaleName:
    """Cached tailscale DNS name (avoids shelling out every poll)."""

    def __init__(self, ttl: float = 120.0) -> None:
        self._ttl = ttl
        self._at = 0.0
        self._value = None

    def get(self) -> str | None:
        now = time.time()
        if now - self._at > self._ttl:
            self._at = now
            try:
                data = json.loads(_run(["tailscale", "status", "--json"], 5.0) or "{}")
                name = (data.get("Self") or {}).get("DNSName") or ""
                self._value = name.rstrip(".") or None
            except (ValueError, KeyError):
                self._value = None
        return self._value


# --------------------------------------------------------------------------- #
# inference target probes
# --------------------------------------------------------------------------- #


def _http_get(url: str, key: str | None = None, timeout: float = 3.0) -> str:
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


_GEN_RE = re.compile(r"([\d,]+)\s+tokens generated\s+at\s+([\d.,]+)\s*T/s")
_DRAFT_RE = re.compile(r"draft\s+(\d+)/(\d+)\s+accepted\s+\((\d+)%\)")
_TOTAL_RE = re.compile(r"total\s+([\d.]+)\s*s")
_ISO_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)
_CLOCK_RE = re.compile(r"\b(\d{2}):(\d{2}):(\d{2})[.,](\d{1,6})")
_BLOCK_SPLIT = re.compile(
    r"(?=\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\b\d{2}:\d{2}:\d{2}[.,]\d{3}\b)"
)


def _tzinfo(name: str | None):
    """Resolve a tz name ('UTC', 'Asia/Bangkok', ...) to a tzinfo, or None."""
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        return dt.timezone.utc if name.upper() == "UTC" else None


def _to_epoch(block: str, tz=None) -> float | None:
    """Timestamp of a log block, ISO (docker --timestamps) or bare clock time.

    Container logs are usually UTC while the host may not be, so prefer a real
    ISO instant when one is present. A naive timestamp is read in `tz` (default:
    host local time).
    """
    m = _ISO_RE.search(block)
    if m:
        text = m.group(1).replace(",", ".")
        # fromisoformat wants at most 6 fractional digits; docker emits 9
        text = re.sub(r"(\.\d{6})\d+", r"\1", text)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            stamp = dt.datetime.fromisoformat(text)
            if stamp.tzinfo is None and tz is not None:
                stamp = stamp.replace(tzinfo=tz)
            return stamp.timestamp()
        except ValueError:
            pass
    m = _CLOCK_RE.search(block)
    if m:
        now = dt.datetime.now()
        frac = float("0." + m.group(4)) if m.group(4) else 0.0
        cand = now.replace(
            hour=int(m.group(1)), minute=int(m.group(2)), second=int(m.group(3)), microsecond=0
        )
        if tz is not None:
            cand = cand.replace(tzinfo=tz)
        epoch = cand.timestamp()
        if epoch > time.time() + 60:
            epoch -= 86400
        return epoch + frac
    return None


_LINE_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+\|?\s*(.*)$",
    re.M,
)


def _records(raw: str, tzinfo=None) -> list[tuple[str | None, str]]:
    """Split log text into (timestamp, message) records.

    Two shapes are supported:
      * `docker logs --timestamps`, where every printed line (including wrapped
        continuation lines) is prefixed with a nanosecond timestamp. Fragments of
        one record are emitted microseconds apart, so lines are grouped while
        their timestamps stay within a small tolerance.
      * plain log files, where a record starts at a timestamp token and the
        message is on one line.
    """
    if _LINE_TS_RE.search(raw):
        groups: list[tuple[str | None, str]] = []
        cur: list[str] = []
        cur_ts: str | None = None
        group_epoch: float | None = None
        for line in raw.splitlines():
            match = _LINE_TS_RE.match(line)
            if not match:
                if cur:
                    cur.append(line)
                continue
            ts_text, text = match.group(1), match.group(2)
            epoch = _to_epoch(ts_text, tzinfo)
            if (
                cur
                and group_epoch is not None
                and epoch is not None
                and epoch - group_epoch > 0.05
            ):
                groups.append((cur_ts, " ".join(cur)))
                cur, cur_ts, group_epoch = [], None, None
            if not cur:
                cur_ts, group_epoch = ts_text, epoch
            cur.append(text)
        if cur:
            groups.append((cur_ts, " ".join(cur)))
        return groups
    return [(None, block) for block in _BLOCK_SPLIT.split(raw)]


def parse_records(raw: str, tz=None) -> list[dict]:
    """Every generation record in the log text (server-side accounting)."""
    tzinfo = _tzinfo(tz) if isinstance(tz, str) else tz
    records = []
    for ts_text, block in _records(raw, tzinfo):
        gen = _GEN_RE.search(block)
        if not gen:
            continue
        draft = _DRAFT_RE.search(block)
        total = _TOTAL_RE.search(block)
        records.append(
            {
                "ts": _to_epoch(ts_text or block, tzinfo),
                "tokens": int(gen.group(1).replace(",", "")),
                "tps": _num(gen.group(2).replace(",", "")),
                "accept_pct": float(draft.group(3)) if draft else None,
                "total_s": _num(total.group(1)) if total else None,
            }
        )
    return records


def parse_completions(raw: str, window_s: float = 300.0, tz=None) -> dict:
    """Aggregate stats over the recent window from generation log lines."""
    records = parse_records(raw, tz)
    if not records:
        return {}
    cutoff = time.time() - window_s
    recent = [r for r in records if r["ts"] is None or r["ts"] >= cutoff]
    # The newest record in a tail can be cut mid-message; prefer the last complete one.
    complete = [r for r in records if r["total_s"] is not None]
    latest = complete[-1] if complete else records[-1]
    gen_time = sum(r["total_s"] or 0.0 for r in recent)
    out = {
        "tps": latest["tps"],
        "draft_accept_pct": latest["accept_pct"],
        "last_total_s": latest["total_s"],
        "requests_window": len(recent),
        "tokens_window": sum(r["tokens"] for r in recent),
        "window_s": int(window_s),
    }
    if gen_time > 0:
        out["avg_tps_window"] = round(sum(r["tokens"] for r in recent) / gen_time, 1)
    return out


def _load_key_file(path: str) -> str | None:
    """Read the first API key out of a TabbyAPI-style api_tokens.yml."""
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return None
    match = re.search(r'api_key:\s*(?:\n\s*-\s*)?"([^"]+)"', text)
    return match.group(1) if match else None


def _log_lines(spec: dict) -> str:
    kind = spec.get("type", "file")
    lines = int(spec.get("lines", 400))
    if kind == "docker":
        return _run(
            ["docker", "logs", "--timestamps", "--tail", str(lines), spec["container"]],
            8.0,
        )
    if kind == "journal":
        return _run(
            ["journalctl", "-u", spec["unit"], "--no-pager", "-n", str(lines)], 8.0
        )
    if kind == "file":
        path = spec["path"]
        try:
            if os.path.isdir(path):
                logs = [
                    os.path.join(path, name)
                    for name in os.listdir(path)
                    if name.endswith(".log")
                ]
                if not logs:
                    return ""
                path = max(logs, key=os.path.getmtime)
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 262144))
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return ""
    return ""


def _prom_counters(text: str) -> dict:
    def counter(name: str):
        m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)", text, re.M)
        return float(m.group(1)) if m else None

    def gauge(name: str):
        return counter(name)

    return {
        "running": gauge("vllm:num_requests_running"),
        "queue": gauge("vllm:num_requests_waiting"),
        "gen_tokens_total": counter("vllm:generation_tokens_total"),
        "prompt_tokens_total": counter("vllm:prompt_tokens_total"),
    }


class TargetProbe:
    """Probes one inference endpoint; keeps previous counters for rate maths."""

    def __init__(self, spec: dict, secrets: dict) -> None:
        self.spec = spec
        self.secrets = secrets
        self._prev = None  # (ts, gen_total, prompt_total)
        self._lock = threading.Lock()

    def _key(self) -> str | None:
        spec = self.spec
        if spec.get("key"):
            return spec["key"]
        env = spec.get("key_env")
        if env and env in self.secrets:
            return self.secrets[env]
        if spec.get("key_file"):
            return _load_key_file(spec["key_file"])
        return None

    def probe(self) -> dict:
        spec = self.spec
        base = spec["url"].rstrip("/")
        key = self._key()
        result = {
            "name": spec.get("name") or base,
            "type": spec.get("type", "openai"),
            "url": base,
            "up": False,
        }
        started = time.time()
        try:
            models = json.loads(_http_get(f"{base}/v1/models", key, 3.0))
            result["up"] = True
            result["models"] = [
                m.get("id") for m in models.get("data", []) if isinstance(m, dict)
            ]
            result["latency_ms"] = int((time.time() - started) * 1000)
        except (urllib.error.URLError, ValueError, OSError) as exc:
            result["error"] = str(exc)[:160]
            return result

        # TabbyAPI / KoboldCPP style props
        try:
            props = json.loads(_http_get(f"{base}/props", key, 3.0))
            if isinstance(props, dict):
                if isinstance(props.get("total_slots"), int):
                    result["slots"] = props["total_slots"]
                gen = props.get("default_generation_settings") or {}
                if gen.get("n_ctx"):
                    result["ctx"] = gen["n_ctx"]
                if props.get("model_path"):
                    result["model_path"] = props["model_path"]
        except (urllib.error.URLError, ValueError, OSError):
            pass

        # Prometheus metrics (vLLM / SGLang)
        try:
            text = _http_get(f"{base}/metrics", key, 3.0)
            counters = _prom_counters(text)
            result["running"] = int(counters["running"]) if counters["running"] is not None else None
            result["queue"] = int(counters["queue"]) if counters["queue"] is not None else None
            now = time.time()
            with self._lock:
                if self._prev and counters["gen_tokens_total"] is not None:
                    dt = now - self._prev[0]
                    if dt > 0:
                        result["tps"] = round(
                            (counters["gen_tokens_total"] - self._prev[1]) / dt, 1
                        )
                        if counters["prompt_tokens_total"] is not None:
                            result["prefill_tps"] = round(
                                (counters["prompt_tokens_total"] - self._prev[2]) / dt, 1
                            )
                self._prev = (
                    now,
                    counters["gen_tokens_total"],
                    counters["prompt_tokens_total"],
                )
            result["tokens_total"] = (
                int(counters["gen_tokens_total"])
                if counters["gen_tokens_total"] is not None
                else None
            )
        except (urllib.error.URLError, ValueError, OSError):
            pass

        # Log-derived stats (works when the server has no /metrics)
        log_spec = spec.get("log")
        if log_spec:
            parsed = parse_completions(
                _log_lines(log_spec),
                window_s=spec.get("window_s", 300),
                tz=log_spec.get("tz"),
            )
            if parsed:
                result.update(parsed)
        return result


# --------------------------------------------------------------------------- #
# state + poll loop
# --------------------------------------------------------------------------- #


class State:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.cpu = CpuSampler()
        self.tailscale = TailscaleName()
        self.poll_seconds = float(config.get("poll_seconds", 2.0))
        secrets = {}
        for env in config.get("secrets", []):
            if env in os.environ:
                secrets[env] = os.environ[env]
        self.probes = [TargetProbe(t, secrets) for t in config.get("targets", [])]
        self.latest: dict = {}
        self._lock = threading.Lock()

    def collect(self) -> dict:
        config = self.config
        sample = {
            "identity": {
                "hostname": socket.gethostname(),
                "tailscale": self.tailscale.get(),
                "version": VERSION,
                "agent_uptime_s": int(time.time() - _STARTED),
            },
            "host": {
                "cpu": {"pct": self.cpu.sample(), **collect_load()},
                "mem": collect_mem(),
                "disk": collect_disk(config.get("disk_path", "/")),
            },
            "gpu": collect_gpu(),
            "targets": [p.probe() for p in self.probes],
            "ts": time.time(),
        }
        with self._lock:
            self.latest = sample
        return sample

    def loop(self) -> None:
        while True:
            try:
                self.collect()
            except Exception as exc:  # never let the poller die
                with self._lock:
                    self.latest = {"error": str(exc)[:200], "ts": time.time()}
            time.sleep(self.poll_seconds)

    def get(self) -> dict:
        with self._lock:
            return self.latest or self.collect()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class Handler(BaseHTTPRequestHandler):
    server_version = f"spark-stats/{VERSION}"
    state: State

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/api/stats", "/stats", "/"):
            body = json.dumps(self.state.get(), default=str).encode()
            self._send(200, body, "application/json")
        elif path in ("/healthz", "/health"):
            self._send(200, b'{"ok":true}', "application/json")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")

    def log_message(self, *_args) -> None:  # keep the journal quiet
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="spark-stats metrics endpoint")
    parser.add_argument("--config", default=os.environ.get("SPARK_DASH_CONFIG", "config/agent.example.json"))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    state = State(config)
    threading.Thread(target=state.loop, daemon=True).start()
    state.collect()  # warm the cache before serving

    host = args.host or config.get("listen", "0.0.0.0")
    port = args.port or int(config.get("port", 8787))
    httpd = ThreadingHTTPServer((host, port), Handler)
    Handler.state = state
    print(f"spark-stats {VERSION} on http://{host}:{port} (config {args.config})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
