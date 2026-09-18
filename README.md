# Qwen3.8-Flash-Next (uncensored EXL3) — one DGX Spark

A single `docker compose up` away from serving **Qwen3.8-Flash-Next** — the uncensored EXL3 re-quant
— on an NVIDIA DGX Spark (GB10, aarch64), with an **OpenAI-compatible API** and speculative decoding
on. Same workflow as the GLM-5.3-Flash DGX image: get the image, mount the model, talk to it like a
cloud model.

```
docker compose up -d          # after setting MODEL_DIR in .env
curl http://127.0.0.1:5000/v1/chat/completions ...
```

Measured **79.6–79.9 tok/s** single-stream (greedy), **71.6%** draft acceptance, full 262,144-token
cache. See [Expected results](#expected-results).

## What you get

| | |
| --- | --- |
| Model | [`Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw`](https://huggingface.co/Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw) — EXL3 3.05 bpw, 72.4 GB |
| Architecture | `Qwen4ExpForConditionalGeneration` — 125B LM + 51B n-gram + 4B MTP, **6B active**/token |
| Engine | exllamav3 fork [`vcruz305/exllamav3@523ecd3`](https://github.com/vcruz305/exllamav3/commit/523ecd3) (v1.5.0) compiled for **sm_121** |
| Server | [TabbyAPI](https://github.com/theroyallab/tabbyAPI) — `/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming, tool calls |
| Spec decode | MTP drafter (`ndt=5`), dynamic draft, 8-bit KV |
| Context | 262,144 native (needle exact to ~240k) |

---

## Setup

### 0. Prerequisites

- A **DGX Spark (GB10, aarch64)** with the NVIDIA driver, Docker, and `nvidia-container-toolkit`.
- Verify the GPU reaches containers:
  ```bash
  docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
  ```
- ~75 GB free disk for the model (plus ~22 GB for the image).

### 1. Get the model (~72 GB, resumable)

```bash
git clone https://github.com/jayleaton/qwen38-flash-next-exl3-spark
cd qwen38-flash-next-exl3-spark
./scripts/download-model.sh ~/ai/models/qwen38-flash-next-unc-3bpw
```

`download-model.sh` prints the path it wrote. It needs the HF CLI
(`pip install -U "huggingface_hub[cli]"`).

### 2. Configure

```bash
cp .env.example .env
cp api_tokens.yml.example api_tokens.yml     # replace BOTH placeholder keys
$EDITOR .env                                 # set MODEL_DIR to the path from step 1
```

Generate real keys with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`. TabbyAPI
hot-reloads `api_tokens.yml`, so you can rotate keys without a restart.

### 3. Run

```bash
docker compose up -d
docker compose logs -f          # watch "Model loaded" and the API key lines
```

First start builds nothing (if you pulled the image) but does load 66 GB of weights: ~20–60 s. The
entrypoint drops the model's page cache first, which is required on GB10 (see Troubleshooting).

### 4. Verify

```bash
KEY=$(grep -A1 'api_key:' api_tokens.yml | tail -1 | tr -d ' -"')

# 1) the model is listed, with the full context window
curl -s http://127.0.0.1:5000/v1/models -H "Authorization: Bearer $KEY"
#    -> id "qwen38-flash-next-unc-3bpw", n_ctx 262144

# 2) a real completion
curl -s http://127.0.0.1:5000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Say hello in one short sentence."}],
       "max_tokens":256,"temperature":0}'
#    -> content "Hello!" (plus a reasoning_content block)
```

---

## Expected results

One DGX Spark (GB10, 121 GiB unified), single stream, greedy, 400 new tokens:

| Run | Decode | Draft acceptance |
| --- | ---: | ---: |
| code prompt, MTP drafter on | **79.9 tok/s** | 71.6% |
| same, no drafter (baseline) | 31.5 tok/s | — |

| Resource | Expect |
| --- | --- |
| Cold load (container) | ~20–60 s |
| Prefill | ~1,150 tok/s |
| Weights resident | 66.75 GB |
| n-gram table (resident) | ~20 GB |
| KV cache at 262,144 | ~3 GB (8-bit KV, 12 KB/token) |
| Total | ~90 GB of 121 GiB |

Context: the full 262,144 is configured. Needle retrieval is exact to ~240k and fails past 262k —
that is the model's trained window, not a configuration limit.

**Two things change the shape of the numbers:**

- **Past ~163,840 prompt tokens**, MTP acceptance collapses to zero and speculation becomes a small
  loss. This is a property of the model. Agent workloads that stay under ~160k keep the full speed.
- **Prose is slower than code** (~53 vs ~80 tok/s) because n-gram/MTP acceptance is lower on natural
  language. Code, logs, and structured text are the fast path.

If your decode is ~20 tok/s rather than ~80, jump straight to Troubleshooting — it is almost always
`ngram_ram` or the page cache.

---

## Using it like an OpenAI endpoint

### From another machine (Mac / Hermes / opencode)

Keep the container on loopback and let Tailscale provide transport + identity:

```bash
# on the Spark
tailscale serve --bg 5000            # https://<spark>.<tailnet>.ts.net  ->  127.0.0.1:5000
tailscale serve status
```

```bash
# on the client
export OPENAI_BASE_URL="https://<spark>.<tailnet>.ts.net/v1"
export OPENAI_API_KEY="$KEY"
```

Any OpenAI client works. Python:

```python
from openai import OpenAI
client = OpenAI()   # reads OPENAI_BASE_URL / OPENAI_API_KEY
r = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content)
```

A runnable version is [`examples/openai_client.py`](examples/openai_client.py).
`scripts/start.sh` wraps `docker compose up` and (with `SERVE=1`) the Tailscale command.

### Client notes

- **Model ids** accepted in the `model` field: `qwen38-flash-next-unc-3bpw`, or the aliases
  `Qwen3.8-Flash-Next-Uncensored-exl3-3bpw` / `qwen3.8-flash-next`.
- **Thinking is on by default.** The model returns `reasoning_content` (a `<think>…</think>` block)
  alongside `content`. Disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}`.
- **Give it budget.** `max_tokens >= 1024` for tool calls; with thinking on, a small `max_tokens` can
  be consumed by the reasoning block and leave `content` empty.
- **Streaming** works normally (`"stream": true`).
- **Tools** use the Qwen format (`tool_format: qwen3_coder` in `config.yml`).

---

## Configuration

Everything lives in [`config.yml`](config.yml) (TabbyAPI). The parts that matter:

```yaml
model:
  cache_size: 262144      # trained window
  cache_mode: "8,8"       # 8-bit KV: 12 KB/token, ~3 GB for the full window
  ngram_ram: true         # REQUIRED for this pack -- see below
  tensor_parallel: false  # Qwen4Exp raises NotImplementedError under TP
  tool_format: qwen3_coder
  reasoning: true
draft_model:
  draft_mode: mtp         # the model's own MTP head as the drafter
  draft_num_tokens: 5
  dynamic_draft: true
```

Kernel tuning is baked into the image as environment variables (GB10-specific):
`EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_GR_INT8=1 EXL3_MTP_HEAD_N=65536 EXL3_NGRAM_STREAM=0`.
Override them in `compose.yaml` under `environment:` if you want to experiment.

### The one setting you must not drop: `ngram_ram: true`

This pack ships its 51B n-gram (PLE) embedding table as a **single unsharded 19.8 GB file**.
exllamav3's default streams its rows from disk on every forward; on this pack that costs ~35 ms per
speculative round of host time and collapses decode from **~80 tok/s to ~21 tok/s**. `ngram_ram: true`
pins the table in RAM (~20 GB, trivial on 121 GiB) and restores full speed.
Measurements: [docs/FINDINGS.md](docs/FINDINGS.md).

---

## Troubleshooting

**First stop for any problem:** `docker compose logs --tail 100`.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Decode ≈ **20 tok/s** instead of ≈ 80 | `ngram_ram` is not `true`, or an env override disabled it | Confirm `ngram_ram: true` in `config.yml`; do not set `EXL3_NGRAM_STREAM=1`. See FINDINGS. |
| `Insufficient VRAM in split for model and cache` at load | exllamav3's autosplit reads `MemFree` (not `MemAvailable`); the 72 GB download left page cache | The entrypoint fadvises the model pages. If it still fails, on the **host** with no model running: `sync; echo 3 \| sudo tee /proc/sys/vm/drop_caches`, then `docker compose restart`. |
| Container restart-loops with `ModuleNotFoundError: uvloop` | an old image built before the arm64 fix | Rebuild/pull the current image — it installs `uvloop` explicitly (TabbyAPI gates it to x86_64). |
| `ValueError: Failed to load config.json from /model/…` | wrong mount layout | TabbyAPI resolves `model_dir / model_name`. Mount the pack at `/models/qwen38-flash-next-unc-3bpw` (as `compose.yaml` does). |
| `entrypoint: no model at /models/…` | `MODEL_DIR` unset or wrong | Set it in `.env`; confirm with `docker compose config \| grep -A2 volumes`. |
| `401 Unauthorized` | key mismatch | Use the key from `api_tokens.yml` (or the one TabbyAPI prints at startup). Editing the file hot-reloads it. |
| `address already in use :5000` | port conflict | Change `network.port` in `config.yml` and the host mapping in `compose.yaml`. |
| Can't reach it from the Mac | container is on loopback | `tailscale serve --bg 5000`, then use the `https://…ts.net/v1` URL. |
| GPU not visible / `could not select device driver` | missing `nvidia-container-toolkit` | Install it, then re-test with the `docker run --gpus all … nvidia-smi` command above. |
| Very slow first token on a long prompt | cold prefill (~1,150 tok/s, so a 100k prompt takes ~90 s) | Expected. Reuse the same session/prefix so the cache is warm. |
| Output is cut off with empty `content` | `max_tokens` consumed by the reasoning block | Raise `max_tokens` (≥1024), or set `enable_thinking: false`. |
| Container OOM-killed the host | unified memory pressure | See the memory-protection notes in the parent project; keep other large processes off the box while loading. |

If something isn't covered, open an issue with `docker compose logs` and the output of
`free -h` and `nvidia-smi`.

---

## Building the image yourself

```bash
docker build -t qwen38-flash-next-exl3:3bpw .
```

It compiles exllamav3 for sm_121 (~20–30 min) and installs TabbyAPI. See [docs/BUILD.md](docs/BUILD.md)
for build args, verification, and how to publish to GHCR.

---

## Notes and limits

- **Single stream is the fast path.** `max_batch_size: 1`; this is a personal/agent endpoint, not a
  high-concurrency server.
- **Draft confidence.** Upstream `chat.py` used `-dds -dc 0.6`; TabbyAPI's dynamic draft uses
  exllamav3's default `0.4` and does not expose the knob. Code is unaffected; prose loses a few tok/s.
- **Not vLLM.** vLLM does not load EXL3 checkpoints; its Qwen3.8-Flash-Next path is BF16/NVFP4. This
  image is exllamav3 end to end.
- **Licensing.** Our glue is MIT. The image bundles TabbyAPI (**AGPL-3.0**) and exllamav3 (MIT); the
  weights are `Lygodactylus` (see the model card). Confirm terms before redistributing the image.

## Credits

This repo is packaging and glue around other people's work. The credit belongs to:

| Contribution | Source |
| --- | --- |
| Base model — Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen team) |
| Uncensored EXL3 re-quant (the pack this serves) | [Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw](https://huggingface.co/Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw) |
| EXL3 quant format, kernels, upstream engine | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) (turboderp) |
| GB10 / aarch64 guards and the decode-kernel improvements | [vcruz305/exllamav3 `523ecd3`](https://github.com/vcruz305/exllamav3/commit/523ecd3) |
| The single-Spark tuning that gets ~80 tok/s (MTP `ndt=5`, dynamic draft, 8-bit KV, int8 mixers, core pinning) | [vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe) — Cruz ([@ViC305](https://x.com/ViC305)) |
| OpenAI-compatible HTTP server | [theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI) |
| Reference single-Spark Docker/recipe workflow this image follows | the GLM-5.3-Flash EXL3 DGX image (`ghcr.io/0xsero/glm53-flash-exl3-plain`) |
| Platform | NVIDIA DGX Spark (GB10), CUDA 13 |

If this is useful to you, please go star and credit the upstream projects above — the measured
performance here is their work, not ours. `ngram_ram` handling and the GB10 tuning in particular come
straight from the exllamav3 fork and its recipe.

## Repository layout

```
Dockerfile                     image build (CUDA 13 devel + torch cu130 + exllamav3 sm_121 + TabbyAPI)
config.yml                     TabbyAPI model/serving config
entrypoint.sh                  fadvise pages, optional core pinning, launch TabbyAPI
compose.yaml                   one-command run
scripts/download-model.sh      fetch the EXL3 pack
scripts/advise-model-files.py  unprivileged page-cache drop for the model
scripts/start.sh               host convenience wrapper (docker compose + tailscale serve)
examples/openai_client.py      OpenAI Python client example
docs/OPENCODE.md               opencode provider config + hand-off prompt
docs/FINDINGS.md               why ngram_ram is required; measured split; tuning
docs/BUILD.md                  building and publishing the image
```
