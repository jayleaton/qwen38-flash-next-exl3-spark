# BUILD — the container image

## What the image contains

- `nvidia/cuda:13.0.2-devel-ubuntu24.04` (arm64) — `-devel` because exllamav3 is compiled for
  **sm_121** at build time (no aarch64 release wheels exist).
- Python 3.12 venv at `/opt/venv`, `torch==2.14.0+cu130` (aarch64 wheels from the cu130 index).
- [vcruz305/exllamav3@523ecd3](https://github.com/vcruz305/exllamav3/commit/523ecd3) (v1.5.0) built
  with `TORCH_CUDA_ARCH_LIST=12.1`, plus its `requirements.txt`.
  Upstream: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3).
- [theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI) (`main`) — the OpenAI-compatible
  server.
- `config.yml`, `entrypoint.sh`, `scripts/advise-model-files.py`.

The model is **not** in the image; it is mounted at `/models/qwen38-flash-next-unc-3bpw`.
See the [Credits](../README.md#credits) section for the full attribution list.

## Build

```bash
docker build -t qwen38-flash-next-exl3:3bpw .
```

Expect 20–35 min on a Spark (the exllamav3 CUDA compile dominates). The build pins the exllamav3 and
TabbyAPI revisions via `--build-arg`:

```bash
docker build \
  --build-arg EXLLAMAV3_COMMIT=523ecd3 \
  --build-arg TABBY_COMMIT=main \
  -t qwen38-flash-next-exl3:3bpw .
```

### Verify the build

```bash
docker run --rm --gpus all qwen38-flash-next-exl3:3bpw \
  python -c "import torch, exllamav3_ext; print(torch.__version__, 'ext OK')"
```

## Publish (not done yet)

When you are ready to ship it:

```bash
docker tag qwen38-flash-next-exl3:3bpw ghcr.io/OWNER/qwen38-flash-next-exl3:3bpw
docker push ghcr.io/OWNER/qwen38-flash-next-exl3:3bpw
```

Then set `image:` in `compose.yaml` to that reference, and users only need
`docker compose up -d`.

Licensing note before publishing: the image redistributes **TabbyAPI (AGPL-3.0)**. Decide the repo
license accordingly (AGPL-3.0 is the safe choice for the combined image) and keep the upstream
LICENSE files in `third_party/` for attribution.

## Multi-arch / CI notes

- Only `linux/arm64` matters here (GB10). Do not attempt an `amd64` build — the sm_121 target and
  the GB10 kernel tuning are the point.
- A GitHub Actions arm64 runner (or a self-hosted Spark) can build and push on tags. The exllamav3
  compile makes each build ~30 min, so cache the pip/torch layers aggressively.
