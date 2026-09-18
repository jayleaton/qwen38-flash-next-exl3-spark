# FINDINGS — tuning Qwen3.8-Flash-Next uncensored EXL3 on one Spark

All numbers: single stream, greedy (`top_k=1`), 400 new tokens, after dropping the model's page
cache, on one DGX Spark (GB10, 121 GiB unified, ~273 GB/s).

## Headline

`-mtp -ndt 5 -dds -dc 0.6 -cq 8,8 -ngr` → **79.89 tok/s** (`-cs 32768`) / **79.63 tok/s**
(`-cs 262144`), draft acceptance **71.6%**. No-draft baseline: **31.5 tok/s**.

## 1. `ngram_ram` (the `-ngr` flag) is not optional for this pack

First run with the recipe's flags *except* `-ngr` gave **21.55 tok/s** — with a healthy 71.6% draft
acceptance, so it was not an acceptance problem.

`split_time.py` (NDT=5) split one speculative round:

| Component | Time | Expected |
| --- | ---: | ---: |
| target verify forward (q=6) | 44.87 ms | ~52 ms |
| draft forwards (4.9/round) | 0.59 ms each | fast |
| **host / gather / sync** | **35.75 ms/round** | milliseconds |

The GPU work was correct; the **host-side** cost was the whole gap. Cause: Qwen4Exp's 51B n-gram
(PLE) embedding table. exllamav3's default **streams n-gram rows from disk per forward**. With the
table pinned in RAM (`ngram_ram: true`, chat.py `-ngr`) that cost disappears.

Why the upstream recipe did not need it: turboderp's pack ships the table **sharded** (32.6 GB across
shards) and streams acceptably. `Lygodactylus`'s uncensored pack ships it **unsharded** (one 19.8 GB
`ngram_embedding.safetensors`), which does not.

**Rule for any uncensored Qwen3.8-Flash-Next EXL3 re-quant: check the n-gram layout and set
`ngram_ram: true`.**

## 2. Autosplit reads `MemFree`, not `MemAvailable`

The first load after downloading the pack failed with
`RuntimeError: Insufficient VRAM in split for model and cache` while 116 GiB was "available".
exllamav3's autosplit consults `cudaMemGetInfo`, which on GB10 tracks `MemFree`; page cache from the
72 GB download counted against it. `scripts/advise-model-files.py` (posix_fadvise `DONTNEED`) fixes
the model's own pages from inside the container; a host-wide `drop_caches` clears anything else.

## 3. What each flag is worth (from the upstream recipe, reproduced here)

| Lever | Effect |
| --- | --- |
| MTP drafter (`draft_mode: mtp`) | ~33 → ~50–56 tok/s depending on acceptance |
| `-ndt 5` (draft depth) | +8 over `ndt=3` on code; verify costs 37 ms at q=2, 52 at q=5, 86 at q=9 |
| `dynamic_draft` (`-dds`) | ±0 code, **+8 prose** |
| `-cq 8,8` (8-bit KV) | +3 at 4k, +7 at 240k; KV is 12 KB/token, 3 GB for the full window |
| `EXL3_GR_INT8=1` | +5 code, +7 DevOps, +4 prose (hyperconnection mixers int8 instead of fp16) |
| `EXL3_MOE_COOP_WIDE=1` | +5 (wide 128-col MoE decode tile for GB10's 48 SMs) |
| `EXL3_INT8_GEMV=0` | +3 (the int8-activation GEMV is slower than fp16 on GB10) |
| `taskset -c 5-9,15-19` | +2 (the 10 Cortex-X925 cores; the launcher thread otherwise lands on an A725) |
| **`ngram_ram` / `-ngr`** | **+58** for this pack (see §1) |

## 4. TabbyAPI vs `chat.py`

TabbyAPI exposes `ngram_ram`, `draft_mode: mtp`, `dynamic_draft`, `draft_num_tokens`, `cache_mode`,
`cache_size`, `tool_format` and `reasoning` — everything above except `draft_confidence`. Its dynamic
draft uses exllamav3's default `0.4` rather than `0.6`, which is a wash on code and costs a few
tok/s on prose. Passing `0.6` requires a custom server (the `Generator(dynamic_draft_tokens=True,
draft_confidence=0.6)` constructor argument).

## 5. Reproduce on the host (no Docker)

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches      # no model running
~/bench-qwen38.sh ctx262k -mtp -ndt 5 -dds -dc 0.6 -cq 8,8
grep -aE "^Context:|Draft" ~/bench_qwen38_ctx262k.log
# Context: 76 new tokens ... Generate: 400 tokens at 79.63 t/s - Draft: 295 / 412 accepted (71.6%)
```
