# Using it from opencode

The endpoint is OpenAI-compatible, so opencode talks to it through the
`@ai-sdk/openai-compatible` provider. Point it at the Tailscale URL of the Spark.

Prerequisite on the **Spark**: expose the loopback port over Tailscale.

```bash
# on the DGX Spark
sudo tailscale serve --bg --https=5000 http://127.0.0.1:5000
tailscale serve status          # -> https://<spark>.<tailnet>.ts.net:5000 -> 127.0.0.1:5000
```

Then, on the client machine, `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "dgx-qwen": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DGX Spark — Qwen3.8-Flash-Next (uncensored EXL3)",
      "options": {
        "baseURL": "https://<spark>.<tailnet>.ts.net:5000/v1",
        "apiKey": "{env:DGX_QWEN_API_KEY}"
      },
      "models": {
        "qwen3.8-flash-next": {
          "name": "Qwen3.8-Flash-Next uncensored EXL3 (3.05bpw)",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 262144, "output": 32768 }
        }
      }
    }
  }
}
```

The model is then `dgx-qwen/qwen3.8-flash-next`. To make it the default, add:

```json
{ "model": "dgx-qwen/qwen3.8-flash-next" }
```

Config is read once at startup — restart opencode after editing.

## Prompt you can hand to an agent

Copy the block below into an agent on the client machine, filling in the two `<…>` values
(the Spark's Tailscale hostname and the API key from `api_tokens.yml` on the Spark).

> **Task: add the DGX Spark Qwen model to opencode on this machine.**
>
> Configure opencode to use a local LLM served by an OpenAI-compatible endpoint on another machine
> on my Tailscale network.
>
> Facts:
> - Base URL: `https://<SPARK>.<TAILNET>.ts.net:5000/v1`
> - API key (bearer): `<API_KEY>`
> - Model id for the `model` field: `qwen3.8-flash-next`
> - The server is TabbyAPI on a DGX Spark: an uncensored EXL3 quant of Qwen3.8-Flash-Next, 6B
>   active params, 262,144-token context.
>
> Steps:
> 1. Verify reachability from this machine:
>    `curl -s https://<SPARK>.<TAILNET>.ts.net:5000/v1/models -H "Authorization: Bearer <API_KEY>"`
>    Expect JSON containing `"id": "qwen38-flash-next-unc-3bpw"` and `"n_ctx": 262144`. If it fails,
>    confirm Tailscale is up here and that the Spark reports `tailscale serve status` with port 5000
>    mapped; report the exact error and stop.
> 2. Edit (or create) `~/.config/opencode/opencode.json`. Preserve any existing keys and merge in a
>    provider named `dgx-qwen` using `npm: "@ai-sdk/openai-compatible"`, the baseURL above, and
>    `"apiKey": "{env:DGX_QWEN_API_KEY}"`. Declare a model `qwen3.8-flash-next` with
>    `"tool_call": true`, `"reasoning": true`, and
>    `"limit": { "context": 262144, "output": 32768 }`. Keep `"$schema": "https://opencode.ai/config.json"`.
>    Do not hardcode the key in the file.
> 3. Persist the key: append `export DGX_QWEN_API_KEY="<API_KEY>"` to the shell profile
>    (`~/.zshrc` or `~/.bashrc`) and export it in the current session.
> 4. Restart opencode (it does not hot-reload config) and smoke-test:
>    `opencode run --model dgx-qwen/qwen3.8-flash-next "Say hello in one short sentence."`
> 5. Report the config path, the final provider block, the curl output, the smoke-test output, and
>    anything that failed.
>
> Caveats (do not "fix" these):
> - Thinking is on by default; the model emits a `reasoning_content` block, so give it ≥1024 output
>   tokens or `content` can be empty.
> - Keep prompts under ~160k tokens to retain the MTP speed-up; decoding slows past ~164k.
> - The server is single-stream (`max_batch_size: 1`); avoid many parallel opencode sessions.
> - Tool calls use the Qwen format; if they fail, report the raw response rather than swapping models.
