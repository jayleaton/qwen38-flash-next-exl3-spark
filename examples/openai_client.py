#!/usr/bin/env python3
"""Minimal OpenAI-compatible client for the Qwen3.8-Flash-Next EXL3 server.

    pip install openai
    export OPENAI_API_KEY=...                       # the key from api_tokens.yml
    export OPENAI_BASE_URL=http://127.0.0.1:5000/v1 # or your Tailscale URL
    python examples/openai_client.py "explain unified memory in one paragraph"
"""
import os
import sys

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:5000/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
)

prompt = " ".join(sys.argv[1:]) or "Write a haiku about unified memory."

resp = client.chat.completions.create(
    model=os.environ.get("MODEL", "qwen3.8-flash-next"),
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1024,
    temperature=0,
    # Thinking is on by default; disable it per request when you want raw content.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)

# Streaming:
#   for chunk in client.chat.completions.create(..., stream=True):
#       print(chunk.choices[0].delta.content or "", end="")
