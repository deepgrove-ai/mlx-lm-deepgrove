# Maple on mlx-lm

Maple is a 20B-A1B ternary MoE with 24 layers, 256 experts, top-8, 512-token sliding
window on 3 of every 4 layers. Weights are 2-bit packed `{-α, 0, +α}`, one α per
row. 

## Setup

On an Apple Silicon Mac running macOS 26.2 or later, install
[uv](https://docs.astral.sh/uv/) and run:

```sh
git clone git@github.com:deepgrove-ai/mlx-lm-deepgrove.git
cd mlx-lm-deepgrove
./setup.sh
source .venv/bin/activate
hf download deepgrove/maple-preview-2bit-mlx --local-dir maple-2bit-mlx
```

`./setup.sh` installs everything needed. No Xcode or manual build is required.

## Run

```sh
python -m mlx_lm generate --model ./maple-2bit-mlx --trust-remote-code --flash-head \
  --prompt "Write a haiku about a grove." --temp 1.0 --top-p 0.95 --top-k 20

python -m mlx_lm chat --model ./maple-2bit-mlx --trust-remote-code --max-tokens -1 \
  --temp 1.0 --top-p 0.95
```

Enable flash head for extra speed.
```sh
python -m mlx_lm chat --model ./maple-2bit-mlx --trust-remote-code --max-tokens -1 \
  --temp 1.0 --top-p 0.95 --flash-head
```

## HTTP API

`mlx_lm.server` exposes an OpenAI-compatible HTTP endpoint. The easiest way to
serve Maple-Preview is the bundled launcher, which registers both output heads
as selectable models:

```sh
./scripts/serve_maple.sh            # default: 127.0.0.1:8080
```

This serves the local `./maple-2bit-mlx` checkpoint under two aliases
(override with `MAPLE_MODEL=deepgrove/maple-preview-2bit-mlx` to auto-download):

| model id | head |
| --- | --- |
| `maple-preview` | exact lm_head (default) |
| `maple-preview-flash` | approximate FlashHead lm_head (~20% faster decode) |

Aliases are defined in `maple_models.json` (see `--model-registry` in
`python -m mlx_lm.server --help`); each entry may pin `flash_head` or any
other `model_config` override, so multiple variants can be served from one
process. The checkpoint auto-downloads from Hugging Face on first load.

```sh
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "maple-preview-flash",
       "messages": [{"role": "user", "content": "Say hello."}]}'

curl localhost:8080/v1/models   # lists both aliases
```

Same API with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
resp = client.chat.completions.create(
    model="maple-preview-flash",
    messages=[{"role": "user", "content": "What is 12 * 12?"}],
    stream=True,
)
for chunk in resp:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Notes:

- Maple is a reasoning model: `<think>`…`</think>` output is stripped from
  `content` and returned in the `reasoning` field of each choice. Disable
  thinking with `--chat-template-args '{"enable_thinking": false}'`.
- Requires `--trust-remote-code` (the checkpoint ships its own `maple.py`).
- Not for production: the server implements only basic security checks.
- See `mlx_lm/SERVER.md` for the full request/response reference.

### Tool calling

This branch hardens tool-call handling against Maple's flaky tool-call
generation. Maple emits calls inside `<tool_call>…</tool_call>` blocks, and it
frequently produces output that breaks naive parsing — blocks that never
close, back-to-back JSON objects, repeated calls, or calls with garbled
arguments keys. The server now:

- recovers multiple back-to-back JSON objects inside a single block and
  forward-scans past malformed spans (garbled objects, stray markers,
  trailing prose); only objects with a string `"name"` key are emitted as
  tool calls, so argument sub-objects are never confused for calls;
- aborts generation when a block re-opens `<tool_call>` instead of closing
  it (repetition loop) and keeps only the first complete call;
- aborts blocks that grow past `--max-tool-call-chars` without a closing tag
  (default 2048; `scripts/serve_maple.sh` uses 8192);
- collapses consecutive calls to the same function with equivalent arguments
  (exact, strict-superset, or renamed-key duplicates) into a single call,
  preferring the more complete or corrected version.

Each `tool_calls` entry carries `id`, `type: "function"`, and a
`function` object with `name` and `arguments` (JSON string), matching the
OpenAI chat completions schema. Smoke-tested end-to-end in
`tests/test_server.py` and `tests/test_tool_parsing.py`.

| chip | head | decode tok/s | prefill tok/s | peak |
| --- | --- | --- | --- | --- |
| M4 | exact (default) | 169 | 1075 | 6.51 GB |
| M4 | `--flash-head` | **218** | 1075 | 6.69 GB |
| M5 Pro | exact (default) | 387 | 3511 | 6.96 GB |
| M5 Pro | `--flash-head` | **477** | 3513 | 6.99 GB |

M5 Pro: standard setup, 20-core GPU, 48 GB, macOS 26.4, MLX 0.32.0;
battery power, default scheduling.
512-token prompt / 256 generated, temp 1.0, top-p 0.95, 512 FlashHead probes;
median of 6 runs. Peak measured separately per head, including model loading.
M4 results are from the previous version.

## Convert

```sh
python -m mlx_lm.ternary /path/to/maple-bf16 -o maple-2bit-mlx --flash-head
```

Streams and converts shard by shard, so the 38 GB bf16 source is never fully resident.

- `--flash-head` — ~2 min of k-means, score 4748
  vocabulary-cluster centroids, then compute exact logits only for the top 512
  clusters (special tokens always scored). Greedy is exact whenever the true
  argmax is in a probed cluster. Attach to an already-converted
  directory with `python -m mlx_lm.ternary maple-2bit-mlx --flash-head-only`
  (rewrites in place; point it at a real directory, not hardlinks).
- `--group-scales` — repeat each row's α across every group (+0.6 GB), only for
  tools that read MLX quantized checkpoints generically. Default stores the row
  scale once as `row_alpha`; `sanitize()` expands it at load.

## Diff vs upstream mlx-lm

| file | what |
| --- | --- |
| `mlx_lm/models/maple.py` | the model (also copied into every converted checkpoint) |
| `mlx_lm/ternary.py` | bf16 → ternary converter + FlashHead generator |
| `mlx_lm/server.py` | OpenAI-compatible HTTP server + `--model-registry` per-model config |
| `maple_models.json` | registry mapping `maple-preview` / `maple-preview-flash` to the checkpoint |
| `scripts/serve_maple.sh` | one-command HTTP server launcher for Maple-Preview |
| `tests/test_maple_kernels.py` | kernel + precision self-check — `pytest tests/test_maple_kernels.py -v` |
| `tests/test_maple_service.py` | registry + alias tests — `pytest tests/test_maple_service.py -v` |
| `generate.py`, `chat.py`, `server.py`, `benchmark.py` | support for `--flash-head` flag |
| `setup.sh` | uv venv + editable install |
