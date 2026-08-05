# Maple on mlx-lm

Maple is a 20B-A1B ternary MoE with 24 layers, 256 experts, top-8, 512-token sliding
window on 3 of every 4 layers. Weights are 2-bit packed `{-α, 0, +α}`, one α per
row. 

This fork runs on the stock MLX build for portability. We intend to release a faster custom
library in the coming days.

## Setup

Requires Apple Silicon and [uv](https://docs.astral.sh/uv/).

```sh
git clone git@github.com:deepgrove-ai/mlx-lm-deepgrove.git
cd mlx-lm-deepgrove
./setup.sh
source .venv/bin/activate
hf download deepgrove/maple-preview-2bit-mlx --local-dir maple-2bit-mlx
```

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

| chip | head | decode tok/s | prefill tok/s | peak |
| --- | --- | --- | --- | --- |
| M4 | exact (default) | 169 | 1075 | 6.51 GB |
| M4 | `--flash-head` | **218** | 1075 | 6.69 GB |
| M5 Pro | exact (default) | 359 | 3773 | 6.73 GB |
| M5 Pro | `--flash-head` | **395** | 3857 | 6.92 GB |

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
