#!/bin/sh
# Create a uv venv and install mlx-lm with Maple support.
set -e
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--no-native" ]; }; then
    echo "Usage: $0 [--no-native]" >&2
    exit 2
fi
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv not found — install it: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ -d .venv ] || uv venv --python 3.12
uv pip install --python .venv/bin/python --only-binary :all: -e . rich "mlx==0.32.0"
if [ "${1:-}" != "--no-native" ]; then
    python_tag=$(.venv/bin/python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    case "$python_tag" in
        cp312|cp313) ;;
        *) echo "Prebuilt kernels need Python 3.12 or 3.13 (or use --no-native)." >&2; exit 1 ;;
    esac
    kernels_url=https://huggingface.co/deepgrove/maple-preview-2bit-mlx/resolve/ad3e2a0a772a2131271483ba6770a7cbb9846b6f/wheels
    uv pip install --python .venv/bin/python --only-binary :all: \
        "$kernels_url/mlx_lm_maple_kernels-0.1.0-$python_tag-$python_tag-macosx_26_0_arm64.whl"
fi
echo "Done. Activate with: source .venv/bin/activate"
