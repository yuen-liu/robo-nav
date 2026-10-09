set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_LINK_MODE=copy
cd ~/robonav/LightNav-0
uv venv .venv --python /usr/local/bin/python3.11
uv pip install --python .venv/bin/python -e ".[vllm,video]"
[ -d checkpoints/LightNav-0 ] || .venv/bin/hf download LightOriginsHQ/LightNav-0 --local-dir checkpoints/LightNav-0
.venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
