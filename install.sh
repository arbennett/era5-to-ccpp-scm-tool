curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
source .venv/bin/activate
python -m ipykernel install --user --name=scm