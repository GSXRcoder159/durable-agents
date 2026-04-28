# CSE 585 Final Project

## Quick Start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -e ".[dev]"
```

3. Configure environment variables (`.env` in repo root):

```bash
GOOGLE_API_KEY=your_key_here
# Optional:
# DB_PATH=db.sqlite
```

## Run

Run a normal agent execution:

```bash
python -m src
```

Run with an explicit run id:

```bash
python -m src run <run_id>
```

Inspect a run:

```bash
python -m src inspect <run_id>
```

Recover a run:

```bash
python -m src recover <run_id>
```

Run the crash/recovery demo:

```bash
python scripts/demo.py
```
