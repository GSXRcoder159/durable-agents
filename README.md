# CSE 585 Final Project

## Compile and Run

This project is plain Python, so there is no separate build step. The commands
below install the dependencies, check that the code compiles, and run the agent.

1. Install the dependencies:

   ```bash
   uv sync
   ```

2. Add a `.env` file in the project root:

   ```bash
   GOOGLE_API_KEY=your_key_here
   # Optional:
   # DB_PATH=db.sqlite
   ```

3. Run the agent:

   ```bash
   uv run python -m src
   ```

Useful commands:

```bash
uv run python -m src run <run_id>        # run with a fixed ID
uv run python -m src inspect <run_id>    # view logs for a run
uv run python -m src recover <run_id>    # recover a saved run
uv run python scripts/demo.py            # run the crash/recovery demo
```

## Experiments

The experiment scripts are in `scripts/` and save their output under `results/`.

```bash
uv run python scripts/exp1.py    # crash recovery vs. full rerun
uv run python scripts/exp2.py    # idempotency replay vs. duplicate tool calls
uv run python scripts/exp3.py    # recovery under injected tool failures
uv run python scripts/exp4.py    # final-output correctness after recovery
```

Each script has a few optional flags, but the defaults are enough for a normal
run.
