# AI Agent & Contributor Guide

## Purpose
This document helps human contributors and AI agents understand the repository structure and conventions so they can:
- Add new workflows that plug into the existing `run.py` machinery ✅
- Reuse and extend the `core/` library with new, well-documented methods ✅
- Write consistent, testable code that integrates with the overall architecture ✅

---

## Architecture at a glance 🔧
- **`workflows/`**: lightweight modules that define **`get_config()`** and default parameters for a particular task. These are the user-facing configs.
- **`workflow_runner/`**: the business logic. Each runner exposes a **`run(config)`** function which executes the workflow using functions in `core/`.
- **`core/`**: reusable, domain-specific functions (image loading, segmentation, visualization, I/O, etc.). Keep these small and well-documented.
- **`run.py`**: entry point. It loads `workflows.<name>.get_config()` and then calls the appropriate runner in `workflow_runner/` (maps with `shared_runner_map` if needed).

> Tip: follow the **separation of concerns**: `workflows/` = config + CLI defaults; `workflow_runner/` = orchestration; `core/` = pure business logic.

---

## Key conventions ⚙️
- Workflow modules must implement:
  - `def get_config() -> dict:` returning typed default values (strings, ints/floats, booleans, lists).
  - Prefer descriptive keys (e.g., `metadata_path`, `slices_dir`, `output_dir`, `process_scaffold`).
- Runner modules must implement:
  - `def run(config: dict) -> Any:` which performs the action and returns meaningful result(s) (or `None`).
- Config overrides from CLI (`run.py --set key=value`) try to preserve types when a key exists in defaults — ensure default values have the right type.
- Use small, deterministic core functions that accept/return numpy arrays and simple Python types. Add docstrings and type hints.
- Prefer `print()` with emojis for quick CLI feedback (current codebase style), but consider `logging` for complex runners.

---

## How to add a new workflow (step-by-step) 🧭
1. Create `workflows/<your_workflow>.py` with `get_config()` returning default values.
2. Decide whether an existing runner can be reused or a new runner is needed.
   - If reusing, ensure the workflow config matches expected keys in the runner.
   - If adding a new runner, create `workflow_runner/<your_workflow>.py` with `run(config)`.
3. Add any new core logic to `core/` (see next section) — keep functions small and testable.
4. Add unit tests under `tests/` (examples provided below). Use small sample data for speed.
5. Update `run.py` `shared_runner_map` if multiple workflow names should map to the same runner.
6. Manual test: `python run.py --workflow <your_workflow> --set key=value` and iterate.

---

## How to add or modify `core/` methods 🛠️
- Document inputs/outputs with docstrings and type hints (numpy arrays, shapes, units, etc.).
- Keep functions pure if possible (avoid global state, I/O inside core functions — put I/O in runner or small helper).
- Add tests for edge cases and typical inputs.
- If the function is widely useful, export it via `core/__init__.py` so other modules can import from `core` directly.

> Rule of thumb: A new core function should be covered by at least one unit test and a short usage example in its docstring.

---

## Coding style & best practices ✅
- Add a concise docstring with **purpose**, **args**, **returns**, and **side effects**.
- Use type hints for public functions.
- Keep dependencies minimal in `core/` (numpy, scipy, skimage are fine given this project).
- When changing an interface, update all callers and add tests that reflect the change.

---

## Example templates (copy & adapt) 🧩

Workflow template (in `workflows/your_workflow.py`):

```python
def get_config():
    return {
        "input_path": "data/some_input",
        "output_dir": "output",
        "do_extra_cleanup": True,
        "threshold": 0.5,
    }
```

Runner template (in `workflow_runner/your_workflow.py`):

```python
from core.some_core_module import load_data, process_data, save_results

def run(config):
    print(f"🚀 Running workflow with config: {config}")
    data = load_data(config["input_path"])
    result = process_data(data, threshold=config["threshold"])
    save_results(result, config["output_dir"])  # keep I/O here
    return result
```

Core function template (in `core/some_core_module.py`):

```python
import numpy as np

def process_data(volume: np.ndarray, threshold: float) -> np.ndarray:
    """Apply threshold and return binary mask.

    Args:
        volume: 3D numpy array (values in [0,1] or arbitrary scale)
        threshold: float threshold

    Returns:
        3D binary numpy array
    """
    return (volume > threshold).astype(np.uint8)
```

---

## Testing & validation ✅
- Add tests in `tests/test_<module>.py` that import the function and run a few small cases.
- Use deterministic small inputs (tiny arrays) so CI runs quickly.
- Example test snippet:

```python
import numpy as np
from core.some_core_module import process_data

def test_process_data():
    x = np.array([0.0, 0.6, 0.2]).reshape((1,1,3))
    out = process_data(x, threshold=0.5)
    assert out.sum() == 1
```

---

## Checklist for PRs / contributions 📋
- [ ] Add or update tests that cover new behavior.
- [ ] Add docstrings and update `AI_AGENT_GUIDE.md` if you add a new pattern.
- [ ] Use `get_config()` defaults to declare types (so CLI overrides work correctly).
- [ ] Ensure the runner prints useful progress and saves artifacts in `output_dir` or `data/`.

---

## Troubleshooting & tips 💡
- List workflows with: `python run.py --list`.
- Run a workflow: `python run.py --workflow <name> --set key=value`
- Use `--set` to quickly override defaults (types inferred from defaults when possible).
- Prefer adding small helper functions to `core/` instead of large monolithic scripts.

---

## Next steps I can help with 🔧
- Add a workflow template file to `workflows/` and a runner template to `workflow_runner/`.
- Create a `tests/` folder with CI-friendly unit tests and a minimal GitHub Actions workflow.
- Expand `AI_AGENT_GUIDE.md` with example datasets and a step-by-step tutorial.

---
