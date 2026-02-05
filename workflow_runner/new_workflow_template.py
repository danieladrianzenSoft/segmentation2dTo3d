"""Runner template for `workflow_runner/`.

Copy this file and adapt `run(config)` to orchestrate core functions and perform I/O.
"""
from pathlib import Path
from core.example_core import load_and_process, save_result


def run(config):
    """Run the workflow using the provided config dict.

    This runner demonstrates the expected pattern:
    - parse config
    - call pure core functions
    - perform I/O (save outputs)
    """
    print(f"🚀 Running new_workflow_template with config: {config}")

    input_path = config.get("input_path")
    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load/process using core functions
    data = load_and_process(input_path, threshold=config.get("threshold", 0.5))

    # Save results (I/O kept in runner)
    out_file = output_dir / "new_workflow_output.npz"
    save_result(out_file, data)

    print(f"✅ Completed. Saved output to {out_file}")
    return out_file
