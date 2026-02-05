"""New workflow template for `workflows/`.

This module provides a minimal `get_config()` that returns typed defaults. Copy or rename this file for new workflows.
"""

def get_config():
    """Return default configuration values for the workflow.

    Make sure values have the correct types so CLI `--set` overrides can infer types.
    """
    return {
        "input_path": "data/input",
        "output_dir": "output",
        "verbose": False,
        "threshold": 0.5,
    }
