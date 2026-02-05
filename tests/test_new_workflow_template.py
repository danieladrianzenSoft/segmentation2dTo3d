"""Unit tests for the workflow/runner/core templates."""
import tempfile
import numpy as np
from workflow_runner.new_workflow_template import run


def test_new_workflow_template(tmp_path):
    cfg = {
        "input_path": "dummy",
        "output_dir": str(tmp_path),
        "threshold": 0.5,
    }
    out = run(cfg)
    # load the output and assert expected contents
    data = np.load(str(out))
    assert "result" in data.files
    assert data["result"].sum() == 2
