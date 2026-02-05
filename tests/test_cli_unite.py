import os
import subprocess
import sys
import trimesh

from pathlib import Path


def make_simple_glb(path: str):
    s = trimesh.primitives.Sphere(radius=0.5)
    s.export(path, file_type="glb")


def test_cli_unite(tmp_path):
    # Create a small set of pore files
    for i in range(4):
        p = tmp_path / f"pore{i}.glb"
        make_simple_glb(str(p))

    out = tmp_path / "combined_cli.glb"
    cmd = [sys.executable, "cli/cli.py", "unite", "--input-dir", str(tmp_path), "--output", str(out), "--start-index", "0", "--end-index", "3", "--color"]

    subprocess.check_call(cmd)

    assert out.exists()
    scene = trimesh.load(str(out), force="scene")
    assert len(scene.geometry) >= 4
