import os
import trimesh
from core.mesh_combination_methods import unite_glb_files


def make_simple_glb(path: str):
    s = trimesh.primitives.Sphere(radius=0.5)
    s.export(path, file_type="glb")


def test_unite_two_simple(tmp_path):
    a = tmp_path / "a.glb"
    b = tmp_path / "b.glb"
    make_simple_glb(str(a))
    make_simple_glb(str(b))

    out = str(tmp_path / "combined.glb")
    res = unite_glb_files(str(tmp_path), out, pattern="*.glb", compress=False)

    assert os.path.exists(res)

    # Load and ensure at least two geometries were combined
    scene = trimesh.load(res, force="scene")
    assert len(scene.geometry) >= 2


def test_unite_with_indices(tmp_path):
    # Create pore0..pore4.glb and select a slice by numeric indices (1..3 inclusive)
    for i in range(5):
        p = tmp_path / f"pore{i}.glb"
        make_simple_glb(str(p))

    out = str(tmp_path / "combined_idx.glb")
    res = unite_glb_files(str(tmp_path), out, pattern="pore*.glb", compress=False, start_index=1, end_index=3)

    assert os.path.exists(res)
    scene = trimesh.load(res, force="scene")
    # Expect 3 geometries: pore1, pore2, pore3
    assert len(scene.geometry) == 3


def test_unite_with_string_indices(tmp_path):
    # Ensure string inputs (from CLI) are handled
    for i in range(6):
        p = tmp_path / f"pore{i}.glb"
        make_simple_glb(str(p))

    out = str(tmp_path / "combined_idx_str.glb")
    res = unite_glb_files(str(tmp_path), out, pattern="pore*.glb", compress=False, start_index="2", end_index="4")

    assert os.path.exists(res)
    scene = trimesh.load(res, force="scene")
    assert len(scene.geometry) == 3


def test_unite_with_coloring(tmp_path):
    # Create 3 pore files and test both coloring methods
    for i in range(3):
        p = tmp_path / f"pore{i}.glb"
        make_simple_glb(str(p))

    out = str(tmp_path / "combined_colored_material.glb")
    res = unite_glb_files(str(tmp_path), out, pattern="pore*.glb", compress=False, color=True, color_method="material", alpha=1.0)

    assert os.path.exists(res)
    scene = trimesh.load(res, force="scene")
    assert len(scene.geometry) == 3

    # Ensure exported file preserves vertex colors for each mesh (fast per-mesh coloring)
    colors = []
    for mesh in scene.geometry.values():
        vc = getattr(mesh.visual, "vertex_colors", None)
        assert vc is not None
        assert vc.shape[0] >= 1
        colors.append(tuple(vc[0][:3]))
    assert len(set(colors)) > 1


def test_numeric_sorting(tmp_path):
    # Create files that would be out of order alphabetically
    files = ["pore1.glb", "pore2.glb", "pore10.glb", "pore11.glb", "pore3.glb"]
    for name in files:
        p = tmp_path / name
        make_simple_glb(str(p))

    from core.mesh_combination_methods import list_glb_files
    normal = list_glb_files(str(tmp_path), pattern="pore*.glb", numeric_sort=False)
    numeric = list_glb_files(str(tmp_path), pattern="pore*.glb", numeric_sort=True)

    # Alphabetical order puts pore10 before pore2
    assert normal[0].endswith("pore1.glb")
    assert normal[1].endswith("pore10.glb")

    # Numeric-aware order should place pore2 before pore10
    assert numeric[0].endswith("pore1.glb")
    assert numeric[1].endswith("pore2.glb")
    assert numeric[2].endswith("pore3.glb")
    assert numeric[3].endswith("pore10.glb")
    assert numeric[4].endswith("pore11.glb")
    # Now test vertex coloring still works (slower path)
    out2 = str(tmp_path / "combined_colored_vertex.glb")
    res2 = unite_glb_files(str(tmp_path), out2, pattern="pore*.glb", compress=False, color=True, color_method="vertex", alpha=1.0)
    assert os.path.exists(res2)
    scene2 = trimesh.load(res2, force="scene")
    colors2 = []
    for mesh in scene2.geometry.values():
        vc = getattr(mesh.visual, "vertex_colors", None)
        assert vc is not None
        colors2.append(tuple(vc[0][:3]))
    assert len(set(colors2)) > 1
