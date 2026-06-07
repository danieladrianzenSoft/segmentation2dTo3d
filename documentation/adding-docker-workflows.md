# Adding Workflows for Docker Access

This guide explains how to expose a workflow so it can be called from external repos via the Docker image (`ghcr.io/seguralab/segmentation-workflows`).

## Prerequisites

The workflow must already exist as:
- `workflows/<name>.py` — with a `get_config()` function returning default config
- `workflow_runner/<name>.py` — with a `run(config)` function

## Steps

### 1. Register the workflow as public

In `workflow_registry.py`, add the workflow name to `PUBLIC_WORKFLOWS`:

```python
PUBLIC_WORKFLOWS = [
    "unite_meshes",
    "mesh_generation",
    "your_new_workflow",  # <-- add here
]
```

This is the allowlist enforced when the Docker entrypoint runs with `--public`. Workflows not in this list cannot be invoked from the container.

### 2. Create a default config file

Add a JSON file at `configs/<name>.json` with Docker-appropriate defaults. Paths should reference `/data/` since that's the standard volume mount point inside the container.

Example (`configs/mesh_generation.json`):

```json
{
  "input_dir": "/data/input",
  "output_dir": "/data/output",
  "file_type": "json",
  "batch_process": true,
  "show_edge_pores": true,
  "save_metadata": true,
  "save_mesh": true,
  "scrape_subdirectories": false
}
```

Guidelines for configs:
- Use `/data/input` and `/data/output` (or similar `/data/...` paths) as defaults — callers mount their volumes to `/data`
- Only include parameters that differ from or are essential for Docker usage
- Omit development-only parameters (e.g., `file_index` which is for interactive selection)
- Use `null` for optional parameters you want to leave unset

### 3. Rebuild and push the Docker image

After merging to `main`, the GitHub Actions workflow (`.github/workflows/docker-image.yml`) automatically builds and pushes a new image tagged `latest`. You can also tag a release (`v*`) for versioned images.

To test locally before pushing:

```bash
docker build --build-arg INSTALL_GLTF=true -t seg-workflows:test .
docker run --rm -v $(pwd)/data:/data:rw seg-workflows:test \
  --workflow your_new_workflow \
  --config /app/configs/your_new_workflow.json
```

## Usage from external repos

Once the image is published, callers invoke the workflow like this:

```bash
docker run --rm \
  -v /path/to/local/data:/data:rw \
  ghcr.io/seguralab/segmentation-workflows:latest \
  --workflow mesh_generation \
  --config /app/configs/mesh_generation.json
```

Override specific parameters with `--set`:

```bash
docker run --rm \
  -v /path/to/local/data:/data:rw \
  ghcr.io/seguralab/segmentation-workflows:latest \
  --workflow mesh_generation \
  --config /app/configs/mesh_generation.json \
  --set input_dir=/data/my_jsons output_dir=/data/my_meshes show_edge_pores=false
```

Or provide a custom config file mounted into the container:

```bash
docker run --rm \
  -v /path/to/local/data:/data:rw \
  -v /path/to/custom_config.json:/app/configs/mesh_generation.json:ro \
  ghcr.io/seguralab/segmentation-workflows:latest \
  --workflow mesh_generation \
  --config /app/configs/mesh_generation.json
```

## Summary checklist

| Step | File | Action |
|------|------|--------|
| 1 | `workflow_registry.py` | Add workflow name to `PUBLIC_WORKFLOWS` |
| 2 | `configs/<name>.json` | Create default Docker config with `/data/` paths |
| 3 | Push to `main` | CI builds and publishes the updated image |

## Notes

- The `--public` flag is baked into the Docker entrypoint (`Dockerfile`), so you never need to pass it manually when running the container
- The container runs as non-root user `appuser` — ensure output directories within `/data` are writable
- For workflows that need gltf-pipeline (Draco compression), the build arg `INSTALL_GLTF=true` is already set in CI
