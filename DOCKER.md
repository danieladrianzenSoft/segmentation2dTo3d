# Docker image for segmentation-workflows CLI

This repository provides a small CLI image that exposes the `unite` command to
combine multiple `.glb` files into a single mesh file. The CLI is designed to
be used as an ephemeral container (run, produce artifact(s), exit) from your
.NET backend or other orchestration.

Quick build

```bash
# Build locally
docker build -t segmentation-workflows:latest .

# Run the workflow runner (example)
docker run --rm -v $(pwd)/data:/data segmentation-workflows:latest \
  --workflow unite_meshes \
  --config /app/configs/unite_meshes.json \
  --set start_index=0 end_index=100
```

Notes
- The image installs `gltf-pipeline` globally (optional) to support Draco
  compression (used if you set `compress=true` in config or pass `--set compress=true`).
  If you don't need compression, you can remove the `npm install -g gltf-pipeline`
  line in the `Dockerfile` to reduce image size.
- Prefer mapping host directories or using shared volumes/S3 for large files
  rather than streaming large payloads through HTTP.

Example: Using the image from .NET
- Use Docker SDK for .NET (or shelling out) to run the container with proper
  volume mounts. Capture logs and the exit code and read the output artifact
  from the shared volume.

Advanced
- You can publish this image to a registry and pull it from any host.
- For long-run or heavy usage, orchestrate via k8s Jobs (recommended).

Publish to GitHub Container Registry (GHCR) 
- Example GitHub Actions workflow to build & publish is included in `.github/workflows/docker-image.yml`.
- Alternatively, build and push locally:

```bash
# Build (optionally disable the gltf tool to reduce size)
docker build --build-arg INSTALL_GLTF=true -t ghcr.io/seguralab/segmentation-workflows:latest .

# Login and push to GHCR (replace <OWNER> with your user/org)
echo "${GHCR_PAT}" | docker login ghcr.io -u seguralab --password-stdin
docker push ghcr.io/seguralab/segmentation-workflows:latest
```

Notes:
- If you prefer Docker Hub, change the `docker/login-action` and tags in the CI workflow to `docker.io/seguralab/segmentation-workflows:latest`.
- The included GitHub Actions workflow will publish on pushes to `main` and when tags like `v*` are pushed. Ensure the repository permits GitHub Actions to publish packages in repository settings.
