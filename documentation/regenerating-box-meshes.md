# Regenerating Box-Stored Meshes and Replacing Prod Meshes

End-to-end runbook covering:

1. Regenerating `.glb` mesh files from source voxel `.json` files (locally).
2. Pushing the canonical archive back to Box `DomainMeshes/{Particles,Subunits}/`.
3. Building a flat `ProdReplace/` folder whose filenames exactly match what
   `lovamap-01` (the prod server) expects.
4. Rsyncing that flat folder into prod.

Use this procedure any time the mesh generation code changes in a way that
invalidates previously generated outputs — for example, the chirality fix that
removed the Y/Z swap in `core/mesh_generation_methods.py`.

The workflow is the same for particles and subunits (pores). Do the mesh
regeneration one category at a time; verify particle outputs before kicking off
subunits, since both share the same mesh generation code path.

Companion diagnostic: [`prod-vs-local-inventory.md`](./prod-vs-local-inventory.md) +
[`prod-vs-local-inventory.csv`](./prod-vs-local-inventory.csv) capture the
prod ↔ box ↔ local coverage snapshot at the time of the last regeneration.
Regenerate them (see Part 3, step 3b) whenever the inputs change.

---

## The four stores

| store | path | role |
|---|---|---|
| **Box source** | `Box/.../MIMC/Data/Domains/{Particles,Subunits}/` | Voxel `.json` input files. Read-only source of truth. |
| **Local scratch** | `~/local-scratch/{Particles,Subunits,ParticleMeshes,SubunitMeshes,ProdReplace}/` | Working area. Fast SSD. Both inputs and outputs live here during regen. |
| **Box canonical archive** | `Box/.../MIMC/Data/DomainMeshes/{Particles,Subunits}/` | Structured mirror of local outputs, subdirectory-preserving. |
| **Box prod-staging** | `Box/.../MIMC/Data/DomainMeshes/ProdReplace/{Particles,Pores}/` | Flat folder, prod-matching filenames. Feeder for the prod rsync. |
| **Prod** | `lovamap-01:/srv/lovamap-shared-data/lovamap-gw/production/Domains/` | Flat UUID-named `.glb` files, served by the LOVAMAP gateway. |

---

## Why local-scratch instead of running directly against Box

The Box CloudStorage mount is a lazy-sync, network-backed filesystem. Running
a multi-hour batch against it has real failure modes that local SSD does not:

- Reads stall on cache miss (each file fetched on first access).
- Writes can land on the mount but still be syncing to Box's servers in the
  background; a half-finished sync at run time can produce truncated `.glb`
  files on disk that look complete until a viewer chokes on them.
- The mesh `.glb` and its sidecar `_metadata.json` are written in separate
  passes — on Box they can land at different times, leaving readers with
  inconsistent state mid-run.
- Sync-conflict files (`.DS_Store`, `*.conflict.glb`, etc.) can appear.
- A network drop hangs the run instead of erroring cleanly.

Working from local-scratch makes the run atomic, fast, and resumable. The
transfer back to Box is a separate, sequential step that's easy to verify and
recover from.

---

## Prerequisites

- Box CloudStorage mounted at `/Users/mimc/Library/CloudStorage/Box-Box/`.
- Local scratch dir at `/Users/mimc/local-scratch/`. Needs ~15 GB free — a few
  GB for source JSONs, several more for generated GLBs, and 4-5 GB for the
  `ProdReplace/` copy on top.
- Python env such that `python -m workflow_runner.mesh_generation` runs.
- The prod endpoint that returns `category, filename` for every mesh currently
  in prod. Output goes into `mesh-original-filenames.txt` at the repo root.
  Format:
  ```
  Particles, beadInfo_spheres_{60,100}_0_{100,0}_2.glb
  Particles, labeledDomain_spheres_s01_{soft-0.25,hard-0.75}.glb
  Pores, Yining_S_3_segment_100@20260319T044231.glb
  ...
  ```
  Convention: particles have no `@YYYYMMDDTHHMMSS` upload-timestamp suffix,
  pores do. This is how we tell type when we build the staging folder.
- Confirm the Box top-level path is `Box-Box/Lindsay Riley/...` (no `PhD`
  suffix):
  ```bash
  ls '/Users/mimc/Library/CloudStorage/Box-Box/' | grep -i lindsay
  ```

---

## Part 1 — Regenerate meshes locally

### 1a. Pull inputs from Box to local-scratch

Only `.json` files are needed. `.dat` files in `Simulated/Dats/` are skipped
here and by the runner (see `file_type` config below).

```bash
mkdir -p /Users/mimc/local-scratch/Particles/
mkdir -p /Users/mimc/local-scratch/Subunits/

rsync -ah --progress \
  --include='*/' --include='*.json' --exclude='*' \
  '/Users/mimc/Library/CloudStorage/Box-Box/Lindsay Riley/Electronic Notebook/Void Space Project/MIMC/Data/Domains/Particles/' \
  /Users/mimc/local-scratch/Particles/

rsync -ah --progress \
  --include='*/' --include='*.json' --exclude='*' \
  '/Users/mimc/Library/CloudStorage/Box-Box/Lindsay Riley/Electronic Notebook/Void Space Project/MIMC/Data/Domains/Subunits/' \
  /Users/mimc/local-scratch/Subunits/
```

Trailing slashes on both paths are required. Without them, rsync will nest the
directory (creating `local-scratch/Particles/Particles/...`).

The first transfer is slow because each file is faulted into the Box mount
cache on read. Subsequent runs reuse the cache and are much faster.

### 1b. Configure the workflow (Particles first)

Edit `workflows/mesh_generation.py`. There is a "Local scratch" block near the
bottom of `get_config()` — comment out the Subunits pair and uncomment the
Particles pair. Never delete these variables; toggle them so switching between
runs is a two-line diff.

```python
# Local scratch (mirrors Box Particles tree; rsync outputs back to Box DomainMeshes/Particles when done)
# PARTICLES
input_dir = "/Users/mimc/local-scratch/Particles"
output_dir = "/Users/mimc/local-scratch/ParticleMeshes"
# Meshes
# input_dir = "/Users/mimc/local-scratch/Subunits"
# output_dir = "/Users/mimc/local-scratch/SubunitMeshes"
```

Verify the config dict:

- `"file_type": "json"` — restricts batch processing to `.json` only. Without
  this, batch mode processes `.json + .dat + .txt + .csv` and would double-run
  DAT-side files.
- `"batch_process": True`
- `"scrape_subdirectories": True` — required so the runner descends into
  `Real/*` and `Simulated/*` subdirectories.
- `"overwrite_existing": False` — will skip outputs that already exist. Set
  `True` only when intentionally replacing everything.

### 1c. Run

```bash
cd /Users/mimc/Documents/MIMC/Repos/SegmentationWorkflows/segmentation2dTo3d
python run.py --workflow mesh_generation
```

The runner mirrors input subdirectory structure into output, so outputs land at
`local-scratch/ParticleMeshes/Real/...` and `.../Simulated/...`.

On a first run this takes many hours. On a re-run with `overwrite_existing:
False`, it iterates all inputs but skips ones with existing outputs; log line
`Skipping X: output already exists.` appears for each. Actual mesh generation
happens only for missing outputs.

### 1d. Verify local output health

Run these checks after each category finishes:

```bash
python3 << 'EOF'
from pathlib import Path
root = Path("/Users/mimc/local-scratch/ParticleMeshes")  # or SubunitMeshes
glbs = list(root.rglob("*.glb"))
metas = list(root.rglob("*_metadata.json"))
print(f".glb count:            {len(glbs)}")
print(f"_metadata.json count:  {len(metas)}")

# Orphan check — .glb without its _metadata.json sidecar (or vice versa)
orphans_glb = [g for g in glbs if not g.with_name(g.stem + "_metadata.json").exists()]
orphans_meta = [m for m in metas if not m.with_name(m.name.replace("_metadata.json", ".glb")).exists()]
print(f"\nOrphan .glb (no metadata): {len(orphans_glb)}")
for o in orphans_glb: print(f"  {o}")
print(f"Orphan metadata (no .glb): {len(orphans_meta)}")
for o in orphans_meta: print(f"  {o}")
EOF
```

An orphan `.glb` with no metadata is almost always an interrupted mesh
generation — the raw uncompressed mesh (typically much larger than siblings —
tens or hundreds of MB) got written before the compression + metadata pass
finished. Delete both the raw `.glb` and any partial sidecar, and re-run:

```bash
rm '/path/to/orphan.glb' '/path/to/orphan_metadata.json' 2>/dev/null
python run.py --workflow mesh_generation
```

Because `overwrite_existing: False` skips everything that already has an
output, only the deleted orphan is regenerated. Let it finish uninterrupted.

### 1e. Chirality spot-check

Open 2–3 `.glb` files in a viewer (any GLB-capable viewer works; the
`documentation/mesh-generation-workflow.md` doc has visualization notes) and
confirm:

- Orientation reads correctly (particles tall along the viewer's up axis;
  trimesh's GLB exporter inserts a root-node rotation that converts Z-up
  vertex data to Y-up at render time).
- Faces are not inverted — lighting and ambient occlusion look correct.
- The accompanying `*_metadata.json` files have sensible `volume` and
  `surfArea` values.

If anything looks wrong, do not proceed to the Box push. Fix the code,
regenerate locally, re-verify.

### 1f. Repeat for Subunits

Flip `workflows/mesh_generation.py` to the Subunits pair and re-run:

```python
# input_dir = "/Users/mimc/local-scratch/Particles"
# output_dir = "/Users/mimc/local-scratch/ParticleMeshes"
input_dir = "/Users/mimc/local-scratch/Subunits"
output_dir = "/Users/mimc/local-scratch/SubunitMeshes"
```

Then rerun the same health check + chirality spot-check on
`SubunitMeshes/`.

---

## Part 2 — Push canonical archive to Box

The Box `DomainMeshes/{Particles,Subunits}/` folders mirror the local
subdirectory structure. Push order doesn't matter, but rsync in this sequence
is a good default (safe → destructive-ish):

```bash
# Subunits — creates the folder on Box if it doesn't exist
rsync -ah --progress \
  /Users/mimc/local-scratch/SubunitMeshes/ \
  '/Users/mimc/Library/CloudStorage/Box-Box/Lindsay Riley/Electronic Notebook/Void Space Project/MIMC/Data/DomainMeshes/Subunits/'

# Particles — delta only, if the folder already exists
rsync -ah --progress \
  /Users/mimc/local-scratch/ParticleMeshes/ \
  '/Users/mimc/Library/CloudStorage/Box-Box/Lindsay Riley/Electronic Notebook/Void Space Project/MIMC/Data/DomainMeshes/Particles/'
```

Trailing slashes are required on both source and destination.

### 2a. Delete stale files from Box (if any)

`rsync` (without `--delete`) doesn't remove destination files that no longer
exist in the source. If you deleted files locally (e.g. renamed outputs, fixed
truncation bug), the old names still live on Box after rsync.

To reconcile, run this diagnostic after rsync completes:

```bash
python3 << 'EOF'
from pathlib import Path
LOCAL = Path("/Users/mimc/local-scratch/ParticleMeshes")
BOX   = Path("/Users/mimc/Library/CloudStorage/Box-Box/Lindsay Riley/Electronic Notebook/Void Space Project/MIMC/Data/DomainMeshes/Particles")
local_files = {str(f.relative_to(LOCAL)) for f in LOCAL.rglob("*") if f.is_file()}
box_files   = {str(f.relative_to(BOX))   for f in BOX.rglob("*")   if f.is_file()}
stale_on_box = sorted(box_files - local_files)
print(f"Files only on Box (stale): {len(stale_on_box)}")
for f in stale_on_box[:30]: print(f"  {f}")
EOF
```

If the "stale" list is expected (files deleted intentionally locally, e.g.
misnamed outputs from a prior buggy run), delete them from Box explicitly. Use
Python for safety — shell globbing with `{`, `}`, `,` in filenames is a
minefield:

```python
from pathlib import Path
BOX = Path("/Users/mimc/Library/CloudStorage/Box-Box/Lindsay Riley/Electronic Notebook/Void Space Project/MIMC/Data/DomainMeshes/Particles")
# Populate `targets` with Path objects from the stale list above
for p in targets: p.unlink()
```

Alternatively, add `--delete` to your rsync — but only after verifying the
"stale" list to be sure you're not deleting anything you meant to keep.

### 2b. Wait for Box's cloud upload

The rsync `cp` to the mount completes quickly, but the actual upload to Box's
servers happens in the background. Watch the Box menu bar icon until "Syncing
N items" clears.

Spot-check via the Box web UI that a couple of the new `.glb` files are
actually there with the right size.

---

## Part 3 — Build `ProdReplace/` (flat, prod-name-matching)

Prod stores files with **exact-match filenames** derived from the original
upload names. Some of those names differ from the source filenames in Box
`Domains/` — e.g. stiffness variants that were normalized `.{` → `_{` at
upload time. Building `ProdReplace/` means renaming/copying local outputs into
a flat folder where every file's name matches its prod entry exactly.

### 3a. Fetch the current prod filename list

Hit the endpoint that returns `category, filename` pairs and save to
`mesh-original-filenames.txt` at the repo root. The number of rows should
equal `(prod particle mesh count) + (prod pore mesh count)`.

### 3b. Build the manifest (dry-run)

This is a read-only sanity check. Confirms every prod entry maps to a local
source, reports any missing files, and shows how duplicates get resolved.

```python
import re, json
from pathlib import Path
from collections import defaultdict

prod_entries = []
for line in Path("mesh-original-filenames.txt").read_text().splitlines():
    line = line.strip()
    if not line: continue
    cat, _, fname = line.partition(",")
    prod_entries.append((cat.strip(), fname.strip()))

def index(root):
    out = defaultdict(list)
    for p in Path(root).rglob("*.glb"):
        out[p.name].append(p)
    return out
local_p = index("/Users/mimc/local-scratch/ParticleMeshes")
local_s = index("/Users/mimc/local-scratch/SubunitMeshes")

def strip_ts(n): return re.sub(r"@\d{8}T\d{6}(?=\.glb$)", "", n)

def find_particle_source(prod_name):
    if prod_name in local_p: return local_p[prod_name], "exact"
    canon = strip_ts(prod_name)
    if canon in local_p: return local_p[canon], "canonical"
    # Stiffness normalization: prod uses `_{`, local (from source) uses `.{`
    candidate = prod_name.replace("_{", ".{", 1)
    if candidate in local_p: return local_p[candidate], "stiffness-rename"
    return [], "MISSING"

def find_pore_source(prod_name):
    if prod_name in local_s: return local_s[prod_name], "exact"
    canon = strip_ts(prod_name)
    if canon in local_s: return local_s[canon], "canonical"
    return [], "MISSING"

def pick_dupe(paths):
    """When multiple candidates, prefer DatsJsonified subdir."""
    if len(paths) == 1: return paths[0]
    dj = [p for p in paths if "DatsJsonified" in p.parts]
    return dj[0] if dj else paths[0]

manifest = []
missing = []
stats = defaultdict(int)
for cat, prod_name in prod_entries:
    finder = find_particle_source if cat == "Particles" else find_pore_source
    sources, mode = finder(prod_name)
    stats[f"{cat}:{mode}"] += 1
    if not sources:
        missing.append((cat, prod_name))
        continue
    chosen = pick_dupe(sources)
    meta = chosen.with_name(chosen.stem + "_metadata.json")
    manifest.append({
        "category": cat,
        "target_name": prod_name,
        "source_glb": str(chosen),
        "source_metadata": str(meta) if meta.exists() else None,
        "mode": mode,
    })

print("=== MAPPING MODES ===")
for k in sorted(stats): print(f"  {k}: {stats[k]}")
print(f"\nMapped: {len(manifest)} / {len(prod_entries)}")
print(f"Missing: {len(missing)}")
for cat, n in missing: print(f"  {cat}: {n}")
print(f"Metadata sidecars: {sum(1 for m in manifest if m['source_metadata'])} / {len(manifest)}")
total = sum(Path(m['source_glb']).stat().st_size + (Path(m['source_metadata']).stat().st_size if m['source_metadata'] else 0) for m in manifest)
print(f"Total bytes to copy: {total/1e9:.2f} GB")

Path("/tmp/prodreplace_manifest.json").write_text(json.dumps(manifest, indent=1))
print("Saved /tmp/prodreplace_manifest.json")
```

The dry-run should show `Missing: 0` and `Metadata sidecars: N / N`. If not,
stop and diagnose — either a prod entry has no local source (regenerate it or
find where it lives), or a mesh was written without its metadata sidecar
(orphan check from Part 1d).

The `stiffness-rename` mapping mode handles particle prod names like
`labeledDomain_spheres_s01_{soft-0.25,hard-0.75}.glb` by looking up the local
file `labeledDomain_spheres_s01.{soft-0.25,hard-0.75}.glb` (dot vs
underscore). The rename is only applied to particles; pore prod names already
match local pore names exactly.

The `pick_dupe` logic handles the 149 particle names that appear in both
`ParticleMeshes/Simulated/DatsJsonified/` and `ParticleMeshes/Simulated/Jsons/`
— both come from byte-identical source JSONs, so `DatsJsonified/` is picked
deterministically. See "Common pitfalls" for the underlying source-duplication
issue.

### 3c. Materialize the folder

```python
import json, shutil
from pathlib import Path

manifest = json.loads(Path("/tmp/prodreplace_manifest.json").read_text())
target_root = Path("/Users/mimc/local-scratch/ProdReplace")
for sub in ("Particles", "Pores"):
    (target_root / sub).mkdir(parents=True, exist_ok=True)

for m in manifest:
    cat_folder = "Particles" if m["category"] == "Particles" else "Pores"
    target_glb  = target_root / cat_folder / m["target_name"]
    target_meta = target_root / cat_folder / (m["target_name"][:-4] + "_metadata.json")
    shutil.copy2(m["source_glb"], target_glb)
    if m["source_metadata"]:
        shutil.copy2(m["source_metadata"], target_meta)

# Verify counts
for sub in ("Particles", "Pores"):
    p = target_root / sub
    print(f"ProdReplace/{sub}: {len(list(p.glob('*.glb')))} .glb + {len(list(p.glob('*_metadata.json')))} metadata")
```

Local SSD copy is fast — 4-5 GB completes in ~10 seconds. Idempotent, safe to
re-run.

Spot-check that the stiffness rename actually happened:

```bash
ls /Users/mimc/local-scratch/ProdReplace/Particles/labeledDomain_spheres_s0*.glb | head -3
```

Filenames should have `_{` (not `.{`) before the stiffness ratio.

### 3d. Push `ProdReplace/` to Box

```bash
rsync -ah --progress \
  /Users/mimc/local-scratch/ProdReplace/ \
  '/Users/mimc/Library/CloudStorage/Box-Box/Lindsay Riley/Electronic Notebook/Void Space Project/MIMC/Data/DomainMeshes/ProdReplace/'
```

Wait for Box cloud sync to complete (same story as Part 2b).

---

## Part 4 — Push `ProdReplace/` to prod (`lovamap-01`)

### 4a. Pre-flight

- Box cloud upload for `ProdReplace/` is fully complete (menu bar clear, web
  UI shows all files).
- Chirality spot-check done on 2-3 files under `ProdReplace/` (Part 1e criteria).
- Maintenance window scheduled. The rsync itself is fast, but the LOVAMAP
  gateway may need to be restarted after so it picks up new mesh contents.

### 4b. rsync from Box to `lovamap-01`

Command template (fill in your SSH details / target path):

```bash
rsync -ahz --progress \
  '/Users/mimc/Library/CloudStorage/Box-Box/Lindsay Riley/Electronic Notebook/Void Space Project/MIMC/Data/DomainMeshes/ProdReplace/Particles/' \
  dra20@lovamap-01:/srv/lovamap-shared-data/lovamap-gw/production/Domains/Particles-staging/

rsync -ahz --progress \
  '/Users/mimc/Library/CloudStorage/Box-Box/Lindsay Riley/Electronic Notebook/Void Space Project/MIMC/Data/DomainMeshes/ProdReplace/Pores/' \
  dra20@lovamap-01:/srv/lovamap-shared-data/lovamap-gw/production/Domains/Pores-staging/
```

The exact destination path depends on how the gateway ingests these — is it
a flat `Domains/` folder with UUID-named files (in which case a rename step
on the server is needed), or a staging folder that the gateway atomically
promotes? Confirm with the gateway team before running.

### 4c. Verify prod

- Refetch `mesh-original-filenames.txt` from the endpoint and compare row
  count and category split to what was there before.
- Spot-check a few meshes via the gateway UI/API.
- If chirality is still wrong after the push, all mesh regeneration was for
  nothing. Investigate before touching prod again.

---

## Part 5 — Cleanup

Once prod is confirmed correct and Box has synced:

```bash
# Local scratch — safe to delete once you're confident in prod
rm -rf /Users/mimc/local-scratch/{Particles,Subunits,ParticleMeshes,SubunitMeshes,ProdReplace}
```

If disk isn't tight, keeping the scratch dirs around for a day or two after
prod push is a free safety net.

---

## Common pitfalls

### rsync-related

- **Missing trailing slash on the rsync source.** Without it, rsync copies
  the source directory itself as a child of the destination, nesting it one
  level deeper than intended (`local-scratch/Particles/Particles/...`).
  Always include the trailing slash on both sides.
- **rsync doesn't remove destination files not present in source.** After
  renaming/deleting files locally, stale copies remain on Box. Run the
  Part 2a diagnostic and clean up.
- **Pushing to Box before Box's cloud upload finishes.** The rsync to the
  mount completes instantly; the cloud upload runs in the background. Wait
  for the Box menu bar to clear before treating "pushed to Box" as done.
- **Deleting local-scratch before Box finishes uploading.** Same problem as
  above.

### Workflow config

- **`scrape_subdirectories: False` when input has nested folders.** The runner
  will find zero files. Both particle and subunit input trees require this
  set to `True`.
- **`overwrite_existing: True` on a full regen when you meant to resume.** All
  existing outputs get regenerated. Set `True` only when intentionally
  replacing everything.
- **`file_type` unset.** In batch mode, the runner defaults to processing
  `.json + .dat + .txt + .csv`. For these reruns, keep `"file_type": "json"`.
- **Path typo: `Lindsay Riley` vs `Lindsay Riley PhD`.** Older commented paths
  in `mesh_generation.py` use the `PhD` suffix; the current Box mount uses
  just `Lindsay Riley`. Confirm with `ls` before kicking off a long transfer.

### Filename gotchas

- **`.split('.')[0]` truncation bug (fixed 2026-07).** The runner used to
  derive output name via `basename.split('.')[0]`, which truncates at the
  *first* dot. This broke any input filename with `.` in the base name — most
  notably the stiffness variants `labeledDomain_spheres_sXX.{soft-X.XX,hard-X.XX}.json`,
  which all got written as `sXX.glb` with the stiffness suffix lost. Fixed
  by using `os.path.splitext(basename)[0]` (see `workflow_runner/mesh_generation.py:15,17`).
  If you see suspiciously short output names, this bug regressed.
- **`.{` vs `_{` in stiffness names.** Source `.json` files in Box use `.{`
  before the stiffness ratio; prod stores them with `_{`. The mesh runner
  faithfully copies the source name. The rename is applied only at Part 3
  (staging to `ProdReplace/`).
- **Duplicate source data in Box.** `Box/Domains/Particles/Simulated/Jsons/`
  historically contained a 149-file byte-identical subset of
  `DatsJsonified/`. That produced duplicate outputs locally. The manifest's
  `pick_dupe` picks `DatsJsonified/` in all cases (they're identical), but
  the underlying source duplication should be cleaned up on Box separately.

### Interrupted runs

- **Interrupted mesh generation leaves an orphan `.glb` without metadata.**
  Signature: a suspiciously large `.glb` (10-100x the size of its siblings —
  it's the raw uncompressed mesh) with no matching `_metadata.json`. On a
  restart with `overwrite_existing: False`, the runner sees the `.glb` and
  skips it, so the orphan is never regenerated. The Part 1d health check
  catches this — delete the raw `.glb` and re-run to fix.
- **The largest meshes are the ones most likely to get interrupted.** The
  `_6.glb` in the `beadInfo_spheres_{50,100}_0_{100,0}_*` set is a known
  ~300 MB raw / ~20 MB compressed file — do not interrupt runs when it's
  processing.

### Prod-side

- **Type ambiguity from filename alone.** In the prod list, a name like
  `Yining_S_3_segment_100.glb` could be a particle or a pore. The endpoint
  must return an explicit `category` (`Particles` or `Pores`) — do not try to
  infer from the filename.
- **Prod may serve files whose source is not in current Box `Domains/`.**
  Historical uploads from experimental pipelines may live only in prod. Any
  such gap will surface as `MISSING` in the Part 3b dry-run. Track those
  down before finalizing the manifest, or accept that they'll stay at
  pre-fix chirality until their sources are located.
- **Prod filename normalization is category-specific.** Particles: only the
  `.` immediately before `{` becomes `_`; dots inside `0.25` stay. Pores:
  all dots and commas inside `{...}` become underscores (`0_25` instead of
  `0.25`), and `@YYYYMMDDTHHMMSS` upload timestamps are appended. The
  manifest logic in Part 3b already handles both.
