---
applyTo: "packaging/docker/khiopspydev/**"
---

# Docker Packaging Changes

Use these rules for files under `packaging/docker/khiopspydev/`. Apply the
shared guidance from `.github/copilot-instructions.md` first, then this
packaging-specific guidance.

## Scope

The Dockerfiles in `packaging/docker/khiopspydev/` build **development images**
used by the CI workflows `tests.yml`, `pip.yml`, and `api-docs.yml` to run
tests, build packages, and generate documentation. They are built and published
by the `dev-docker.yml` workflow.

## Image Variants

There are two Dockerfiles:

| Dockerfile                 | OS targets       | Package manager | Remote file drivers |
|----------------------------|------------------|---|---|
| `Dockerfile.ubuntu-debian` | Ubuntu 22.04 or Debian 13 | apt | System-wide `.deb` (GCS, S3, Azure) + fakeS3 |
| `Dockerfile.rocky`         | Rocky 8, Rocky 9 | dnf | None |

All images are published to `ghcr.io/khiopsml/khiops-python/khiopspydev-<os>`.

### Debian and Ubuntu

`Dockerfile.ubuntu-debian` is a unique Dockerfile to build either a Ubuntu or a Debian target image 
because there is no difference except for the base image used.

### Rocky

`Dockerfile.rocky` uses `dnf` instead of `apt` and installs Python 3.11
explicitly on Rocky 8/9 (which ship with Python ≤ 3.9). It does **not** install
system-wide remote file drivers or fakeS3 (no Ruby/gem support). It also does
not copy `run_fake_remote_file_servers.sh` (only `run_service.sh`).

## Build Arguments

All Dockerfiles accept these `ARG` values, supplied by `dev-docker.yml`:

| ARG | Description | Example |
|---|---|---|
| `KHIOPSDEV_OS` | OS tag for the base image (`ubuntu22.04`, `rocky8`, `rocky9`, `debian13`) | `ubuntu22.04` |
| `SERVER_REVISION` | Git ref for the `khiops-server` image (copied into the final stage) | `main` |
| `PYTHON_VERSIONS` | Space-separated Python versions for Conda environments | `3.10 3.11 3.12 3.13 3.14` |

## Multi-Stage Build Structure

Each Dockerfile uses a multi-stage build:

1. **`khiopsdev`** — Based on `ghcr.io/khiopsml/khiops/khiopsdev-<os>:latest`.
   Installs dev tools (git, pip, pandoc, wget, zip, unzip) and Miniforge for Conda. 
   On Debian/Ubuntu, also installs `ruby-dev` (needed for fakeS3).
2. **`server`** — Pulls the `khiops-server` binary from
   `ghcr.io/khiopsml/khiops-server:<SERVER_REVISION>`.
3. **`base`** (final) — Copies the server binary into the `khiopsdev` stage. On
   Ubuntu/Debian, also installs fakeS3 via Ruby gem and exposes port 4569.

## Conda Environments

For each Python version in `PYTHON_VERSIONS`, a Conda environment is created:

- **`py<version>`** — Bare Python (for pip-based test installs).

## Helper Scripts

| Script | Purpose |
|---|---|
| `run_service.sh` | Runs `/usr/bin/service` (the khiops-server binary) if present; otherwise exits. Copied into all images. |
| `run_fake_remote_file_servers.sh` | Launches fakeS3 in background on the port extracted from `$AWS_ENDPOINT_URL`. Serves pre-provisioned files from `tests/resources/remote-access`. Copied into Ubuntu/Debian images only. |

## CI Workflow Integration

### `dev-docker.yml` (Build and Push)

- **Triggers**: PRs touching `Dockerfile.*` or the workflow file;
  `workflow_dispatch` for manual builds.
- **Matrix**: `ubuntu22.04`, `rocky8`, `rocky9`, `debian13`.
- **Concurrency**: Per-workflow + per-PR/ref, with `cancel-in-progress: true`.
- **Push**: Only on manual dispatch with `push: true`. The `set-latest` tag is
  restricted to the `main` or `main-v10` branches.
- **Image tags**: `<CURRENT_IMAGE_TAG>` (e.g., `pip-packages`),
  optionally also `latest`.
- **Build context**: `./packaging/docker/khiopspydev/`.
- **Build args**: Passes `KHIOPSDEV_OS`, `SERVER_REVISION`,
  `PYTHON_VERSIONS`.
- **Important**: The `add-hosts: s3-bucket.localhost:127.0.0.1` input is required
  because buildx mounts `/etc/hosts` read-only, so the fakeS3 hostname cannot
  be added inside the Dockerfile.

### `tests.yml` (Consumer)

Runs the main test matrix inside `khiopspydev-ubuntu22.04` containers across
Python 3.10–3.14. Integration tests also run on `rocky8`, `rocky9`, and
`debian13` containers.

### `pip.yml` (Consumer)

Builds and tests the source distribution package inside
`khiopspydev-ubuntu22.04`, `khiopspydev-rocky9`, and `khiopspydev-debian13`
containers.

### `api-docs.yml` (Consumer)

Builds the Sphinx documentation inside the `khiopspydev-ubuntu22.04` container.

## Editing Rules

- **Apply shared changes to all relevant Dockerfiles.** Ubuntu and Debian
  Dockerfiles are near-duplicates; always update both.
- **Bump Miniforge version** by updating the download URL, filename, and SHA-256
  checksum in all three Dockerfiles.
- **Adding Python versions**: Update `DEFAULT_PYTHON_VERSIONS` in
  `dev-docker.yml` and add the new version to the `matrix.python-version` lists
  in consumer workflows (`tests.yml`, etc.).
- **Adding system dependencies**: Install them in the appropriate `RUN` block
  (apt for Debian/Ubuntu, dnf for Rocky).
- **fakeS3**: Pinned to version `1.2.1` because `>= 1.3` requires a license key.
  If fakeS3 becomes incompatible, consider alternatives (the Dockerfile comments
  mention `s3rver` as a candidate).
- After modifying Dockerfiles, **images are rebuilt and pushed** via a manual
  `dev-docker.yml` run with `push: true` before merging, so that consumer
  workflows pick up the changes.
