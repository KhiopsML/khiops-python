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

There are three Dockerfiles, one per OS family:

| Dockerfile | OS targets | Package manager | Remote file drivers |
|---|---|---|---|
| `Dockerfile.ubuntu` | Ubuntu 22.04 | apt | System-wide `.deb` (GCS, S3, Azure) + fakeS3 |
| `Dockerfile.debian` | Debian 13 | apt | System-wide bookworm `.deb` (GCS, S3, Azure) + fakeS3 |
| `Dockerfile.rocky` | Rocky 8, Rocky 9 | dnf | None |

All images are published to `ghcr.io/khiopsml/khiops-python/khiopspydev-<os>`.

### Debian and Ubuntu

`Dockerfile.debian` and `Dockerfile.ubuntu` are nearly identical. They diverge
because Debian 13 remote file driver packages are not available, so Debian
forces the Debian 12 (bookworm) builds for the GCS, S3, and Azure driver `.deb`
packages. Any shared change should be applied to both files. There is an open
TODO to unify them (see the comment at the top of `Dockerfile.debian`).

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
| `KHIOPS_REVISION` | Khiops native release tag to install | `11.0.0` |
| `SERVER_REVISION` | Git ref for the `khiops-server` image (copied into the final stage) | `main` |
| `PYTHON_VERSIONS` | Space-separated Python versions for Conda environments | `3.10 3.11 3.12 3.13 3.14` |
| `KHIOPS_GCS_DRIVER_REVISION` | GCS remote file driver version (Debian/Ubuntu only) | `0.0.16` |
| `KHIOPS_S3_DRIVER_REVISION` | S3 remote file driver version (Debian/Ubuntu only) | `0.0.15` |
| `KHIOPS_AZURE_DRIVER_REVISION` | Azure remote file driver version (Debian/Ubuntu only) | `0.0.6` |

## Multi-Stage Build Structure

Each Dockerfile uses a multi-stage build:

1. **`khiopsdev`** — Based on `ghcr.io/khiopsml/khiops/khiopsdev-<os>:latest`.
   Installs dev tools (git, pip, pandoc, wget), the Khiops native binary, and
   Miniforge for Conda. On Debian/Ubuntu, also installs system-wide remote file
   drivers (GCS, S3, Azure `.deb` packages) and `ruby-dev` (needed for fakeS3).
2. **`server`** — Pulls the `khiops-server` binary from
   `ghcr.io/khiopsml/khiops-server:<SERVER_REVISION>`.
3. **`base`** (final) — Copies the server binary into the `khiopsdev` stage. On
   Ubuntu/Debian, also installs fakeS3 via Ruby gem and exposes port 4569.

## Conda Environments

For each Python version in `PYTHON_VERSIONS`, a Conda environment is created:

- **`py<version>`** — Bare Python (for pip-based test installs).

A special **`py3_khiops10_conda`** environment is always created with
`khiops-core==10.3.2` to test backward compatibility with Khiops major
version 10.

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
- **Image tags**: `<KHIOPS_REVISION>.<IMAGE_INCREMENT>` (e.g., `11.0.0.0`),
  optionally also `latest`.
- **Build context**: `./packaging/docker/khiopspydev/`.
- **Build args**: Passes `KHIOPS_REVISION`, `KHIOPSDEV_OS`, `SERVER_REVISION`,
  `PYTHON_VERSIONS`, `KHIOPS_GCS_DRIVER_REVISION`, `KHIOPS_S3_DRIVER_REVISION`,
  and `KHIOPS_AZURE_DRIVER_REVISION`.
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
- **Remote file drivers**: Version bumps go in the
  `DEFAULT_KHIOPS_GCS_DRIVER_REVISION` / `DEFAULT_KHIOPS_S3_DRIVER_REVISION` /
  `DEFAULT_KHIOPS_AZURE_DRIVER_REVISION` env vars in `dev-docker.yml`. Note:
  there is a known workaround in the Ubuntu and Debian Dockerfiles for a release
  tag typo in the Azure driver repository (the download URL hard-codes tag
  `0.0.7` regardless of the revision ARG — see the `XXX` comment).
- **fakeS3**: Pinned to version `1.2.1` because `>= 1.3` requires a license key.
  If fakeS3 becomes incompatible, consider alternatives (the Dockerfile comments
  mention `s3rver` as a candidate).
- After modifying Dockerfiles, **images are rebuilt and pushed** via a manual
  `dev-docker.yml` run with `push: true` before merging, so that consumer
  workflows pick up the changes.
