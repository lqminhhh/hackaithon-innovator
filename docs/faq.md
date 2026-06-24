# FAQ / Frequent Issues

This file collects practical issues that judges or developers may hit when
trying to run the submission container.

## 1. `docker run` fails because Docker is not running

Typical symptoms:

- `Cannot connect to the Docker daemon`
- `docker version` shows the client but not the server

What this means:

- Docker CLI is installed
- but the Docker daemon is not running or not reachable

Normal fix on a standard machine:

- start Docker Desktop, or
- start the Docker service on the host

Quick check:

```bash
docker version
```

You should see both:

- `Client`
- `Server`

If the server section is missing, Docker is not ready yet.

## 2. In notebook or restricted cloud environments, Docker may need manual startup

Some environments do not start the Docker daemon automatically. In those cases,
users may need to start `dockerd` manually before `docker run` works.

Example pattern:

```bash
pkill dockerd || true
pkill containerd || true

dockerd \
  --host=unix:///var/run/docker.sock \
  --storage-driver=vfs \
  --iptables=false \
  --bridge=none \
  --ip-forward=false \
  --ip-masq=false \
  >/tmp/dockerd.log 2>&1 &

sleep 25
docker version
```

Why this happens:

- the environment has Docker installed
- but does not provide a running Docker daemon by default

This is an environment setup issue, not a submission-specific requirement.

## 3. `docker run --gpus all` does not work

Likely causes:

- NVIDIA Container Toolkit is not installed
- Docker is running, but GPU passthrough is not configured
- the machine does not expose the NVIDIA GPU to Docker

Quick check:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this fails, the GPU container runtime is not ready yet.

## 4. `vLLM unavailable`

This message usually means the fast GPU backend failed to initialize.

Common causes:

- unsupported or fragile GPU/backend combination
- incompatible driver/runtime environment
- broken local Python environment outside Docker

If you are using the official Docker image, prefer debugging the host GPU setup
first before changing the repository code.

## 5. The container starts but says no input file was found

The submission entrypoint expects one of:

- `/data/private_test.csv`
- `/data/public_test.csv`
- `/data/private_test.json`
- `/data/public_test.json`

Make sure the host mount is correct:

```bash
-v "$PWD/data:/data"
```

And make sure the file inside `data/` uses one of the names above.

## 6. Where is the output written?

The submission container writes:

```text
/output/pred.csv
```

Make sure the host output mount exists:

```bash
mkdir -p output
```

And run with:

```bash
-v "$PWD/output:/output"
```
