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

- the `nvidia-container-toolkit` package is not installed
- Docker is running, but GPU passthrough is not configured
- the machine does not expose the NVIDIA GPU to Docker

Quick check:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

If this fails, the GPU container runtime is not ready yet.

On Linux hosts, make sure the NVIDIA driver works first:

```bash
nvidia-smi
```

Then make sure Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

If host `nvidia-smi` works but Docker `nvidia-smi` fails, the usual missing
piece is `nvidia-container-toolkit` or Docker's NVIDIA runtime configuration.
After installing or reconfiguring the toolkit, restart Docker before rerunning
the preflight check.

## 4. Which GPUs are supported?

Officially supported:

- NVIDIA Ampere or newer CUDA GPUs with at least 16 GB VRAM
- examples: RTX 3090/4090, RTX 4080 16 GB, RTX A5000/A6000, A100, L4, or
  similar CUDA capable GPUs

Technically supported but not recommended:

- Tesla T4 16 GB

We ask judges not to use T4 for the official run. T4 can be too slow, too close
to the memory limit, and more likely to show Docker/vLLM runtime mismatch or
execution issues. If T4 is the only available GPU, allocate much more runtime
and run the preflight checks carefully before starting the full private set.

## 5. Why can a 2000 question run take 30 hours or more?

The final runner uses conservative settings because the exact judge GPU is
unknown. On our tested 16 GB VRAM GPU, a private set around 2000 questions can
take 30 hours or more, especially when many questions enter Wave 2
self-consistency.

This does not necessarily mean the container is stuck. Wave 2 is slower because
it repeats reasoning on harder questions. On 16 GB hardware, batching is also
more conservative to avoid OOM.

Practical advice:

- allocate at least 30 hours for a 2000 question run on 16 GB VRAM
- use a GPU with more VRAM if available
- avoid T4 for official judging
- keep the output directory mounted so partial/final `pred.csv` writes are
  preserved

## 6. Not enough disk space to pull or run the image

The Docker image is approximately 16.2 GB. We recommend at least 25 GB free
disk space so Docker has room for the image, extracted layers, cache, and output
files.

Check available space:

```bash
df -h .
docker system df
```

If space is low, remove unused Docker data only if it is safe for your machine:

```bash
docker system prune
```

Do not run prune if you need to keep unused local images or containers.

## 7. `vLLM unavailable`

This message usually means the fast GPU backend failed to initialize.

Common causes:

- unsupported or fragile GPU/backend combination
- incompatible driver/runtime environment
- broken local Python environment outside Docker
- missing Docker GPU passthrough because `nvidia-container-toolkit` is not
  installed or not configured
- using a technically supported but fragile GPU such as T4

If you are using the official Docker image, prefer debugging the host GPU setup
first before changing the repository code.

## 8. The container starts but says no input file was found

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

## 9. Where is the output written?

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
