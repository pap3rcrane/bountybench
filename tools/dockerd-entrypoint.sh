#!/bin/bash

# SSH Key Setup
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [ -f /root/.ssh/id_rsa ]; then
  chmod 600 /root/.ssh/id_rsa
  eval "$(ssh-agent -s)"
  ssh-add /root/.ssh/id_rsa
  echo "[entrypoint] SSH key loaded."
else
  echo "[entrypoint] No SSH key at /root/.ssh/id_rsa – skipping."
fi

# Function to check if Docker daemon is already running
check_dockerd() {
    # Use 'docker info' to check if the daemon is responsive
    docker info > /dev/null 2>&1
    return $?
}

echo "[entrypoint] Checking if Docker daemon is already running..."
if check_dockerd; then
    echo "[entrypoint] Docker daemon is already running, skipping startup."
else
    echo "[entrypoint] Starting Docker daemon..."
    DOCKERD_ARGUMENTS=()
    if [ -n "${DOCKER_REGISTRY_MIRROR:-}" ]; then
        DOCKERD_ARGUMENTS+=(--registry-mirror "$DOCKER_REGISTRY_MIRROR")
        if [[ "$DOCKER_REGISTRY_MIRROR" == http://* ]]; then
            MIRROR_HOST="${DOCKER_REGISTRY_MIRROR#http://}"
            DOCKERD_ARGUMENTS+=(--insecure-registry "$MIRROR_HOST")
        fi
        echo "[entrypoint] Using Docker registry mirror: $DOCKER_REGISTRY_MIRROR"
    fi
    dockerd "${DOCKERD_ARGUMENTS[@]}" > /var/log/dockerd.log 2>&1 &

    echo "[entrypoint] Waiting for Docker daemon to come up..."
    # Wait up to 30 seconds for the daemon to start
    timeout 30 sh -c "while (! docker info > /dev/null 2>&1); do sleep 1; done"
    if [ $? -ne 0 ]; then
        echo "[entrypoint] Error: Docker daemon failed to start within 30 seconds."
        echo "[entrypoint] Contents of /var/log/dockerd.log:"
        cat /var/log/dockerd.log
        exit 1
    fi
    echo "[entrypoint] Docker daemon is running."
fi

if [ -n "${DOCKERHUB_USERNAME:-}" ] || [ -n "${DOCKERHUB_TOKEN:-}" ]; then
    if [ -z "${DOCKERHUB_USERNAME:-}" ] || [ -z "${DOCKERHUB_TOKEN:-}" ]; then
        echo "[entrypoint] Both DOCKERHUB_USERNAME and DOCKERHUB_TOKEN are required."
        exit 1
    fi
    if ! printf '%s' "$DOCKERHUB_TOKEN" \
        | docker login --username "$DOCKERHUB_USERNAME" --password-stdin; then
        echo "[entrypoint] Docker Hub login failed."
        exit 1
    fi
    echo "[entrypoint] Docker Hub authentication configured."
fi

echo "[entrypoint] Starting main process: $@"
exec "$@"
