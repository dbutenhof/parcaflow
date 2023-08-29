# Package path for this plugin module relative to the repo root
ARG package=parcaflow

# STAGE 1 -- Build module dependencies and run tests
# The 'poetry' and 'coverage' modules are installed and verson-controlled in the
# quay.io/arcalot/arcaflow-plugin-baseimage-python-buildbase image to limit drift
FROM quay.io/arcalot/arcaflow-plugin-baseimage-python-buildbase:0.2.0@sha256:7b72424c08c51d1bb6215fac0e002fd9d406b6321dcd74233ea53ec653280be8 as build

COPY poetry.lock /app/
COPY pyproject.toml /app/

# Convert the dependencies from poetry to a static requirements.txt file
RUN python -m poetry install --without dev --no-root \
 && python -m poetry export -f requirements.txt --output requirements.txt --without-hashes

COPY ${package}/ /app/${package}
COPY tests /app/${package}/tests

ENV PYTHONPATH /app/${package}
WORKDIR /app/${package}

# Run tests and return coverage analysis
#RUN python -m coverage run tests/test_${package}.py \
# && python -m coverage html -d /htmlcov --omit=/usr/local/*


# STAGE 2 -- Build final plugin image
FROM quay.io/arcalot/arcaflow-plugin-baseimage-python-osbase:0.2.0@sha256:a57baf7714d13b4fb0a01551990eed927b1f1251cd502ad01bcb05ffeeff31d8
ARG package

COPY --from=build /app/requirements.txt /app/
COPY --from=build /htmlcov /htmlcov/
COPY LICENSE /app/
COPY README.md /app/
COPY ${package}/ /app/${package}

# We rely on podman commands within the container, so we need to install the
# podman-remote command inside the container and provide it with the command
# socket from the host system. The socket can be configured via containers.conf
# or by `--url`, but we'll just map it into the container at the default path
# /run/user/$UID/podman/podman.sock. We bind a listener to the socket outside
# the contaner by starting the service with the command
# `systemctl --user enable --now podman.socket`.
RUN dnf install -y podman-remote

# Install all plugin dependencies from the generated requirements.txt file
RUN python -m pip install -r requirements.txt

WORKDIR /app/${package}

ENTRYPOINT ["python", "parcaflow.py"]
CMD []

LABEL org.opencontainers.image.source="https://github.com/arcalot/arcaflow-plugin-template-python"
LABEL org.opencontainers.image.licenses="Apache-2.0+GPL-2.0-only"
LABEL org.opencontainers.image.vendor="Arcalot project"
LABEL org.opencontainers.image.authors="Arcalot contributors"
LABEL org.opencontainers.image.title="Python Plugin Template"
LABEL io.github.arcalot.arcaflow.plugin.version="1"

# `buildah commit <sha256> localhost/parcaflow`
