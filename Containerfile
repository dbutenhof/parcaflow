# Package path for this plugin module relative to the repo root
ARG package=parcaflow

# STAGE 1 -- Build module dependencies and run tests
# The 'poetry' and 'coverage' modules are installed and verson-controlled in the
# quay.io/arcalot/arcaflow-plugin-baseimage-python-buildbase image to limit drift
FROM quay.io/pbench/pbench-agent-all-centos-8:b0.73 as build

COPY poetry.lock /app/
COPY pyproject.toml /app/

RUN python3 --version
RUN dnf install -y python3.10 python3-pip
RUN python3 --version

RUN python3 -m pip install poetry==1.4.2 \
 && python3 -m poetry config virtualenvs.create false

ENV PYTHONPATH /app/${package}
WORKDIR /app/${package}

# Setup coverage
RUN python3 -m pip install coverage==7.2.7 \
 && mkdir /htmlcov
# Convert the dependencies from poetry to a static requirements.txt file
# RUN python3 -m poetry install --without dev --no-root \
#  && python3 -m poetry export -f requirements.txt --output requirements.txt --without-hashes

COPY ${package}/ /app/${package}
COPY tests /app/${package}/tests

# Run tests and return coverage analysis
#RUN python -m coverage run tests/test_${package}.py \
# && python -m coverage html -d /htmlcov --omit=/usr/local/*


# STAGE 2 -- Build final plugin image
FROM quay.io/pbench/pbench-agent-all-centos-8:b0.73
ARG package=parcaflow

RUN dnf -y module install python39 && dnf install -y python3.9 python3-pip
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN python --version

# COPY --from=build /app/requirements.txt /app/
# COPY --from=build /htmlcov /htmlcov/
COPY LICENSE /app/
COPY README.md /app/
COPY requirements.txt /app/
COPY ${package}/ /app/${package}

WORKDIR /app/${package}

# Install all plugin dependencies from the generated requirements.txt file
RUN python -m pip install -r ../requirements.txt

ENTRYPOINT ["python", "parcaflow.py"]
CMD []

LABEL org.opencontainers.image.source="https://github.com/arcalot/arcaflow-plugin-template-python"
LABEL org.opencontainers.image.licenses="Apache-2.0+GPL-2.0-only"
LABEL org.opencontainers.image.vendor="Arcalot project"
LABEL org.opencontainers.image.authors="Arcalot contributors"
LABEL org.opencontainers.image.title="Python Plugin Template"
LABEL io.github.arcalot.arcaflow.plugin.version="1"

# `buildah commit <sha256> localhost/parcaflow`
