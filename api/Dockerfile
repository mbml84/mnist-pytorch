FROM python:3.10-slim-bullseye as build

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_VERSION=1.2.2

WORKDIR app/

# Poetry install
RUN pip install "poetry==$POETRY_VERSION"

# copy project dependencies files here to ensure they will be cached
COPY poetry.lock pyproject.toml ./

# install runtime dependencies
RUN poetry export --with dev --without-hashes --no-interaction --no-ansi -f requirements.txt -o requirements.txt
RUN pip install -r requirements.txt --use-deprecated=legacy-resolver

COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/

RUN useradd ubuntu \
    && chmod -R a+w /var/run/ \
    && chmod -R a+w /var/log \
    && usermod -aG sudo ubuntu \
    && echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

EXPOSE 8000
