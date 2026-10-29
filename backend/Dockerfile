FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicit copies instead of `COPY . .` -- deliberately does NOT include
# data/. `COPY . .` would otherwise pull in every hospital's raw CSV
# (backend/data/hospitalN/*.csv) straight into this image, which defeats
# the whole point of splitting hospitals into their own containers with
# their own images (see fl/Dockerfile). Removing the docker-compose volume
# mount alone was not enough to close that gap -- the data would still have
# ended up baked into this image's layers at build time otherwise. If you
# add a new top-level backend/ directory, it needs its own line here to be
# included; that's intentional friction, not an oversight.
COPY main.py .
COPY api/ api/
COPY services/ services/
COPY db/ db/
COPY blockchain/ blockchain/
COPY chaincode/ chaincode/
COPY scripts/ scripts/
COPY fl/ fl/

ENV DATABASE_URL=postgresql://postgres:postgres@postgres:5432/vienko

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

