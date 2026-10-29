# Docker demo (backend + Postgres + MinIO + MLflow + 3 real hospital containers)

## Topology

`docker-compose up` builds and starts, in addition to Postgres/MinIO/MLflow:

- `fl-hospital1`, `fl-hospital2`, `fl-hospital3` — one container per hospital,
  each built from `fl/Dockerfile` with a different `HOSPITAL_NAME` build arg.
  Each image only ever has **that single hospital's CSV** copied into it — so
  hospital2's container structurally cannot contain hospital1's or
  hospital3's data; it was never in that image's filesystem to begin with.
  They sit on an internal `fl-net` Docker network and are never published to
  the host (`expose`, not `ports`) -- nothing outside this compose stack can
  reach a hospital's collaborator directly.
- `backend` joins `fl-net` too, and talks to the three collaborators over
  real **HTTPS with mutual TLS** by container hostname
  (`FL_DEPLOYMENT_MODE=network`, `FL_TLS_ENABLED=1` -- see
  `services/openfl_service.py`, `fl/aggregator.py`, `fl/tls.py`). Every
  collaborator requires callers to present a client certificate signed by
  the same private CA (`scripts/generate_certs.py`) -- traffic is both
  encrypted and authenticated, not just encrypted. It never trains
  locally in this mode, and as of this revision it has **no volume mount to
  backend/data at all** -- it has no filesystem access to any hospital's raw
  data, full stop. It only ever receives dataset bytes over HTTP, from
  `fl/register_dataset_client.py` running inside each hospital's own
  container (step 3 below).
- Training rounds also tolerate real failures now: collaborator calls run
  **concurrently** with a per-client timeout, a collaborator that dies or
  times out is dropped from that round instead of crashing the whole
  experiment, and a round only counts if at least half the collaborators
  responded (configurable via `FL_MIN_QUORUM_FRACTION` /
  `FL_CLIENT_TIMEOUT_SECONDS` on the `backend` service).

## 0) Generate the synthetic data AND the mTLS certs FIRST (before building images)

```bash
cd backend
python3 scripts/generate_synthetic_data.py
python3 scripts/generate_certs.py
```

Both have to happen before `docker-compose build` / `up`, not after:

- Each `fl-hospitalN` image `COPY`s its hospital's CSV in at **build time**
  (`fl/Dockerfile`). If the CSV doesn't exist on the host yet, the build
  fails outright — a fast, obvious failure that is far better than the
  previous version, where dataset generation quietly happened at *runtime*
  inside a container that (as of this revision) doesn't have anywhere to
  persist it to.
- `docker-compose.yml` mounts specific cert files
  (`./certs/hospital1.crt`, etc.) into each service; if `backend/certs/`
  doesn't exist yet, `docker-compose up` fails immediately on a missing
  bind-mount source rather than silently starting without TLS.

`backend/certs/ca.key` is the CA's private key -- it must never be
committed (already covered by `.gitignore`) and never gets copied into any
Docker image; only the public `ca.crt` plus each service's own leaf
cert/key are mounted in.

## 1) Start the stack

```bash
docker-compose up -d --build
```

This builds 4 images: `backend`, and one `fl-hospital{1,2,3}` image each
(each one only containing its own hospital's data — see step 5 to verify).

## 2) Seed institutions + users

```bash
docker-compose exec backend python scripts/seed_data.py
```

In Docker mode this step no longer touches any dataset file — it only
creates institutions/users, then prints the commands for step 3.

## 3) Register each hospital's dataset (run FROM each hospital's own container)

```bash
docker-compose exec fl-hospital1 python3 -m fl.register_dataset_client
docker-compose exec fl-hospital2 python3 -m fl.register_dataset_client
docker-compose exec fl-hospital3 python3 -m fl.register_dataset_client
```

Each of these reads the CSV baked into *that* container's own image and
POSTs the bytes to the backend's `/datasets/register` endpoint. The
backend never reads a local path — this is the same code path it would use
if a genuinely external hospital's system uploaded a file.

- MinIO console: `http://localhost:9001` (user/pass: `minioadmin` / `minioadmin`)

## 4) Run the demo federated training pipeline

```bash
docker-compose exec backend python scripts/run_demo_fl.py
```

Or trigger it via the API and watch it live:

```bash
curl -X POST http://localhost:8000/experiments/start/1
# then open a WebSocket to ws://localhost:8000/experiments/1/ws,
# or poll GET http://localhost:8000/experiments/1/rounds
```

## 5) Verify it's really isolated, not just "a simulation with extra steps"

```bash
docker-compose ps                              # fl-hospital1/2/3 as separate services
docker-compose logs fl-hospital1                # shows only hospital1's own log lines
docker exec -it $(docker-compose ps -q fl-hospital2) ls /app/data
# -> only hospital2/, nothing from hospital1 or hospital3

docker exec -it $(docker-compose ps -q backend) ls /app
# -> no data/ directory at all -- the backend genuinely cannot read
#    hospital data locally anymore, in Docker mode
```

## 5b) Verify mTLS is actually enforced, not just configured

```bash
# Plain curl WITHOUT a client cert should be rejected by a collaborator:
docker exec -it $(docker-compose ps -q backend) \
  curl -k https://fl-hospital1:9101/health
# -> connection error / TLS handshake failure, NOT a JSON response

# With the real client cert, it should succeed:
docker exec -it $(docker-compose ps -q backend) \
  curl --cert /app/certs/aggregator.crt --key /app/certs/aggregator.key \
       --cacert /app/certs/ca.crt https://fl-hospital1:9101/health
# -> {"status": "ok", "hospital": "hospital1", ...}
```

## 6) Verify via API

- Backend health: `http://localhost:8000/docs`
- List datasets: `GET http://localhost:8000/datasets`
- Predict with a trained model: `POST http://localhost:8000/experiments/{id}/predict`
- MLflow: `http://localhost:5001`

## Reset everything (clean slate)

```bash
cd backend
docker-compose down -v
docker-compose up -d --build
```




## Upgrading from an earlier version of this project (IMPORTANT)

The data model changed: **every hospital now holds data for every disease**
(3 hospitals x 3 diseases = 9 datasets), so "request from all eligible
hospitals" can genuinely reach more than one hospital. Existing databases and
images predate that, so do a clean rebuild rather than patching in place:

```bash
cd backend
python3 scripts/generate_synthetic_data.py   # now writes 9 CSVs
python3 scripts/generate_certs.py            # skip if backend/certs/ already exists
docker-compose down -v                       # wipes Postgres: old requests have no project_id
docker-compose up -d --build                 # fl-hospital images embed the data, so they must rebuild
docker-compose exec backend python scripts/seed_data.py
docker-compose exec fl-hospital1 python3 -m fl.register_dataset_client   # registers all 3 diseases
docker-compose exec fl-hospital2 python3 -m fl.register_dataset_client
docker-compose exec fl-hospital3 python3 -m fl.register_dataset_client
```

Why `down -v` rather than relying on the lightweight schema-sync: that sync only
adds missing *columns*. It does not create the new partial unique indexes
(one-active-project, one-pending-request, one-active-block) on tables that
already exist, so only a fresh database gets the race-condition protection.

Run the tests (no database needed):

```bash
python3 tests/test_crypto_and_privacy.py
python3 tests/test_eligibility.py
python3 tests/test_register_client.py
python3 tests/test_fl_end_to_end.py     # spawns real collaborator processes, ~1 min
```
