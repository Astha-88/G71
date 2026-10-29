# One image per hospital, built with a different HOSPITAL_NAME build arg
# (see docker-compose.yml's fl-hospital1/2/3 services). Only that single
# hospital's CSV is copied in below -- inspect any of these images and
# you will find exactly one hospital's data and no others, which is a
# stronger isolation guarantee than "the aggregator process just doesn't
# read the other files."
FROM python:3.11-slim

ARG HOSPITAL_NAME
ENV HOSPITAL_NAME=${HOSPITAL_NAME}
ENV PORT=9101
ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir numpy==2.4.2 requests==2.32.3 cryptography==43.0.3

COPY fl/ fl/
COPY data/${HOSPITAL_NAME}/ data/${HOSPITAL_NAME}/

EXPOSE 9101

CMD ["python3", "-m", "fl.collaborator_server"]
