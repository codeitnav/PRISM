FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

# docker-compose.yml doesn't request GPU passthrough (no
# `deploy.resources.reservations.devices`), so this image never has CUDA
# available at runtime regardless of the host. Installing torch/torchvision
# from the CPU-only wheel index avoids pulling ~5GB of unused NVIDIA CUDA
# packages that plain `pip install torch` resolves by default - see the
# comment in requirements.txt for how this was confirmed. If GPU passthrough
# is added later, drop the --index-url line and add the `deploy` stanza to
# docker-compose.yml instead.
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
