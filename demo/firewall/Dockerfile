# ONNX runtime image: no torch, scipy or scikit-learn. Code and model files are
# bind-mounted read-only per service by compose.yaml.
FROM python:3.12-slim
RUN pip install --no-cache-dir onnxruntime==1.30.0 transformers==4.57.6 \
        tokenizers==0.22.2 numpy==2.4.6 pyyaml==6.0.3
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HOME=/tmp/hf \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1
WORKDIR /app
