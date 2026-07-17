FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN addgroup --system verilogic && adduser --system --ingroup verilogic verilogic

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

USER verilogic

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "verilogic_ns_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
