FROM python:3.12

RUN pip install uv

WORKDIR /app/

COPY pyproject.toml uv.lock ./

# Installing uv  and python packages
RUN uv sync --frozen
# Activate local environment 
ENV PATH="/app/.venv/bin:$PATH" 

# Copy necessary directories
COPY models/ models/
COPY src/ src/

CMD ["uvicorn", "src.api.backend:app", "--host", "0.0.0.0", "--port", "80"]
