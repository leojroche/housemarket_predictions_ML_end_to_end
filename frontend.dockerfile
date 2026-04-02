FROM python:3.12

RUN pip install uv

WORKDIR /app/

COPY pyproject.toml uv.lock ./

# Installing uv  and python packages
RUN uv sync --frozen
RUN uv pip install streamlit

# Activate local environment 
ENV PATH="/app/.venv/bin:$PATH" 

# Copy necessary python files
COPY src/ src/

ENTRYPOINT ["streamlit", "run", "src/api/frontend.py", "--server.port=6969", "--server.address=0.0.0.0"]

