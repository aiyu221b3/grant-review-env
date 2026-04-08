# Use Python 3.10 as the base
FROM python:3.10-slim

# Create a non-root user for Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements/pyproject files
COPY --chown=user:user requirements.txt* .
COPY --chown=user:user pyproject.toml* .
COPY --chown=user:user uv.lock* .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt || true
# If you have a pyproject.toml, this ensures it's installed
RUN pip install --no-cache-dir . || true

# Copy the rest of the project files
COPY --chown=user:user . .

# Hugging Face Spaces look for port 7860
EXPOSE 7860

# The critical fix: Point to server/app.py and use port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]