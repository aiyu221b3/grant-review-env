# Use Python 3.10 as the base
FROM python:3.10-slim

# Create a non-root user for Hugging Face (Security Best Practice)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements first (this makes building faster)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the project files
COPY --chown=user:user . .

# Hugging Face Spaces look for port 7860
EXPOSE 7860

# This starts the environment server
ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]