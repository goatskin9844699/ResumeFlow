# Use the same base image as devcontainer
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Set working directory
WORKDIR /app

# Copy only packages files first to leverage cache
COPY packages.txt .

# Install system packages if packages.txt exists
RUN if [ -f packages.txt ]; then \
        apt-get update && \
        apt-get upgrade -y && \
        xargs apt-get install -y < packages.txt && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Install Streamlit separately
RUN pip3 install --no-cache-dir streamlit

# Install Playwright browsers and dependencies
RUN playwright install chromium firefox webkit
RUN playwright install-deps

COPY resources/requirements.txt .
# Install Python requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Set the entrypoint to run Streamlit
ENTRYPOINT ["streamlit", "run", "web_app.py", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"] 