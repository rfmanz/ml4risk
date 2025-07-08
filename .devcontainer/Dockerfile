# Start from a base image (adjust as needed)
# Using Ubuntu as a base, but you might want to use a Jupyter base image
FROM ubuntu:22.04

# Set environment variables
ARG NODE_MAJOR=22
ARG NB_USER=jovyan
ARG CONDA_DIR=/opt/conda

# Install basic dependencies (including Python and pip)
RUN apt-get update && \
    apt-get install -y \
      curl \
      gnupg \
      ca-certificates \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and Claude Code
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | \
      gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] \
      https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update -y && \
    apt-get install -y nodejs && \
    npm install -g --unsafe-perm @anthropic-ai/claude-code && \
    apt-get purge --auto-remove -y curl gnupg && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /root/.cache

# Install Rich for enhanced Python terminal output
RUN pip3 install rich

# Create a script to fix permissions (if not already available)
RUN echo '#!/bin/bash\n\
for dir in "$@"; do\n\
    if [ -d "$dir" ]; then\n\
        echo "Fixing permissions for $dir"\n\
        find "$dir" -type d -exec chmod 755 {} \;\n\
        find "$dir" -type f -exec chmod 644 {} \;\n\
    fi\n\
done' > /usr/local/bin/fix-permissions && \
    chmod +x /usr/local/bin/fix-permissions

# Fix permissions (adjust paths as needed)
RUN if [ -d "${CONDA_DIR}" ]; then fix-permissions "${CONDA_DIR}"; fi && \
    if [ -d "/home/${NB_USER}" ]; then fix-permissions "/home/${NB_USER}"; fi

# Switch to root user
USER root

# Set working directory (optional)
WORKDIR /workspace

# Default command (optional)
CMD ["/bin/bash"]
