FROM ollama/ollama:latest

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install packages
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn

# Clone Open WebUI
RUN git clone https://github.com/open-webui/open-webui.git /root/open-webui
WORKDIR /root/open-webui
RUN pip install -r requirements.txt

# Copy Modelfile
WORKDIR /root
COPY Modelfile /root/

# Create startup script
RUN echo '#!/bin/bash\n\
if [ -f "/root/model.gguf" ]; then\n\
    echo "Model file found, creating Ollama model..."\n\
    ollama create custom-model -f Modelfile\n\
else\n\
    echo "Model file not found at /root/model.gguf"\n\
    echo "Please mount the model file when running the container"\n\
fi\n\
ollama serve & cd /root/open-webui && python main.py' > /root/start.sh

RUN chmod +x /root/start.sh

# Expose ports
EXPOSE 11434
EXPOSE 8080

# Start services
CMD ["/root/start.sh"]