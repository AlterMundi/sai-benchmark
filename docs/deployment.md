# Deployment Guide

Complete guide for deploying SAI-Benchmark in production environments.

## Table of Contents

- [Production Architecture](#production-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Service Management](#service-management)
- [Monitoring & Observability](#monitoring--observability)
- [Security](#security)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)

## Production Architecture

### Single-Node Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Server                           │
├─────────────────────────────────────────────────────────────────┤
│                    Load Balancer / Reverse Proxy               │
│                        (nginx/haproxy)                         │
├─────────────────────────────────────────────────────────────────┤
│  SAI-Benchmark API    │   Ollama Service   │   Monitoring      │
│  (Flask/FastAPI)      │   (Port 11434)     │   (Prometheus)    │
├─────────────────────────────────────────────────────────────────┤
│                    Shared Storage                              │
│              (Models, Datasets, Results)                       │
├─────────────────────────────────────────────────────────────────┤
│                    GPU Resources                               │
│         (Automatic allocation via ResourceManager)             │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Node Deployment

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └─────────┬───────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
    ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
    │   Node 1      │ │   Node 2      │ │   Node N      │
    │ (API + Ollama)│ │ (API + Ollama)│ │ (API + Ollama)│
    └───────────────┘ └───────────────┘ └───────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                    ┌─────────▼───────┐
                    │ Shared Storage  │
                    │   (NFS/S3)      │
                    └─────────────────┘
```

## Hardware Requirements

### Production Server Specifications

#### Minimum Requirements
- **CPU**: 16 cores (Intel Xeon or AMD EPYC)
- **RAM**: 64GB DDR4
- **GPU**: 1x NVIDIA RTX 4090 (24GB VRAM)
- **Storage**: 1TB NVMe SSD
- **Network**: 10Gbps Ethernet

#### Recommended Requirements
- **CPU**: 32+ cores (Intel Xeon Gold/Platinum)
- **RAM**: 128GB+ DDR4/DDR5
- **GPU**: 2x NVIDIA A100 (80GB VRAM each)
- **Storage**: 4TB NVMe SSD + 10TB HDD (datasets)
- **Network**: 25Gbps+ Ethernet

#### High-Throughput Setup
- **CPU**: 64+ cores
- **RAM**: 256GB+
- **GPU**: 4x NVIDIA H100 (80GB VRAM each)
- **Storage**: 8TB NVMe SSD + 50TB distributed storage
- **Network**: 100Gbps Infiniband

### Storage Layout

```
/opt/sai-benchmark/
├── app/                    # Application code
├── models/                 # Model cache (500GB+)
├── datasets/              # Test datasets (1TB+)
├── results/               # Benchmark results
├── logs/                  # Application logs
└── config/                # Configuration files

/var/lib/ollama/           # Ollama model storage
├── models/                # Quantized models (200GB+)
└── tmp/                   # Temporary files
```

## Installation & Setup

### 1. System Prerequisites

```bash
# Ubuntu 22.04 LTS
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev \
    nvidia-driver-535 nvidia-cuda-toolkit \
    docker.io docker-compose \
    nginx redis-server postgresql-14 \
    htop nvtop iotop \
    git curl wget

# Enable Docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

### 2. NVIDIA Container Runtime

```bash
# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-runtime

# Configure Docker
sudo systemctl restart docker
```

### 3. Application Deployment

```bash
# Create application user
sudo useradd -m -s /bin/bash sai-benchmark
sudo mkdir -p /opt/sai-benchmark
sudo chown sai-benchmark:sai-benchmark /opt/sai-benchmark

# Switch to application user
sudo su - sai-benchmark
cd /opt/sai-benchmark

# Clone and setup
git clone https://github.com/AlterMundi/sai-benchmark.git app
cd app

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Create directories
mkdir -p ../models ../datasets ../results ../logs ../config
```

### 4. Ollama Installation

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Configure as service
sudo systemctl enable ollama
sudo systemctl start ollama

# Pull production models
ollama pull qwen2.5-vl:7b
ollama pull llama3.2-vision:11b
ollama pull minicpm-v:8b

# Verify installation
ollama list
```

## Configuration

### 1. Production Configuration

Create `/opt/sai-benchmark/config/production.yaml`:

```yaml
# Production SAI-Benchmark Configuration
environment: production
debug: false

# Resource Management
resource_manager:
  gpu_memory_limit: 80.0        # GB (total across all GPUs)
  cpu_core_limit: 32            # CPU cores
  system_memory_limit: 128.0    # GB
  max_concurrent_models: 4      # Simultaneous model instances
  allocation_timeout: 120       # Seconds
  cleanup_interval: 300         # Seconds

# Engine Configuration
engines:
  ollama:
    base_url: "http://localhost:11434"
    timeout: 300
    max_retries: 3
    health_check_interval: 60
  
  huggingface:
    device: "auto"
    torch_dtype: "float16"
    cache_dir: "/opt/sai-benchmark/models"
    max_memory_per_gpu: 0.9
  
  # TODO: Configure OpenAI engine when implemented
  # openai:
  #   api_key: "${OPENAI_API_KEY}"
  #   max_concurrent_requests: 50
  #   rate_limit_per_minute: 500

# Storage Configuration
storage:
  models_dir: "/opt/sai-benchmark/models"
  datasets_dir: "/opt/sai-benchmark/datasets"
  results_dir: "/opt/sai-benchmark/results"
  logs_dir: "/opt/sai-benchmark/logs"
  temp_dir: "/tmp/sai-benchmark"

# Logging Configuration
logging:
  level: "INFO"
  format: "json"
  max_file_size: "100MB"
  backup_count: 10
  handlers:
    - type: "file"
      filename: "/opt/sai-benchmark/logs/application.log"
    - type: "syslog"
      facility: "local0"

# Security Configuration
security:
  api_key_required: true
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_size: 20
  cors:
    enabled: true
    allowed_origins: ["https://yourdomain.com"]

# Performance Configuration
performance:
  max_workers: 8
  worker_timeout: 1800          # 30 minutes
  result_cache_ttl: 3600        # 1 hour
  enable_metrics_collection: true
```

### 2. Environment Variables

Create `/opt/sai-benchmark/config/.env`:

```bash
# Production Environment Variables
export SAI_ENV=production
export SAI_CONFIG_PATH=/opt/sai-benchmark/config/production.yaml
export SAI_LOG_LEVEL=INFO

# API Configuration
export SAI_API_HOST=0.0.0.0
export SAI_API_PORT=8080
export SAI_API_KEY=your-secure-api-key-here

# Database Configuration (for future use)
export SAI_DB_HOST=localhost
export SAI_DB_PORT=5432
export SAI_DB_NAME=sai_benchmark
export SAI_DB_USER=sai_benchmark
export SAI_DB_PASSWORD=secure-password-here

# External Services
export OLLAMA_HOST=http://localhost:11434
export HF_HOME=/opt/sai-benchmark/models
export HF_TOKEN=your-huggingface-token

# TODO: Add when OpenAI engine is implemented
# export OPENAI_API_KEY=your-openai-key
# export ANTHROPIC_API_KEY=your-anthropic-key

# Monitoring
export PROMETHEUS_ENDPOINT=http://localhost:9090
export GRAFANA_URL=http://localhost:3000
```

## Service Management

### 1. Systemd Service Configuration

Create `/etc/systemd/system/sai-benchmark-api.service`:

```ini
[Unit]
Description=SAI-Benchmark API Server
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=sai-benchmark
Group=sai-benchmark
WorkingDirectory=/opt/sai-benchmark/app
Environment=SAI_ENV=production
EnvironmentFile=/opt/sai-benchmark/config/.env
ExecStart=/opt/sai-benchmark/app/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sai-benchmark-api

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/sai-benchmark-worker.service`:

```ini
[Unit]
Description=SAI-Benchmark Background Worker
After=network.target sai-benchmark-api.service
Requires=sai-benchmark-api.service

[Service]
Type=simple
User=sai-benchmark
Group=sai-benchmark
WorkingDirectory=/opt/sai-benchmark/app
Environment=SAI_ENV=production
EnvironmentFile=/opt/sai-benchmark/config/.env
ExecStart=/opt/sai-benchmark/app/venv/bin/python -m celery worker -A api.worker --loglevel=info
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sai-benchmark-worker

[Install]
WantedBy=multi-user.target
```

### 2. Service Management Commands

```bash
# Enable and start services
sudo systemctl enable sai-benchmark-api.service
sudo systemctl enable sai-benchmark-worker.service
sudo systemctl start sai-benchmark-api.service
sudo systemctl start sai-benchmark-worker.service

# Check status
sudo systemctl status sai-benchmark-api
sudo systemctl status sai-benchmark-worker

# View logs
sudo journalctl -u sai-benchmark-api -f
sudo journalctl -u sai-benchmark-worker -f

# Restart services
sudo systemctl restart sai-benchmark-api
sudo systemctl restart sai-benchmark-worker
```

### 3. Nginx Reverse Proxy

Create `/etc/nginx/sites-available/sai-benchmark`:

```nginx
upstream sai_benchmark_backend {
    server 127.0.0.1:8080;
    # Add more servers for load balancing
    # server 127.0.0.1:8081;
    # server 127.0.0.1:8082;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Client body size (for large image uploads)
    client_max_body_size 100M;
    
    # Timeouts
    proxy_connect_timeout 60s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    
    location / {
        proxy_pass http://sai_benchmark_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (for future real-time features)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://sai_benchmark_backend/health;
        access_log off;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/sai-benchmark/app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/sai-benchmark /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Monitoring & Observability

### 1. Prometheus Configuration

Create `/opt/monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "sai_benchmark_rules.yml"

scrape_configs:
  - job_name: 'sai-benchmark'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'ollama'
    static_configs:
      - targets: ['localhost:11434']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
      
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9445']
```

### 2. Key Metrics to Monitor

```yaml
# /opt/monitoring/sai_benchmark_rules.yml
groups:
  - name: sai_benchmark_alerts
    rules:
      - alert: HighGPUMemoryUsage
        expr: gpu_memory_used_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage is high"
          
      - alert: ModelLoadingTimeout
        expr: model_loading_duration_seconds > 300
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model loading is taking too long"
          
      - alert: HighBenchmarkFailureRate
        expr: benchmark_failure_rate > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Benchmark failure rate is high"
          
      - alert: APIResponseTimeHigh
        expr: http_request_duration_seconds{quantile="0.95"} > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API response time is high"
```

### 3. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "SAI-Benchmark Production Dashboard",
    "panels": [
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {"expr": "gpu_utilization_percent"}
        ]
      },
      {
        "title": "Model Inference Latency",
        "type": "graph", 
        "targets": [
          {"expr": "benchmark_inference_duration_seconds"}
        ]
      },
      {
        "title": "Benchmark Success Rate",
        "type": "stat",
        "targets": [
          {"expr": "rate(benchmark_success_total[5m]) / rate(benchmark_total[5m])"}
        ]
      },
      {
        "title": "Active Models",
        "type": "stat",
        "targets": [
          {"expr": "models_loaded_count"}
        ]
      }
    ]
  }
}
```

### 4. Log Aggregation

```yaml
# /opt/monitoring/filebeat.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /opt/sai-benchmark/logs/*.log
    fields:
      service: sai-benchmark
      environment: production
    multiline.pattern: '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
    multiline.negate: true
    multiline.match: after

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "sai-benchmark-%{+yyyy.MM.dd}"

processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
```

## Security

### 1. Network Security

```bash
# Firewall configuration (UFW)
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (adjust port as needed)
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow internal services (restrict to local network)
sudo ufw allow from 10.0.0.0/8 to any port 8080  # API
sudo ufw allow from 10.0.0.0/8 to any port 11434 # Ollama
sudo ufw allow from 10.0.0.0/8 to any port 9090  # Prometheus
sudo ufw allow from 10.0.0.0/8 to any port 3000  # Grafana

# Check status
sudo ufw status verbose
```

### 2. Application Security

```bash
# Create secure API keys
export SAI_API_KEY=$(openssl rand -hex 32)
echo "SAI_API_KEY=$SAI_API_KEY" >> /opt/sai-benchmark/config/.env

# Set proper file permissions
sudo chown -R sai-benchmark:sai-benchmark /opt/sai-benchmark
sudo chmod 755 /opt/sai-benchmark
sudo chmod 644 /opt/sai-benchmark/config/.env
sudo chmod 600 /opt/sai-benchmark/config/production.yaml

# Secure model storage
sudo chmod 755 /opt/sai-benchmark/models
sudo chmod 644 /opt/sai-benchmark/models/*
```

### 3. Container Security (Docker deployment)

```dockerfile
# Secure Dockerfile example
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r sai && useradd -r -g sai sai

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application
COPY --chown=sai:sai . /app
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER sai

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Scaling

### 1. Horizontal Scaling

```yaml
# docker-compose.yml for multi-instance deployment
version: '3.8'
services:
  sai-benchmark-1:
    build: .
    ports:
      - "8080:8080"
    environment:
      - SAI_INSTANCE_ID=1
    volumes:
      - models:/app/models
      - datasets:/app/datasets
      - results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  sai-benchmark-2:
    build: .
    ports:
      - "8081:8080"
    environment:
      - SAI_INSTANCE_ID=2
    volumes:
      - models:/app/models
      - datasets:/app/datasets
      - results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - sai-benchmark-1
      - sai-benchmark-2

volumes:
  models:
  datasets:
  results:
```

### 2. Auto-scaling Configuration

```yaml
# Kubernetes HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sai-benchmark-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sai-benchmark
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: gpu_utilization
        target:
          type: AverageValue
          averageValue: "75"
```

## Troubleshooting

### 1. Common Issues

#### GPU Memory Issues
```bash
# Check GPU status
nvidia-smi

# Clear GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <process_ids>

# Restart Ollama
sudo systemctl restart ollama
```

#### Model Loading Failures
```bash
# Check Ollama status
ollama list
ollama ps

# Re-pull problematic model
ollama rm qwen2.5-vl:7b
ollama pull qwen2.5-vl:7b

# Check logs
sudo journalctl -u ollama -f
```

#### Performance Issues
```bash
# Check system resources
htop
iotop
nvtop

# Check application logs
tail -f /opt/sai-benchmark/logs/application.log

# Monitor metrics
curl http://localhost:8080/metrics
```

### 2. Diagnostic Commands

```bash
# System health check
/opt/sai-benchmark/app/venv/bin/python -c "
from core.resource_manager import ResourceManager
rm = ResourceManager()
print(rm.get_resource_stats())
"

# Model availability check
/opt/sai-benchmark/app/venv/bin/python -c "
from core.model_registry import model_registry
for model in model_registry.list_models():
    print(f'{model.id}: {model.name}')
"

# Engine health check
/opt/sai-benchmark/app/venv/bin/python -c "
from core.engine_registry import engine_registry
for engine_type in ['ollama', 'huggingface']:
    health = engine_registry.health_check(engine_type)
    print(f'{engine_type}: {health}')
"
```

### 3. Performance Tuning

```bash
# Optimize GPU memory allocation
echo 'CUDA_VISIBLE_DEVICES=0,1' >> /opt/sai-benchmark/config/.env
echo 'CUDA_MEMORY_FRACTION=0.9' >> /opt/sai-benchmark/config/.env

# Optimize CPU threading
echo 'OMP_NUM_THREADS=16' >> /opt/sai-benchmark/config/.env
echo 'MKL_NUM_THREADS=16' >> /opt/sai-benchmark/config/.env

# Tune kernel parameters
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_ratio=15' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Backup & Recovery

### 1. Backup Strategy

```bash
#!/bin/bash
# /opt/sai-benchmark/scripts/backup.sh

BACKUP_DIR="/opt/backups/sai-benchmark"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup configuration
tar -czf $BACKUP_DIR/$DATE/config.tar.gz /opt/sai-benchmark/config/

# Backup results (last 30 days)
find /opt/sai-benchmark/results -mtime -30 -type f | \
    tar -czf $BACKUP_DIR/$DATE/recent_results.tar.gz -T -

# Backup database (when implemented)
# pg_dump sai_benchmark > $BACKUP_DIR/$DATE/database.sql

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -type d -mtime +30 -exec rm -rf {} \;

# TODO: Add S3/cloud backup when cloud storage is configured
```

### 2. Recovery Procedures

```bash
# Restore configuration
tar -xzf /opt/backups/sai-benchmark/20241215_120000/config.tar.gz -C /

# Restore results
tar -xzf /opt/backups/sai-benchmark/20241215_120000/recent_results.tar.gz -C /

# Restart services
sudo systemctl restart sai-benchmark-api
sudo systemctl restart sai-benchmark-worker
```

---

## TODO Items

- [ ] **OpenAI/Anthropic Engine Implementation**: Complete missing engine implementations
- [ ] **Database Integration**: Add PostgreSQL for result persistence and querying
- [ ] **Cloud Storage**: Implement S3/Azure Blob integration for model and dataset storage
- [ ] **Kubernetes Deployment**: Create Helm charts for K8s deployment
- [ ] **Real-time Monitoring**: Implement WebSocket endpoints for real-time monitoring
- [ ] **CI/CD Pipeline**: Add automated deployment pipelines
- [ ] **Disaster Recovery**: Implement cross-region backup and recovery
- [ ] **API Versioning**: Add versioned API endpoints for backward compatibility
- [ ] **Multi-tenant Support**: Add tenant isolation for shared deployments
- [ ] **Cost Optimization**: Implement cost tracking and optimization recommendations

This deployment guide provides a comprehensive foundation for production deployment while highlighting areas that need future development.