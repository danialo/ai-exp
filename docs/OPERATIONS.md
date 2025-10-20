# Operations Guide

This guide covers operational aspects of the AI Experience Memory System including deployment, monitoring, maintenance, and performance tuning.

## System Requirements

### Minimum Requirements

- Python 3.11+
- 2GB RAM (for sentence transformers model)
- 500MB disk space (plus storage for experiences)
- OpenAI API access (for LLM responses)

### Recommended Requirements

- Python 3.12+
- 4GB RAM
- SSD storage for database and vector index
- Stable internet connection for OpenAI API

## Deployment

### Local Development

```bash
# Clone and setup
git clone <repository>
cd ai-exp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your settings

# Initialize
python scripts/init_db.py

# Run
python app.py
```

### Production Deployment

#### Using systemd (Linux)

Create `/etc/systemd/system/ai-exp.service`:

```ini
[Unit]
Description=AI Experience Memory System
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/ai-exp
Environment="PATH=/path/to/ai-exp/venv/bin"
ExecStart=/path/to/ai-exp/venv/bin/python app.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-exp
sudo systemctl start ai-exp
sudo systemctl status ai-exp
```

#### Using Docker (Future)

_Docker support not yet implemented. See roadmap._

#### Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Configuration Tuning

### Retrieval Performance

**Semantic vs Recency Balance:**

```bash
# More emphasis on semantic similarity (better for topic clustering)
SEMANTIC_WEIGHT=0.9
RECENCY_WEIGHT=0.1

# More emphasis on recency (better for chronological context)
SEMANTIC_WEIGHT=0.6
RECENCY_WEIGHT=0.4

# Balanced (default)
SEMANTIC_WEIGHT=0.8
RECENCY_WEIGHT=0.2
```

**Number of Memories to Retrieve:**

```bash
# Fewer memories = faster, more focused responses
TOP_K_RETRIEVAL=3

# More memories = slower, richer context (may hit token limits)
TOP_K_RETRIEVAL=10

# Default balanced
TOP_K_RETRIEVAL=5
```

### LLM Settings

**Temperature:**

```bash
# More deterministic, focused responses
LLM_TEMPERATURE=0.3

# More creative, varied responses
LLM_TEMPERATURE=1.0

# Balanced (default)
LLM_TEMPERATURE=0.7
```

**Token Limits:**

```bash
# Shorter responses (faster, cheaper)
LLM_MAX_TOKENS=250

# Longer responses (more detailed)
LLM_MAX_TOKENS=1000

# Default
LLM_MAX_TOKENS=500
```

### Affect Sensitivity

**Valence threshold for empathetic tone:**

In `src/pipeline/lens.py`, adjust `valence_threshold`:

```python
# More sensitive (empathy for mild negativity)
lens = create_experience_lens(
    llm_service,
    retrieval_service,
    valence_threshold=-0.1
)

# Less sensitive (empathy only for strong negativity)
lens = create_experience_lens(
    llm_service,
    retrieval_service,
    valence_threshold=-0.5
)

# Default
valence_threshold=-0.2
```

## Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# System statistics
curl http://localhost:8000/api/stats
```

Response example:

```json
{
  "total_experiences": 42,
  "total_vectors": 84,
  "llm_model": "gpt-3.5-turbo",
  "llm_enabled": true
}
```

### Database Monitoring

```bash
# Check database size
ls -lh data/raw_store.db

# Count experiences by type
sqlite3 data/raw_store.db "SELECT type, COUNT(*) FROM experience GROUP BY type"

# Recent experiences
sqlite3 data/raw_store.db "SELECT id, created_at, type FROM experience ORDER BY created_at DESC LIMIT 10"
```

### Vector Index Monitoring

```bash
# Check vector index size
du -sh data/vector_index/

# Count vectors in index (requires Python)
python -c "
from src.memory.vector_store import create_vector_store
from config.settings import settings
vs = create_vector_store(settings.VECTOR_INDEX_PATH)
print(f'Total vectors: {vs.count()}')
"
```

### Log Monitoring

Application logs are output to stdout/stderr. For systemd deployments:

```bash
# View logs
sudo journalctl -u ai-exp -f

# Filter by priority
sudo journalctl -u ai-exp -p err

# Last 100 lines
sudo journalctl -u ai-exp -n 100
```

## Maintenance

### Database Backup

```bash
# Backup database
cp data/raw_store.db data/backups/raw_store_$(date +%Y%m%d).db

# Backup vector index
tar -czf data/backups/vector_index_$(date +%Y%m%d).tar.gz data/vector_index/
```

### Database Vacuum

SQLite databases can become fragmented over time:

```bash
sqlite3 data/raw_store.db "VACUUM;"
```

### Vector Index Rebuild

If the vector index becomes corrupted or needs optimization:

```bash
# Backup current index
mv data/vector_index data/vector_index.bak

# Rebuild from experiences
python scripts/rebuild_vector_index.py  # (script not yet implemented)
```

### Cleanup Old Reflections

Reflection observations can accumulate. To prune old reflections:

```python
# Python script to delete reflections older than N days
from datetime import datetime, timedelta, timezone
from src.memory.raw_store import create_raw_store
from config.settings import settings

raw_store = create_raw_store(settings.RAW_STORE_DB_PATH)
cutoff = datetime.now(timezone.utc) - timedelta(days=90)

# Note: This requires implementing a delete method (currently stubbed)
# raw_store.delete_experiences_before(cutoff, type="observation")
```

## Performance Optimization

### Embedding Cache

The sentence transformer model is loaded once on startup. To reduce memory:

- Use a smaller model: `sentence-transformers/all-MiniLM-L6-v2` (default, 384 dims)
- Or larger for better quality: `sentence-transformers/all-mpnet-base-v2` (768 dims)

### Vector Index Performance

ChromaDB performance tuning:

1. **Increase batch size** for bulk inserts
2. **Use SSD storage** for the vector index
3. **Consider HNSW indexing** for large datasets (not yet implemented)

### Database Performance

SQLite optimization:

```sql
-- Enable WAL mode (already done in code)
PRAGMA journal_mode=WAL;

-- Increase cache size
PRAGMA cache_size=10000;

-- Synchronous mode for better performance (use with caution)
PRAGMA synchronous=NORMAL;
```

## Troubleshooting

### High Memory Usage

**Symptoms:** Process using >2GB RAM

**Causes:**
- Large sentence transformer model loaded
- Many concurrent requests
- Vector index in memory

**Solutions:**
1. Use a smaller embedding model
2. Restart the application periodically
3. Limit concurrent requests with a queue

### Slow Retrieval

**Symptoms:** `/api/chat` endpoint taking >3 seconds

**Causes:**
- Large vector index
- High `TOP_K_RETRIEVAL` value
- Slow embedding generation

**Solutions:**
1. Reduce `TOP_K_RETRIEVAL`
2. Optimize `SEMANTIC_WEIGHT` and `RECENCY_WEIGHT`
3. Profile with `pytest --profile` to identify bottlenecks

### OpenAI API Errors

**Rate Limits:**

```python
# In src/services/llm.py, add retry logic
from openai import RateLimitError
import time

try:
    response = self.client.chat.completions.create(...)
except RateLimitError:
    time.sleep(60)  # Wait and retry
    response = self.client.chat.completions.create(...)
```

**Token Limits:**

Reduce `LLM_MAX_TOKENS` or limit memory context:

```bash
TOP_K_RETRIEVAL=2
LLM_MAX_TOKENS=300
```

### Database Lock Errors

**Symptoms:** `sqlite3.OperationalError: database is locked`

**Cause:** Multiple processes writing simultaneously

**Solution:**
- Stop web server before running CLI scripts that write
- Or use a separate database for testing
- Or implement connection pooling with retry logic

### Vector Index Corruption

**Symptoms:** Errors on retrieval, inconsistent results

**Solution:**

```bash
# Backup and delete corrupted index
mv data/vector_index data/vector_index.corrupted

# Reinitialize
python scripts/init_db.py

# Rebuild from experiences (requires custom script)
```

## Security Considerations

### API Key Protection

- Never commit `.env` to version control
- Use environment variables in production
- Rotate API keys periodically
- Monitor API usage for anomalies

### Data Privacy

- Experiences contain user interactions
- No built-in encryption (consider full-disk encryption)
- No multi-user isolation (single-user system)
- Consider GDPR/privacy implications for production use

### Network Security

- Use HTTPS in production (nginx with Let's Encrypt)
- Implement rate limiting
- Add authentication for API endpoints (not yet implemented)

## Scaling Considerations

### Current Limitations

- Single-process architecture
- SQLite (not suitable for high concurrency)
- In-memory vector index
- No horizontal scaling

### Future Scaling Options

1. **PostgreSQL** instead of SQLite for raw store
2. **Dedicated vector DB** (Pinecone, Weaviate, Milvus)
3. **Redis caching** for frequently accessed experiences
4. **Async processing** with Celery for ingestion
5. **Load balancing** with multiple app instances
6. **Sharding** by user or time period

## Backup & Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh - Daily backup script

DATE=$(date +%Y%m%d)
BACKUP_DIR=/path/to/backups

# Database
cp data/raw_store.db $BACKUP_DIR/raw_store_$DATE.db

# Vector index
tar -czf $BACKUP_DIR/vector_index_$DATE.tar.gz data/vector_index/

# Rotate old backups (keep 30 days)
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

Schedule with cron:

```bash
0 2 * * * /path/to/backup.sh
```

### Recovery

```bash
# Stop service
sudo systemctl stop ai-exp

# Restore database
cp /path/to/backups/raw_store_20251019.db data/raw_store.db

# Restore vector index
tar -xzf /path/to/backups/vector_index_20251019.tar.gz -C data/

# Start service
sudo systemctl start ai-exp
```

## Metrics & Analytics

### Key Metrics to Track

1. **Experience count** (total, by type)
2. **Vector count**
3. **Average retrieval latency**
4. **LLM API latency**
5. **Memory usage**
6. **Disk usage**
7. **API request rate**
8. **Error rate**

### Example Monitoring Script

```python
# scripts/metrics.py
from src.memory.raw_store import create_raw_store
from src.memory.vector_store import create_vector_store
from config.settings import settings
import json

raw_store = create_raw_store(settings.RAW_STORE_DB_PATH)
vector_store = create_vector_store(settings.VECTOR_INDEX_PATH)

metrics = {
    "experiences": raw_store.count_experiences(),
    "vectors": vector_store.count(),
    "db_size_mb": os.path.getsize(settings.RAW_STORE_DB_PATH) / 1024 / 1024,
}

print(json.dumps(metrics, indent=2))
```

Run periodically and send to monitoring system (Prometheus, Grafana, etc.).

## Support

For issues, questions, or contributions:

- GitHub Issues: <repository-url>/issues
- Documentation: `docs/`
- Architecture: `docs/experience_schema.md`
- Build Plan: `docs/mvp_build_plan.md`
