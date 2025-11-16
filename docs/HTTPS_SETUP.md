# HTTPS Setup for Astra

Quick guide to running Astra API with HTTPS encryption.

## Self-Signed Certificate (Current Setup)

The self-signed certificate provides encryption between endpoints without browser trust.

### Files Created
```
certs/
  cert.pem    # Public certificate (1 year validity)
  key.pem     # Private key (gitignored)
```

### Quick Start

```bash
# Default: Listen on 0.0.0.0:8000
./start_https.sh

# Custom host and port
./start_https.sh 172.239.66.45 8443

# Or manually with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 \
    --ssl-keyfile certs/key.pem \
    --ssl-certfile certs/cert.pem
```

### Access the API

```bash
# From the server itself
curl -k https://localhost:8000/api/health

# From remote (your IP: 172.239.66.45)
curl -k https://172.239.66.45:8000/api/health

# In Python (requests)
import requests
requests.get('https://172.239.66.45:8000/api/health', verify=False)
```

**Note**: `-k` flag (curl) or `verify=False` (requests) skips certificate verification since it's self-signed.

### Certificate Details

```bash
# View certificate info
openssl x509 -in certs/cert.pem -text -noout

# Check expiration
openssl x509 -in certs/cert.pem -noout -dates
```

**Expiration**: 1 year from creation (2025-11-05 to 2026-11-05)

### For MCP Sidecar

Update MCP adapter to use HTTPS:

```python
# mcp_sidecar/adapters/astra_ro.py
class AstraRO:
    def __init__(self, base="https://172.239.66.45:8000", timeout=2.0):
        self.client = httpx.Client(
            base_url=base,
            timeout=timeout,
            verify=False  # Skip cert verification for self-signed
        )
```

Or set environment variable:
```bash
export ASTRA_API=https://172.239.66.45:8000
```

## Upgrading to Production (Let's Encrypt)

When ready for production with your domain:

### Prerequisites
- Domain name pointing to 172.239.66.45
- Port 80 open (for Let's Encrypt validation)
- Port 443 open (for HTTPS)

### Quick Setup

```bash
# Install certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Get certificate (replace your-domain.com)
sudo certbot certonly --standalone -d your-domain.com

# Certificates will be in:
# /etc/letsencrypt/live/your-domain.com/fullchain.pem
# /etc/letsencrypt/live/your-domain.com/privkey.pem

# Update start script to use Let's Encrypt certs
uvicorn app:app --host 0.0.0.0 --port 443 \
    --ssl-keyfile /etc/letsencrypt/live/your-domain.com/privkey.pem \
    --ssl-certfile /etc/letsencrypt/live/your-domain.com/fullchain.pem
```

### Auto-Renewal

Let's Encrypt certs expire every 90 days. Set up auto-renewal:

```bash
# Test renewal
sudo certbot renew --dry-run

# Add to crontab (runs twice daily)
sudo crontab -e
# Add this line:
0 0,12 * * * certbot renew --quiet --post-hook "systemctl restart astra"
```

## Nginx Reverse Proxy (Optional)

For production-grade setup with caching and rate limiting:

```nginx
# /etc/nginx/sites-available/astra
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Modern SSL config
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Proxy to Astra
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/astra /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Troubleshooting

### "Connection refused"
- Check if app is running: `ps aux | grep uvicorn`
- Check port binding: `netstat -tlnp | grep 8000`
- Check firewall: `sudo ufw status`

### "SSL certificate problem"
- Expected with self-signed certs
- Use `-k` (curl) or `verify=False` (Python)
- For production, use Let's Encrypt

### "Permission denied" on port 443
- Ports < 1024 require root
- Solution 1: Use port 8443 instead
- Solution 2: Run with sudo (not recommended)
- Solution 3: Use nginx reverse proxy

### Certificate expired
```bash
# Regenerate (self-signed)
openssl req -x509 -newkey rsa:4096 -nodes \
    -out certs/cert.pem -keyout certs/key.pem -days 365 \
    -subj "/C=US/ST=State/L=City/O=Astra/OU=Dev/CN=172.239.66.45"

# Renew (Let's Encrypt)
sudo certbot renew
```

## Security Notes

### Self-Signed (Current)
- ✅ Encrypts traffic between endpoints
- ✅ Prevents eavesdropping
- ❌ No identity verification (browser warnings)
- ❌ Vulnerable to MITM if attacker controls network

### Let's Encrypt (Production)
- ✅ Encrypts traffic
- ✅ Browser-trusted certificates
- ✅ Identity verification
- ✅ Free and auto-renewable

### Best Practices
1. **Never commit private keys** (already in .gitignore)
2. **Rotate certificates** before expiration
3. **Use strong ciphers** (TLSv1.2+)
4. **Enable HSTS** in production
5. **Monitor certificate expiry**

## Testing

```bash
# Test HTTPS endpoint
curl -k https://172.239.66.45:8000/api/health

# Test SSL/TLS handshake
openssl s_client -connect 172.239.66.45:8000 -showcerts

# Test from Python
python -c "import requests; print(requests.get('https://172.239.66.45:8000/api/health', verify=False).json())"
```

## Environment Variables

Update `.env` for HTTPS:

```bash
# Use HTTPS URLs
ASTRA_API=https://172.239.66.45:8000

# For MCP sidecar
export ASTRA_API=https://172.239.66.45:8000
```
