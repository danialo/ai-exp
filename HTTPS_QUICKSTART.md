# HTTPS Quick Start

Self-signed HTTPS is now configured for Astra.

## Start with HTTPS

```bash
# Default (0.0.0.0:8000)
./start_https.sh

# Custom host/port
./start_https.sh 172.239.66.45 8443
```

## Test Connection

```bash
# Local
curl -k https://localhost:8000/api/health

# Remote (your server IP)
curl -k https://172.239.66.45:8000/api/health
```

**Note**: `-k` flag skips certificate verification (expected for self-signed certs)

## For MCP Sidecar

Update adapter to use HTTPS:

```bash
export ASTRA_API=https://172.239.66.45:8000
```

## Full Documentation

See [docs/HTTPS_SETUP.md](docs/HTTPS_SETUP.md) for:
- Production setup (Let's Encrypt)
- Nginx reverse proxy
- Troubleshooting
- Security best practices

## Certificate Info

- **Valid**: 1 year (Nov 5, 2025 - Nov 5, 2026)
- **Location**: `certs/cert.pem` (public), `certs/key.pem` (private)
- **CN**: 172.239.66.45
