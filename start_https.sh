#!/bin/bash
# Start Astra API with HTTPS using self-signed certificates
#
# Usage:
#   ./start_https.sh                    # Default: 0.0.0.0:8000
#   ./start_https.sh 0.0.0.0 8443       # Custom host and port

HOST=${1:-0.0.0.0}
PORT=${2:-8000}

echo "Starting Astra API with HTTPS..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Certificate: certs/cert.pem"
echo "Private Key: certs/key.pem"
echo ""
echo "Access at: https://$HOST:$PORT"
echo ""

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run uvicorn with SSL
uvicorn app:app \
    --host "$HOST" \
    --port "$PORT" \
    --ssl-keyfile certs/key.pem \
    --ssl-certfile certs/cert.pem \
    --log-level info
