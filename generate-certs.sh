#!/bin/bash

# Generate SSL certificates for Jupyter Lab HTTPS
# This script creates self-signed certificates valid for 365 days

echo "🔐 Generating SSL certificates for HTTPS..."

# Remove existing certificates if they exist
if [ -f "mycert.pem" ] || [ -f "mykey.key" ]; then
    echo "⚠️  Existing certificates found. Removing..."
    rm -f mycert.pem mykey.key
fi

# Generate new certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout mykey.key -out mycert.pem \
    -subj "/C=ES/ST=State/L=City/O=TensorFlow-Lab/OU=Development/CN=localhost" \
    2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ SSL certificates generated successfully!"
    echo "   - Certificate: mycert.pem"
    echo "   - Private Key: mykey.key"
    echo "   - Valid for: 365 days"
    echo ""
    echo "🚀 You can now start the container with:"
    echo "   docker compose up --build"
    echo ""
    echo "🌐 Access Jupyter Lab at:"
    echo "   https://127.0.0.1:8888/lab?token=<your-token>"
else
    echo "❌ Failed to generate certificates"
    exit 1
fi
