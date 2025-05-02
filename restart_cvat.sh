#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e


# Stop the Docker services
echo "Stopping CVAT Docker services..."
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down

# Start the Docker services
echo "Starting CVAT Docker services..."
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

echo "Process completed successfully."
exit 0
