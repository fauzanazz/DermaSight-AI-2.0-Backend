#!/bin/bash

# DermaSight AI Backend Startup Script
echo "Starting DermaSight AI Backend..."

# Wait for database to be ready
echo "Waiting for database to be ready..."
for i in {1..30}; do
    if python -c "
import psycopg2
import os
try:
    # Parse DATABASE_URL
    db_url = os.getenv('DATABASE_URL', 'postgresql+asyncpg://skinscan:skinscan@db:5432/skinscan')
    # Convert to psycopg2 format
    db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')
    
    conn = psycopg2.connect(db_url)
    conn.close()
    print('Database is ready!')
    exit(0)
except Exception as e:
    print(f'Database not ready: {e}')
    exit(1)
"; then
        echo "Database connection successful!"
        break
    else
        echo "Waiting for database... attempt $i/30"
        sleep 2
    fi
done

# Run database initialization
echo "Initializing database..."
python init_db.py || echo "Database initialization completed or skipped"

# Run database migrations
echo "Running database migrations..."
python -m alembic upgrade head

# Check if migrations were successful
if [ $? -eq 0 ]; then
    echo "Database migrations completed successfully"
else
    echo "Database migrations failed, but continuing..."
    # Don't exit - the app can still run with basic functionality
fi

# Seed initial data (optional)
echo "Seeding initial data..."
python seed_data.py || echo "Seeding completed or skipped"

# Start the FastAPI application
echo "Starting FastAPI application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info