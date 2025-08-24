#!/bin/bash

# DermaSight AI Backend Startup Script
echo "Starting DermaSight AI Backend..."

# Wait for database to be ready
echo "Waiting for database to be ready..."
python -c "
import asyncio
import asyncpg
import os
import time

async def wait_for_db():
    max_attempts = 30
    attempt = 0
    
    db_url = os.getenv('DATABASE_URL', 'postgresql+asyncpg://skinscan:skinscan@db:5432/skinscan')
    # Convert SQLAlchemy URL to asyncpg URL
    db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')
    
    while attempt < max_attempts:
        try:
            conn = await asyncpg.connect(db_url)
            await conn.close()
            print('Database is ready!')
            break
        except Exception as e:
            attempt += 1
            print(f'Database not ready, attempt {attempt}/{max_attempts}: {e}')
            await asyncio.sleep(2)
    
    if attempt >= max_attempts:
        print('Failed to connect to database after maximum attempts')
        exit(1)

asyncio.run(wait_for_db())
"

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