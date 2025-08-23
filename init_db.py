import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from app.db import Base
from app.config import settings

async def create_database():
    """Create the database and enable PostGIS extension"""
    db_url = settings.database_url
    db_name = db_url.split('/')[-1]
    
    conn_url = db_url.rsplit('/', 1)[0] + '/postgres'
    
    try:
        engine = create_async_engine(conn_url.replace('+asyncpg', ''))
        
        async with engine.connect() as conn:
            await conn.execute(text(f"CREATE DATABASE {db_name}"))
            await conn.commit()
        
        await engine.dispose()
        print(f"Database {db_name} created successfully")
    except Exception as e:
        print(f"Database might already exist: {e}")

    engine = create_async_engine(db_url)
    
    async with engine.connect() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        await conn.commit()
    
    await engine.dispose()
    print("PostGIS extension enabled")

async def create_tables():
    """Create all tables"""
    engine = create_async_engine(settings.database_url)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    await engine.dispose()
    print("Tables created successfully")

if __name__ == "__main__":
    from sqlalchemy import text
    asyncio.run(create_database())
    asyncio.run(create_tables())