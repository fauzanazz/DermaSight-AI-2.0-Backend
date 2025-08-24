import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.db import Base
from app.config import settings

async def create_database():
    """Create the database and enable PostGIS extension"""
    try:
        db_url = settings.database_url
        db_name = db_url.split('/')[-1]
        
        # Connect to postgres database to create our database
        conn_url = db_url.rsplit('/', 1)[0] + '/postgres'
        
        try:
            engine = create_async_engine(conn_url)
            
            async with engine.connect() as conn:
                # Check if database exists
                result = await conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
                exists = result.fetchone()
                
                if not exists:
                    await conn.execute(text(f"CREATE DATABASE {db_name}"))
                    await conn.commit()
                    print(f"Database {db_name} created successfully")
                else:
                    print(f"Database {db_name} already exists")
            
            await engine.dispose()
        except Exception as e:
            print(f"Database creation error: {e}")

        # Enable PostGIS extension
        try:
            engine = create_async_engine(db_url)
            
            async with engine.connect() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
                await conn.commit()
                print("PostGIS extension enabled")
            
            await engine.dispose()
        except Exception as e:
            print(f"PostGIS extension error: {e}")
            
    except Exception as e:
        print(f"Database initialization error: {e}")

async def create_tables():
    """Create all tables"""
    try:
        engine = create_async_engine(settings.database_url)
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        await engine.dispose()
        print("Tables created successfully")
        
    except Exception as e:
        print(f"Table creation error: {e}")

if __name__ == "__main__":
    print("Initializing database...")
    asyncio.run(create_database())
    asyncio.run(create_tables())