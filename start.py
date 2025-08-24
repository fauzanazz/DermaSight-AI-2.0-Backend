import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print its status"""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"{description} completed successfully")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"{description} failed")
        print(result.stderr)
        return False
    return True

def main():
    print("DermaSight AI Backend Setup")
    
    if not run_command("docker --version", "Checking Docker"):
        print("Docker is required but not found. Please install Docker first.")
        sys.exit(1)
    
    if not run_command("docker-compose --version", "Checking Docker Compose"):
        print("Docker Compose is required but not found. Please install Docker Compose first.")
        sys.exit(1)
    
    print("Starting Docker services...")
    if not run_command("docker-compose up -d db", "Starting PostgreSQL database"):
        sys.exit(1)
    
    print("Waiting for database to be ready...")
    import time
    time.sleep(10)
    
    print("Setting up database and seeding data...")
    if run_command("python init_db.py", "Initializing database"):
        run_command("python seed_data.py", "Seeding sample data")
    
    print("Starting API server...")
    if run_command("docker-compose up -d api", "Starting FastAPI server"):
        print("\n" + "="*60)
        print("DermaSight AI Backend is running!")
        print("API Documentation: http://localhost:8000/docs")
        print("Health Check: http://localhost:8000/health")
        print("View logs: docker-compose logs -f api")
        print("Stop services: docker-compose down")
        print("="*60)

if __name__ == "__main__":
    main()