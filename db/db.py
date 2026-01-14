# db/db.py

"""
Database connections and logging setup for the project.
"""
import logging
import os
import subprocess
import bz2
from datetime import datetime
from sqlalchemy import create_engine

# Configure basic logging for database interactions
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def get_db_engine():
    """Create and return a SQLAlchemy engine for the project database."""
    conn_string = (
        f"postgresql+psycopg://{os.getenv('PROJECT_DB_USERNAME')}:{os.getenv('PROJECT_DB_PASSWORD')}"
        f"@{os.getenv('PROJECT_DB_HOST')}:{os.getenv('PROJECT_DB_PORT')}/{os.getenv('PROJECT_DB_NAME')}"
    )
    logger.info("Creating database engine")
    return create_engine(conn_string)


def backup_database(backup_dir: str, compress: bool = False) -> str:
    """Create a backup of the database. Saves to backup_dir. If compress is True, compress using bz2."""
    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)
    # Construct timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"archives_backup_{timestamp}.sql"
    backup_path = os.path.join(backup_dir, filename)
    if compress:
        backup_path += ".bz2"
    # Set PGPASSWORD for pg_dump
    env = os.environ.copy()
    env['PGPASSWORD'] = os.getenv('PROJECT_DB_PASSWORD')
    # Prepare pg_dump command
    pg_dump_cmd = [
        "pg_dump",
        "-h", os.getenv("PROJECT_DB_HOST", "localhost"),
        "-p", os.getenv("PROJECT_DB_PORT", "5432"),
        "-U", os.getenv("PROJECT_DB_USERNAME"),
        "-d", os.getenv("PROJECT_DB_NAME"),
    ]
    if compress:
        # Stream dump output and compress
        proc = subprocess.Popen(pg_dump_cmd, stdout=subprocess.PIPE, env=env)
        with bz2.open(backup_path, "wb") as f:
            for chunk in proc.stdout:
                f.write(chunk)
        proc.wait()
    else:
        # Write dump directly to file
        cmd_with_file = pg_dump_cmd + ["-f", backup_path]
        subprocess.run(cmd_with_file, check=True, env=env)
    return backup_path
