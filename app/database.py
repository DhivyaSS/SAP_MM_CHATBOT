import logging
# pyrefly: ignore [missing-import]
import psycopg2
# pyrefly: ignore [missing-import]
from psycopg2.extras import RealDictCursor
from app.config import settings

logger = logging.getLogger(__name__)


def get_db_connection():
    """
    Establish a direct connection to PostgreSQL database.
    """
    conn = psycopg2.connect(
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT
    )
    return conn


def init_db():
    """
    Initialize pgvector extension and document_chunks table schema if not present.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # 2. Create document_chunks table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            document_id VARCHAR(255) NOT NULL,
            document_name VARCHAR(255) NOT NULL,
            page_number INT NOT NULL,
            chunk_index INT NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(768),
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        
        # 3. Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON document_chunks(document_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_name ON document_chunks(document_name);")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding 
            ON document_chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database initialized with pgvector vector storage schema successfully.")
    except Exception as e:
        logger.error(f"Error initializing PostgreSQL database: {e}")
        # Note: If PostgreSQL container is not running yet during early setup, this catches gracefully.
