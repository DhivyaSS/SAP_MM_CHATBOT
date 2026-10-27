import logging
import json
from typing import List, Dict, Any, Optional
from app.database import get_db_connection
from app.config import settings
from app.services.chroma_store import ChromaVectorStoreService

logger = logging.getLogger(__name__)


class PGVectorStoreService:
    """
    Unified vector store service supporting both local ChromaDB and PostgreSQL + pgvector.
    Defaults to ChromaDB for zero-dependency local execution, or pgvector when configured.
    """

    @classmethod
    def _use_chroma(cls) -> bool:
        backend = getattr(settings, "VECTOR_STORE_TYPE", "chroma").lower()
        return backend == "chroma"

    @classmethod
    def is_document_ingested(cls, document_name: str) -> bool:
        """
        Checks whether a document with the given name has already been ingested.
        """
        if cls._use_chroma():
            return ChromaVectorStoreService.is_document_ingested(document_name)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE document_name = %s;", (document_name,))
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return count > 0
        except Exception as e:
            logger.warning(f"PostgreSQL check failed, falling back to local ChromaDB: {e}")
            return ChromaVectorStoreService.is_document_ingested(document_name)

    @classmethod
    def save_chunks(cls, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """
        Inserts document chunks along with their embeddings into vector storage.
        """
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("Chunks list and embeddings list must be non-empty and of equal length.")

        if cls._use_chroma():
            return ChromaVectorStoreService.save_chunks(chunks, embeddings)

        try:
            return cls._pg_save_chunks(chunks, embeddings)
        except Exception as e:
            logger.warning(f"PostgreSQL save failed, falling back to local ChromaDB: {e}")
            return ChromaVectorStoreService.save_chunks(chunks, embeddings)

    @classmethod
    def similarity_search(cls, query_vector: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Performs Cosine Similarity search to return top-k matching chunks sorted by relevance.
        """
        if cls._use_chroma():
            return ChromaVectorStoreService.similarity_search(query_vector, top_k=top_k)

        try:
            return cls._pg_similarity_search(query_vector, top_k=top_k)
        except Exception as e:
            logger.warning(f"PostgreSQL search failed, falling back to local ChromaDB: {e}")
            return ChromaVectorStoreService.similarity_search(query_vector, top_k=top_k)

    @classmethod
    def list_documents(cls) -> List[Dict[str, Any]]:
        """
        Get overview of all ingested documents and chunk totals.
        """
        if cls._use_chroma():
            return ChromaVectorStoreService.list_documents()

        try:
            return cls._pg_list_documents()
        except Exception as e:
            logger.warning(f"PostgreSQL list_documents failed, falling back to local ChromaDB: {e}")
            return ChromaVectorStoreService.list_documents()

    # -------------------------------------------------------------------------
    # PostgreSQL + pgvector Internal Methods
    # -------------------------------------------------------------------------
    @classmethod
    def _pg_save_chunks(cls, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        conn = get_db_connection()
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO document_chunks 
        (document_id, document_name, page_number, chunk_index, content, embedding, metadata)
        VALUES (%s, %s, %s, %s, %s, %s::vector, %s);
        """

        inserted_count = 0
        try:
            for chunk, emb in zip(chunks, embeddings):
                emb_str = f"[{','.join(map(str, emb))}]"
                metadata_json = json.dumps(chunk.get("metadata", {}))

                cursor.execute(insert_query, (
                    chunk["document_id"],
                    chunk["document_name"],
                    chunk["page_number"],
                    chunk["chunk_index"],
                    chunk["content"],
                    emb_str,
                    metadata_json
                ))
                inserted_count += 1

            conn.commit()
            logger.info(f"Successfully saved {inserted_count} chunks into pgvector.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting chunks into PostgreSQL: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

        return inserted_count

    @classmethod
    def _pg_similarity_search(cls, query_vector: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or settings.TOP_K
        emb_str = f"[{','.join(map(str, query_vector))}]"

        conn = get_db_connection()
        cursor = conn.cursor()

        search_query = """
        SELECT 
            id,
            document_id,
            document_name,
            page_number,
            chunk_index,
            content,
            (embedding <=> %s::vector) AS distance
        FROM document_chunks
        ORDER BY distance ASC
        LIMIT %s;
        """

        results = []
        try:
            cursor.execute(search_query, (emb_str, k))
            rows = cursor.fetchall()

            for row in rows:
                distance = float(row[6])
                similarity_score = round(1.0 - distance, 4)
                results.append({
                    "id": row[0],
                    "document_id": row[1],
                    "document_name": row[2],
                    "page_number": row[3],
                    "chunk_index": row[4],
                    "content": row[5],
                    "distance": distance,
                    "similarity": similarity_score
                })

            logger.info(f"Retrieved {len(results)} chunks using pgvector cosine search.")
        except Exception as e:
            logger.error(f"Error during pgvector similarity search: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

        return results

    @classmethod
    def _pg_list_documents(cls) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        SELECT 
            document_name, 
            COUNT(*) as chunk_count, 
            MAX(page_number) as max_page,
            MIN(created_at) as ingested_at
        FROM document_chunks
        GROUP BY document_name
        ORDER BY ingested_at DESC;
        """

        docs = []
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            for r in rows:
                docs.append({
                    "document_name": r[0],
                    "chunk_count": r[1],
                    "total_pages": r[2],
                    "ingested_at": str(r[3])
                })
        except Exception as e:
            logger.error(f"Error querying document summary from PostgreSQL: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

        return docs


# Alias for clean architecture
VectorStoreService = PGVectorStoreService
