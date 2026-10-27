import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)

# Try to import chromadb, provide fallback notice if missing
try:
    # pyrefly: ignore [missing-import]
    import chromadb
    # pyrefly: ignore [missing-import]
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:  # pragma: no cover
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb package is not installed. Using local JSON vector storage fallback until 'pip install chromadb' is run.")


class ChromaVectorStoreService:
    """
    Local ChromaDB vector store service.
    Persists vectors to disk under settings.CHROMA_PERSIST_DIRECTORY without requiring Docker or PostgreSQL.
    """
    _client = None
    _collection = None

    @classmethod
    def _get_client_and_collection(cls):
        """
        Lazily initialize persistent ChromaDB client and collection.
        """
        if cls._collection is not None:
            return cls._client, cls._collection

        os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

        if CHROMADB_AVAILABLE:
            cls._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            cls._collection = cls._client.get_or_create_collection(
                name="sap_mm_documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized ChromaDB persistent vector store at: {settings.CHROMA_PERSIST_DIRECTORY}")
        else:
            cls._client = None
            cls._collection = None

        return cls._client, cls._collection

    # -------------------------------------------------------------------------
    # Ingestion Check
    # -------------------------------------------------------------------------
    @classmethod
    def is_document_ingested(cls, document_name: str) -> bool:
        """
        Checks whether chunks for the document already exist in ChromaDB.
        """
        try:
            _, collection = cls._get_client_and_collection()
            if collection is not None:
                results = collection.get(
                    where={"document_name": document_name},
                    limit=1
                )
                return bool(results and results.get("ids") and len(results["ids"]) > 0)
            else:
                return cls._fallback_is_ingested(document_name)
        except Exception as e:
            logger.error(f"Error checking document status in ChromaDB: {e}")
            return False

    # -------------------------------------------------------------------------
    # Save Chunks
    # -------------------------------------------------------------------------
    @classmethod
    def save_chunks(cls, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """
        Saves text chunks, embeddings, and metadata into local ChromaDB.
        """
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("Chunks list and embeddings list must be non-empty and of equal length.")

        _, collection = cls._get_client_and_collection()
        now_iso = datetime.utcnow().isoformat()

        if collection is not None:
            ids = []
            documents = []
            metadatas = []
            valid_embeddings = []

            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{chunk['document_id']}_{chunk['chunk_index']}"
                ids.append(chunk_id)
                documents.append(chunk["content"])
                valid_embeddings.append(emb)

                meta = {
                    "document_id": str(chunk.get("document_id", "")),
                    "document_name": str(chunk.get("document_name", "")),
                    "page_number": int(chunk.get("page_number", 1)),
                    "chunk_index": int(chunk.get("chunk_index", idx)),
                    "created_at": now_iso
                }
                metadatas.append(meta)

            collection.add(
                ids=ids,
                embeddings=valid_embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully saved {len(ids)} chunks to ChromaDB.")
            return len(ids)
        else:
            return cls._fallback_save_chunks(chunks, embeddings, now_iso)

    # -------------------------------------------------------------------------
    # Similarity Search
    # -------------------------------------------------------------------------
    @classmethod
    def similarity_search(cls, query_vector: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Performs Cosine Similarity search against ChromaDB vectors.
        Returns top-k matching chunks sorted by relevance.
        """
        k = top_k or settings.TOP_K
        _, collection = cls._get_client_and_collection()

        if collection is not None:
            total_items = collection.count()
            if total_items == 0:
                return []

            actual_k = min(k, total_items)
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                ids = results["ids"][0]
                docs = results["documents"][0] if results.get("documents") else []
                metas = results["metadatas"][0] if results.get("metadatas") else []
                distances = results["distances"][0] if results.get("distances") else []

                for i in range(len(ids)):
                    dist = float(distances[i]) if i < len(distances) else 0.0
                    similarity = round(1.0 - dist, 4)
                    meta = metas[i] if i < len(metas) else {}

                    formatted_results.append({
                        "id": ids[i],
                        "document_id": meta.get("document_id", ""),
                        "document_name": meta.get("document_name", "Unknown"),
                        "page_number": int(meta.get("page_number", 1)),
                        "chunk_index": int(meta.get("chunk_index", 0)),
                        "content": docs[i] if i < len(docs) else "",
                        "distance": dist,
                        "similarity": similarity
                    })

            logger.info(f"Retrieved {len(formatted_results)} chunks from ChromaDB.")
            return formatted_results
        else:
            return cls._fallback_similarity_search(query_vector, k)

    # -------------------------------------------------------------------------
    # List Documents
    # -------------------------------------------------------------------------
    @classmethod
    def list_documents(cls) -> List[Dict[str, Any]]:
        """
        Returns list of ingested documents and their chunk totals.
        """
        _, collection = cls._get_client_and_collection()
        if collection is not None:
            try:
                data = collection.get(include=["metadatas"])
                metas = data.get("metadatas", [])
                
                doc_stats: Dict[str, Dict[str, Any]] = {}
                for m in metas:
                    name = m.get("document_name", "Unknown")
                    page = int(m.get("page_number", 1))
                    created = m.get("created_at", "")
                    
                    if name not in doc_stats:
                        doc_stats[name] = {
                            "document_name": name,
                            "chunk_count": 0,
                            "total_pages": page,
                            "ingested_at": created
                        }
                    doc_stats[name]["chunk_count"] += 1
                    if page > doc_stats[name]["total_pages"]:
                        doc_stats[name]["total_pages"] = page

                return sorted(list(doc_stats.values()), key=lambda x: x["ingested_at"], reverse=True)
            except Exception as e:
                logger.error(f"Error listing documents from ChromaDB: {e}")
                return []
        else:
            return cls._fallback_list_documents()

    # -------------------------------------------------------------------------
    # Local JSON / Pure-Python Fallback (if chromadb package is not yet installed)
    # -------------------------------------------------------------------------
    @classmethod
    def _get_fallback_file_path(cls) -> str:
        fallback_dir = os.path.join(settings.CHROMA_PERSIST_DIRECTORY, "fallback")
        os.makedirs(fallback_dir, exist_ok=True)
        return os.path.join(fallback_dir, "chunks_store.json")

    @classmethod
    def _read_fallback_data(cls) -> List[Dict[str, Any]]:
        path = cls._get_fallback_file_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    @classmethod
    def _write_fallback_data(cls, data: List[Dict[str, Any]]):
        path = cls._get_fallback_file_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def _fallback_is_ingested(cls, document_name: str) -> bool:
        items = cls._read_fallback_data()
        return any(item.get("document_name") == document_name for item in items)

    @classmethod
    def _fallback_save_chunks(cls, chunks: List[Dict[str, Any]], embeddings: List[List[float]], now_iso: str) -> int:
        items = cls._read_fallback_data()
        added = 0
        for chunk, emb in zip(chunks, embeddings):
            items.append({
                "id": f"{chunk['document_id']}_{chunk['chunk_index']}",
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "embedding": emb,
                "created_at": now_iso
            })
            added += 1
        cls._write_fallback_data(items)
        logger.info(f"Saved {added} chunks into fallback JSON vector store.")
        return added

    @classmethod
    def _fallback_similarity_search(cls, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        import math

        def cosine_sim(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        items = cls._read_fallback_data()
        if not items:
            return []

        scored = []
        for it in items:
            sim = cosine_sim(query_vector, it["embedding"])
            dist = 1.0 - sim
            scored.append({
                "id": it["id"],
                "document_id": it["document_id"],
                "document_name": it["document_name"],
                "page_number": it["page_number"],
                "chunk_index": it["chunk_index"],
                "content": it["content"],
                "distance": dist,
                "similarity": round(sim, 4)
            })

        scored.sort(key=lambda x: x["distance"])
        return scored[:top_k]

    @classmethod
    def _fallback_list_documents(cls) -> List[Dict[str, Any]]:
        items = cls._read_fallback_data()
        stats: Dict[str, Dict[str, Any]] = {}
        for it in items:
            name = it["document_name"]
            page = it["page_number"]
            if name not in stats:
                stats[name] = {
                    "document_name": name,
                    "chunk_count": 0,
                    "total_pages": page,
                    "ingested_at": it.get("created_at", "")
                }
            stats[name]["chunk_count"] += 1
            if page > stats[name]["total_pages"]:
                stats[name]["total_pages"] = page
        return sorted(list(stats.values()), key=lambda x: x["ingested_at"], reverse=True)
