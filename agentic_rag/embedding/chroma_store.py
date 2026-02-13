import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional


class ChromaStore:
    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    def get_or_create(self, name: str):
        return self.client.get_or_create_collection(name=name)

    def upsert(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        col = self.get_or_create(collection_name)
        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def count(self, collection_name: str) -> int:
        return self.get_or_create(collection_name).count()

    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int,
        include: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        col = self.get_or_create(collection_name)
        kwargs: Dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include or ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    def list_collections(self) -> List[str]:
        out: List[str] = []
        for collection in self.client.list_collections():
            if isinstance(collection, str) and collection:
                out.append(collection)
                continue
            name = getattr(collection, "name", None)
            if isinstance(name, str) and name:
                out.append(name)
        return out

    def get_metadatas(
        self,
        collection_name: str,
        limit: int,
        offset: int = 0,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        col = self.get_or_create(collection_name)
        kwargs: Dict[str, Any] = {
            "limit": max(1, limit),
            "offset": max(0, offset),
            "include": ["metadatas"],
        }
        if where:
            kwargs["where"] = where
        try:
            response = col.get(**kwargs)
        except TypeError:
            fallback_kwargs: Dict[str, Any] = {
                "limit": max(1, limit + max(0, offset)),
                "include": ["metadatas"],
            }
            if where:
                fallback_kwargs["where"] = where
            response = col.get(**fallback_kwargs)
            metadatas_raw = response.get("metadatas") or []
            if offset > 0:
                metadatas_raw = metadatas_raw[offset:]
            metadatas_raw = metadatas_raw[: max(1, limit)]
            return [meta for meta in metadatas_raw if isinstance(meta, dict)]
        metadatas = response.get("metadatas") or []
        return [meta for meta in metadatas if isinstance(meta, dict)]

    def get_records(
        self,
        collection_name: str,
        limit: int,
        offset: int = 0,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        col = self.get_or_create(collection_name)
        kwargs: Dict[str, Any] = {
            "limit": max(1, limit),
            "offset": max(0, offset),
            "include": ["documents", "metadatas"],
        }
        if where:
            kwargs["where"] = where
        try:
            response = col.get(**kwargs)
        except TypeError:
            fallback_kwargs: Dict[str, Any] = {
                "limit": max(1, limit + max(0, offset)),
                "include": ["documents", "metadatas"],
            }
            if where:
                fallback_kwargs["where"] = where
            response = col.get(**fallback_kwargs)
            ids = (response.get("ids") or [])[offset: offset + max(1, limit)]
            docs = (response.get("documents") or [])[offset: offset + max(1, limit)]
            metas = (response.get("metadatas") or [])[offset: offset + max(1, limit)]
            return {
                "ids": ids,
                "documents": docs,
                "metadatas": metas,
            }
        return {
            "ids": response.get("ids") or [],
            "documents": response.get("documents") or [],
            "metadatas": response.get("metadatas") or [],
        }
