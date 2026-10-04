import chromadb
from sentence_transformers import SentenceTransformer
import config

class VectorStoreManager:
    def __init__(self):
        # Initialize persistent ChromaDB storage
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        # Load lightweight CPU embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = self.chroma_client.get_or_create_collection(name="study_materials")

    def _chunk_text(self, text: str):
        """Splits long text into overlapping chunks for semantic indexing."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), config.CHUNK_SIZE - config.CHUNK_OVERLAP):
            chunks.append(" ".join(words[i:i + config.CHUNK_SIZE]))
        return chunks

    def add_document(self, doc_id: str, text: str):
        """Chunks, embeds, and stores a document in ChromaDB."""
        chunks = self._chunk_text(text)
        if not chunks:
            return
        
        embeddings = self.embedder.encode(chunks).tolist()
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

    def query_context_with_accuracy(self, query: str, n_results: int = 3):
        """
        Retrieves relevant text chunks and computes a semantic match similarity score.
        """
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "", 0.0
        
        # Calculate cosine/L2 match score percentage
        distances = results['distances'][0]
        top_distance = distances[0]
        match_accuracy = round(max(0.0, (1.0 - top_distance)) * 100, 2)
        
        context_text = " ".join(results['documents'][0])
        return context_text, match_accuracy