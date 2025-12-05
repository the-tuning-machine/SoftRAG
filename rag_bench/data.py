from dataclasses import dataclass
from typing import Set, List
from datasets import load_dataset
import torch
import torch.nn as nn

@dataclass
class RAGExample:
    """A single RAG example with query and ground truth document IDs."""
    query_id: int
    query: str
    relevant_doc_ids: Set[int]  # Global corpus IDs


class DocumentCorpus:
    """Global document corpus with embeddings."""

    def __init__(self):
        self.documents = []  # List of document strings
        self.doc_to_id = {}  # Map doc string to corpus ID
        self.embeddings = None  # Will be filled with embeddings
        self.metadata = []  # Store metadata (source, etc.)

    def add_document(self, doc: str, metadata: dict = None) -> int:
        """Add document to corpus, return its ID."""
        if doc in self.doc_to_id:
            return self.doc_to_id[doc]

        doc_id = len(self.documents)
        self.documents.append(doc)
        self.doc_to_id[doc] = doc_id
        self.metadata.append(metadata or {})
        return doc_id

    def embed_corpus(self, embedder: nn.Module):
        """Embed all documents in the corpus."""
        print(f"Embedding corpus of {len(self.documents)} documents...")
        # Batch embedding for efficiency
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            with torch.no_grad():
                batch_embeds = embedder(batch)
            all_embeddings.append(batch_embeds)

        self.embeddings = torch.cat(all_embeddings, dim=0)
        print(f"Corpus embedded: {self.embeddings.shape}")

    def __len__(self):
        return len(self.documents)


def load_multihop_rag_dataset(split: str = "train", max_examples: int = None):
    """Load MultiHopRAG and build corpus + examples."""
    print(f"Loading MultiHopRAG dataset ({split})...")
    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split=split)

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    # Build global corpus
    corpus = DocumentCorpus()
    examples = []

    for idx, item in enumerate(ds):
        query = item["query"]

        # Add all evidence documents to corpus and track their IDs
        relevant_doc_ids = set()
        for evidence_item in item["evidence_list"]:
            fact = evidence_item.get("fact", "")
            if fact:
                doc_id = corpus.add_document(fact, metadata={
                    "source": evidence_item.get("source", ""),
                    "title": evidence_item.get("title", "")
                })
                relevant_doc_ids.add(doc_id)

        if len(relevant_doc_ids) > 0:
            examples.append(RAGExample(
                query_id=idx,
                query=query,
                relevant_doc_ids=relevant_doc_ids
            ))

    print(f"Loaded {len(examples)} examples")
    print(f"Built corpus with {len(corpus)} unique documents")

    return corpus, examples
