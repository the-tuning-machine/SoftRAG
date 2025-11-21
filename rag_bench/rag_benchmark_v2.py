"""
RAG Benchmark with Real Corpus Retrieval and Soft Sorting.

Correct architecture:
1. Build global corpus from ALL documents in dataset
2. Embed entire corpus once
3. For each query:
   - Retrieve top-k from corpus (first-stage ranking)
   - ReRank top-k with soft sorting
   - Compute loss against ground truth documents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from fast_soft_sort import pytorch_ops
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import json


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


class RealEmbedder(nn.Module):
    """Real embedding model using Sentence-Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Embed texts using sentence-transformers."""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=32
            )
        return embeddings


class FirstStageRetriever(nn.Module):
    """First-stage retriever using cosine similarity with soft top-k selection."""

    def __init__(self, regularization_strength: float = 0.1):
        super().__init__()
        self.regularization_strength = regularization_strength

    def forward(self, query_embed: torch.Tensor, corpus_embeds: torch.Tensor,
                k: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k documents for each query using soft sorting.

        Args:
            query_embed: (batch_size, embed_dim)
            corpus_embeds: (num_docs, embed_dim)
            k: Number of documents to retrieve

        Returns:
            similarities: (batch_size, num_docs) - Original similarity scores
            soft_sorted_scores: (batch_size, num_docs) - Soft sorted scores (top-k are highest)
            soft_ranks: (batch_size, num_docs) - Soft ranks (1 = best)
        """
        # Normalize for cosine similarity
        query_embed = F.normalize(query_embed, p=2, dim=-1)
        corpus_embeds = F.normalize(corpus_embeds, p=2, dim=-1)

        # Compute similarities: (batch_size, num_docs)
        similarities = torch.mm(query_embed, corpus_embeds.t())

        # Use soft sorting to get differentiable top-k
        soft_sorted_scores = pytorch_ops.soft_sort(
            similarities,
            direction="DESCENDING",
            regularization_strength=self.regularization_strength,
            regularization="l2"
        )

        soft_ranks = pytorch_ops.soft_rank(
            similarities,
            direction="DESCENDING",
            regularization_strength=self.regularization_strength,
            regularization="l2"
        )

        return similarities, soft_sorted_scores, soft_ranks


class CrossEncoderReRanker(nn.Module):
    """Reranker using cross-encoder with learnable refinement."""

    def __init__(self, cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__()
        print(f"Loading cross-encoder: {cross_encoder_name}...")
        self.cross_encoder = CrossEncoder(cross_encoder_name)

        # Learnable refinement layer
        self.refinement = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, query: str, documents: List[str],
                initial_scores: torch.Tensor) -> torch.Tensor:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Query string
            documents: List of document strings (top-k)
            initial_scores: (batch_size, k) - Initial scores from retriever

        Returns:
            final_scores: (batch_size, k) - Final reranking scores
        """
        # Get cross-encoder scores
        pairs = [[query, doc] for doc in documents]

        with torch.no_grad():
            cross_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            cross_scores = torch.tensor(cross_scores, dtype=torch.float32).unsqueeze(0)

        # Apply learnable refinement
        refined_scores = self.refinement(cross_scores.unsqueeze(-1)).squeeze(-1)

        # Residual connection with initial scores
        final_scores = refined_scores + 0.1 * initial_scores

        return final_scores


class RAGModel(nn.Module):
    """Complete RAG model with corpus retrieval."""

    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 regularization_strength: float = 0.1):
        super().__init__()
        self.embedder = RealEmbedder(model_name=embedding_model)
        self.retriever = FirstStageRetriever(regularization_strength=regularization_strength)
        self.reranker = CrossEncoderReRanker(cross_encoder_name=reranker_model)
        self.regularization_strength = regularization_strength

    def forward(self, query: str, corpus: DocumentCorpus, k: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full retrieval pipeline with soft top-k selection.

        Args:
            query: Query string
            corpus: Document corpus with embeddings
            k: Number of documents to retrieve

        Returns:
            top_k_indices: (1, k) - Global corpus IDs of top-k docs
            final_scores: (1, k) - Final reranking scores for top-k docs
        """
        # Embed query
        query_embed = self.embedder([query])

        # First-stage retrieval with soft sorting (differentiable)
        similarities, soft_sorted_scores, soft_ranks = self.retriever(
            query_embed, corpus.embeddings, k=k
        )

        # Select top-k based on soft ranks (hard selection for inference)
        top_k_indices = torch.argsort(soft_ranks[0])[:k].unsqueeze(0)

        # Get top-k scores from original similarities for residual connection
        top_k_initial_scores = torch.gather(similarities, 1, top_k_indices)

        # Get documents for reranking
        top_k_docs = [corpus.documents[idx] for idx in top_k_indices[0]]

        # Reranking with cross-encoder
        final_scores = self.reranker(query, top_k_docs, top_k_initial_scores)

        return top_k_indices, final_scores


def ranking_loss_sparse(final_scores: torch.Tensor, top_k_indices: torch.Tensor,
                        relevant_doc_ids: Set[int]) -> torch.Tensor:
    """
    Ranking loss for sparse labels based on final scores.

    Minimize negative scores for relevant docs (maximize positive scores).

    Args:
        final_scores: (1, k) - Final reranking scores of top-k docs
        top_k_indices: (1, k) - Global corpus IDs of retrieved docs
        relevant_doc_ids: Set of relevant document IDs

    Returns:
        loss: Scalar loss
    """
    k = final_scores.size(1)
    top_k_ids = set(top_k_indices[0].tolist())

    # Find which relevant docs are in top-k
    relevant_in_topk = top_k_ids & relevant_doc_ids

    if len(relevant_in_topk) == 0:
        # No relevant docs retrieved - penalize by using mean score
        # Negative mean (want to maximize scores, so minimize negative)
        return -final_scores.mean() + 10.0 * len(relevant_doc_ids)

    # For each relevant doc in top-k, maximize its score (minimize negative score)
    loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    for doc_id in relevant_in_topk:
        # Find position in top-k
        pos = (top_k_indices[0] == doc_id).nonzero(as_tuple=True)[0].item()
        # Minimize negative score = maximize score
        loss = loss - final_scores[0, pos]

    # Normalize by number of relevant docs
    loss = loss / len(relevant_doc_ids)

    return loss


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


def train_rag_model(model: RAGModel, corpus: DocumentCorpus, train_examples: List[RAGExample],
                    num_epochs: int = 3, learning_rate: float = 0.001, k: int = 100):
    """Train the RAG model."""
    optimizer = optim.Adam(model.reranker.refinement.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_relevant_retrieved = 0
        total_relevant = 0

        for i, example in enumerate(train_examples):
            if i % 10 == 0:
                print(f"  Example {i+1}/{len(train_examples)}")

            optimizer.zero_grad()

            # Forward pass
            top_k_indices, final_scores = model(example.query, corpus, k=k)

            # Compute loss
            loss = ranking_loss_sparse(final_scores, top_k_indices, example.relevant_doc_ids)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Track retrieval stats
            top_k_ids = set(top_k_indices[0].tolist())
            num_relevant_retrieved += len(top_k_ids & example.relevant_doc_ids)
            total_relevant += len(example.relevant_doc_ids)

        avg_loss = total_loss / len(train_examples)
        recall_at_k = num_relevant_retrieved / total_relevant if total_relevant > 0 else 0

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Recall@{k}: {recall_at_k:.4f}")


def evaluate_rag_model(model: RAGModel, corpus: DocumentCorpus, test_examples: List[RAGExample], k: int = 100):
    """Evaluate the RAG model."""
    model.eval()

    metrics = {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0
    }

    with torch.no_grad():
        for i, example in enumerate(test_examples):
            if i % 10 == 0:
                print(f"  Evaluating {i+1}/{len(test_examples)}")

            top_k_indices, final_scores = model(example.query, corpus, k=k)

            # Convert to lists
            top_k_ids = set(top_k_indices[0].tolist())
            retrieved_list = top_k_indices[0].tolist()

            # Compute metrics
            relevant_set = example.relevant_doc_ids
            tp = len(top_k_ids & relevant_set)

            precision = tp / k if k > 0 else 0.0
            recall = tp / len(relevant_set) if len(relevant_set) > 0 else 0.0

            # MRR
            mrr = 0.0
            for rank, doc_id in enumerate(retrieved_list, 1):
                if doc_id in relevant_set:
                    mrr = 1.0 / rank
                    break

            # NDCG@k
            dcg = 0.0
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
            for rank, doc_id in enumerate(retrieved_list[:k], 1):
                if doc_id in relevant_set:
                    dcg += 1.0 / np.log2(rank + 1)
            ndcg = dcg / idcg if idcg > 0 else 0.0

            metrics["precision_at_k"] += precision
            metrics["recall_at_k"] += recall
            metrics["mrr"] += mrr
            metrics["ndcg"] += ndcg

    # Average metrics
    num_examples = len(test_examples)
    for key in metrics:
        metrics[key] /= num_examples

    print(f"\nEvaluation Results:")
    print(f"Precision@{k}: {metrics['precision_at_k']:.4f}")
    print(f"Recall@{k}: {metrics['recall_at_k']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"NDCG@{k}: {metrics['ndcg']:.4f}")

    return metrics


def main():
    """Main benchmark function."""
    print("=== RAG Benchmark with Full Corpus Retrieval ===\n")

    # Hyperparameters
    max_examples = 500  # Use more examples for realistic corpus
    train_ratio = 0.8
    k = 20  # Top-k for reranking (realistic for cross-encoder)
    num_epochs = 5
    learning_rate = 0.001
    regularization_strength = 0.1

    # Load dataset and build corpus
    corpus, all_examples = load_multihop_rag_dataset(split="train", max_examples=max_examples)

    print("\nSample Corpus Documents:")
    print(corpus.documents[:5])
    print(all_examples[:5])

    # Split train/test
    split_idx = int(len(all_examples) * train_ratio)
    train_examples = all_examples[:split_idx]
    test_examples = all_examples[split_idx:]

    print(f"\nTrain examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Corpus size: {len(corpus)} documents")

    # Create model
    print("\nCreating RAG model...")
    model = RAGModel(
        embedding_model="all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Embed corpus
    corpus.embed_corpus(model.embedder)

    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.reranker.refinement.parameters())
    print(f"Trainable parameters: {trainable_params}")

    # Evaluate before training
    print(f"\n=== Before Training (k={k}) ===")
    evaluate_rag_model(model, corpus, test_examples, k=k,
                      regularization_strength=regularization_strength)

    # Train
    print(f"\n=== Training (k={k}) ===")
    train_rag_model(
        model, corpus, train_examples,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        k=k,
        regularization_strength=regularization_strength
    )

    # Evaluate after training
    print(f"\n=== After Training (k={k}) ===")
    evaluate_rag_model(model, corpus, test_examples, k=k,
                      regularization_strength=regularization_strength)

    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
