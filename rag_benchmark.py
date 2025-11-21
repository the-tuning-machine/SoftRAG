"""
Benchmark for differentiable RAG using soft sorting/ranking.

This module implements a complete differentiable RAG pipeline:
1. Embed documents and queries
2. Simple ranking (dot product / cosine similarity)
3. Soft top-k selection
4. ReRanking with soft sorting
5. Loss computation and backpropagation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from fast_soft_sort import pytorch_ops
import json
import os


@dataclass
class RAGExample:
    """A single RAG example with query, relevant docs, and corpus."""
    query: str
    relevant_doc_ids: List[int]
    documents: List[str]


class SimpleEmbedder(nn.Module):
    """Simple embedding model using a learnable embedding layer.

    For a real benchmark, replace this with sentence-transformers or similar.
    """

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size

    def tokenize(self, texts: List[str]) -> torch.Tensor:
        """Simple tokenization: hash each word to vocab index."""
        batch_tokens = []
        max_len = 32

        for text in texts:
            words = text.lower().split()[:max_len]
            tokens = [hash(word) % self.vocab_size for word in words]
            # Pad to max_len
            tokens = tokens + [0] * (max_len - len(tokens))
            batch_tokens.append(tokens)

        return torch.tensor(batch_tokens, dtype=torch.long)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Embed texts into vectors."""
        tokens = self.tokenize(texts)
        # Average pooling over token embeddings
        embeds = self.embedding(tokens)
        return embeds.mean(dim=1)


class SentenceTransformerEmbedder(nn.Module):
    """Wrapper for sentence-transformers models (CPU-compatible)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            print("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Embed texts using sentence-transformers."""
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings


class SimpleRanker(nn.Module):
    """Simple first-stage ranker using dot product or cosine similarity."""

    def __init__(self, use_cosine: bool = True):
        super().__init__()
        self.use_cosine = use_cosine

    def forward(self, query_embed: torch.Tensor, doc_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between query and documents.

        Args:
            query_embed: (batch_size, embed_dim)
            doc_embeds: (batch_size, num_docs, embed_dim)

        Returns:
            scores: (batch_size, num_docs)
        """
        if self.use_cosine:
            # Normalize embeddings
            query_embed = nn.functional.normalize(query_embed, p=2, dim=-1)
            doc_embeds = nn.functional.normalize(doc_embeds, p=2, dim=-1)

        # Compute dot product: (batch_size, 1, embed_dim) @ (batch_size, embed_dim, num_docs)
        scores = torch.bmm(
            query_embed.unsqueeze(1),
            doc_embeds.transpose(1, 2)
        ).squeeze(1)

        return scores


class SoftReRanker(nn.Module):
    """
    ReRanker using soft sorting for differentiable top-k selection.

    This module takes initial scores and refines them using a learned
    transformation, then applies soft sorting to get differentiable ranks.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query_embed: torch.Tensor, doc_embeds: torch.Tensor,
                initial_scores: torch.Tensor,
                regularization_strength: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rerank documents using soft sorting.

        Args:
            query_embed: (batch_size, embed_dim)
            doc_embeds: (batch_size, num_docs, embed_dim)
            initial_scores: (batch_size, num_docs)
            regularization_strength: Strength for soft sorting

        Returns:
            soft_sorted_scores: (batch_size, num_docs) - Soft sorted scores
            soft_ranks: (batch_size, num_docs) - Soft ranks
        """
        batch_size, num_docs, embed_dim = doc_embeds.shape

        # Concatenate query (broadcast) with each doc and initial score
        query_expanded = query_embed.unsqueeze(1).expand(-1, num_docs, -1)
        initial_scores_expanded = initial_scores.unsqueeze(-1)

        # Features: [query, doc, initial_score]
        features = torch.cat([
            query_expanded,
            doc_embeds,
            initial_scores_expanded
        ], dim=-1)

        # Compute refined scores
        refined_scores = self.mlp(features).squeeze(-1)

        # Apply soft sorting to get differentiable sorted scores
        # soft_sort expects 2D input: (batch_size, num_docs)
        soft_sorted_scores = pytorch_ops.soft_sort(
            refined_scores,
            direction="DESCENDING",
            regularization_strength=regularization_strength,
            regularization="l2"
        )

        # Also compute soft ranks
        soft_ranks = pytorch_ops.soft_rank(
            refined_scores,
            direction="DESCENDING",
            regularization_strength=regularization_strength,
            regularization="l2"
        )

        return soft_sorted_scores, soft_ranks


class RAGModel(nn.Module):
    """Complete differentiable RAG model."""

    def __init__(self, embedder: nn.Module, use_cosine: bool = True):
        super().__init__()
        self.embedder = embedder
        self.ranker = SimpleRanker(use_cosine=use_cosine)

        # ReRanker input: [query_embed, doc_embed, initial_score]
        input_dim = embedder.embedding_dim * 2 + 1
        self.reranker = SoftReRanker(input_dim=input_dim)

    def forward(self, query: str, documents: List[str],
                regularization_strength: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RAG pipeline.

        Args:
            query: Query string
            documents: List of document strings
            regularization_strength: Strength for soft sorting

        Returns:
            initial_scores: (1, num_docs) - Initial ranking scores
            soft_sorted_scores: (1, num_docs) - Soft sorted scores
            soft_ranks: (1, num_docs) - Soft ranks
        """
        # Embed query and documents
        query_embed = self.embedder([query])
        doc_embeds = self.embedder(documents).unsqueeze(0)

        # Initial ranking
        initial_scores = self.ranker(query_embed, doc_embeds)

        # ReRanking with soft sorting
        soft_sorted_scores, soft_ranks = self.reranker(
            query_embed, doc_embeds, initial_scores,
            regularization_strength=regularization_strength
        )

        return initial_scores, soft_sorted_scores, soft_ranks


def ndcg_loss(soft_ranks: torch.Tensor, relevant_doc_ids: List[int],
              num_docs: int, k: int = 10) -> torch.Tensor:
    """
    Compute NDCG-based loss for ranking.

    The loss encourages relevant documents to have lower (better) ranks.

    Args:
        soft_ranks: (batch_size, num_docs) - Soft ranks from model
        relevant_doc_ids: List of relevant document indices
        num_docs: Total number of documents
        k: Top-k for NDCG computation

    Returns:
        loss: Scalar loss value
    """
    # Create relevance labels
    relevance = torch.zeros(num_docs)
    for doc_id in relevant_doc_ids:
        if doc_id < num_docs:
            relevance[doc_id] = 1.0

    # Discount factor: 1 / log2(rank + 1)
    # Lower rank (better) should give higher score
    # We want to minimize rank for relevant docs

    # Create target: relevant docs should have rank 1, 2, 3, ...
    # Non-relevant docs should have high ranks
    target_ranks = torch.ones_like(soft_ranks[0]) * (num_docs + 1)
    for i, doc_id in enumerate(relevant_doc_ids):
        if doc_id < num_docs:
            target_ranks[doc_id] = i + 1

    # MSE loss on ranks for relevant documents
    loss = torch.sum(relevance * (soft_ranks[0] - target_ranks) ** 2)

    return loss


def ranking_loss(soft_ranks: torch.Tensor, relevant_doc_ids: List[int],
                 num_docs: int) -> torch.Tensor:
    """
    Simple ranking loss: minimize the rank of relevant documents.

    Args:
        soft_ranks: (batch_size, num_docs) - Soft ranks from model
        relevant_doc_ids: List of relevant document indices
        num_docs: Total number of documents

    Returns:
        loss: Scalar loss value
    """
    loss = 0.0
    for doc_id in relevant_doc_ids:
        if doc_id < num_docs:
            # Penalize high ranks for relevant documents
            loss += soft_ranks[0, doc_id]

    # Normalize by number of relevant documents
    if len(relevant_doc_ids) > 0:
        loss = loss / len(relevant_doc_ids)

    return loss


class SyntheticRAGDataset:
    """
    Create a synthetic RAG dataset for testing.

    Each query is designed to match certain documents based on keywords.
    """

    def __init__(self, num_examples: int = 100, num_docs_per_example: int = 50):
        self.examples = []

        topics = [
            "machine learning", "deep learning", "neural networks",
            "natural language processing", "computer vision",
            "reinforcement learning", "optimization", "statistics"
        ]

        for i in range(num_examples):
            topic = topics[i % len(topics)]
            query = f"What is {topic} and how does it work?"

            documents = []
            relevant_ids = []

            for j in range(num_docs_per_example):
                if j < 3:
                    # Relevant documents
                    doc = f"This document explains {topic} in detail. " \
                          f"{topic} is an important concept in AI research."
                    relevant_ids.append(j)
                else:
                    # Irrelevant documents
                    other_topic = topics[(i + j) % len(topics)]
                    doc = f"This document discusses {other_topic}. " \
                          f"It covers various aspects of {other_topic}."

                documents.append(doc)

            self.examples.append(RAGExample(
                query=query,
                relevant_doc_ids=relevant_ids,
                documents=documents
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train_rag_model(model: RAGModel, dataset: SyntheticRAGDataset,
                    num_epochs: int = 10, learning_rate: float = 0.001,
                    regularization_strength: float = 0.1):
    """Train the RAG model."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, example in enumerate(dataset):
            optimizer.zero_grad()

            # Forward pass
            initial_scores, soft_sorted_scores, soft_ranks = model(
                example.query,
                example.documents,
                regularization_strength=regularization_strength
            )

            # Compute loss
            loss = ranking_loss(soft_ranks, example.relevant_doc_ids,
                              len(example.documents))

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def evaluate_rag_model(model: RAGModel, dataset: SyntheticRAGDataset,
                       regularization_strength: float = 0.1,
                       k: int = 10):
    """Evaluate the RAG model on test set."""
    model.eval()

    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0

    with torch.no_grad():
        for example in dataset:
            initial_scores, soft_sorted_scores, soft_ranks = model(
                example.query,
                example.documents,
                regularization_strength=regularization_strength
            )

            # Get top-k by sorting on ranks (lower is better)
            top_k_indices = torch.argsort(soft_ranks[0])[:k].tolist()

            # Compute metrics
            relevant_set = set(example.relevant_doc_ids)
            retrieved_set = set(top_k_indices)

            tp = len(relevant_set & retrieved_set)
            precision = tp / k if k > 0 else 0.0
            recall = tp / len(relevant_set) if len(relevant_set) > 0 else 0.0

            # MRR: reciprocal rank of first relevant document
            mrr = 0.0
            for rank, idx in enumerate(top_k_indices, 1):
                if idx in relevant_set:
                    mrr = 1.0 / rank
                    break

            total_precision += precision
            total_recall += recall
            total_mrr += mrr

    num_examples = len(dataset)
    avg_precision = total_precision / num_examples
    avg_recall = total_recall / num_examples
    avg_mrr = total_mrr / num_examples

    print(f"\nEvaluation Results (top-{k}):")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"MRR: {avg_mrr:.4f}")

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "mrr": avg_mrr
    }


def main():
    """Main benchmark function."""
    print("=== Differentiable RAG Benchmark with Soft Sorting ===\n")

    # Hyperparameters
    num_train_examples = 80
    num_test_examples = 20
    num_docs_per_example = 50
    embedding_dim = 128
    num_epochs = 20
    learning_rate = 0.01
    regularization_strength = 0.1

    # Create datasets
    print("Creating synthetic RAG dataset...")
    full_dataset = SyntheticRAGDataset(
        num_examples=num_train_examples + num_test_examples,
        num_docs_per_example=num_docs_per_example
    )

    train_dataset = SyntheticRAGDataset.__new__(SyntheticRAGDataset)
    train_dataset.examples = full_dataset.examples[:num_train_examples]

    test_dataset = SyntheticRAGDataset.__new__(SyntheticRAGDataset)
    test_dataset.examples = full_dataset.examples[num_train_examples:]

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    # Create model
    print("\nCreating RAG model with simple embedder...")
    embedder = SimpleEmbedder(vocab_size=10000, embedding_dim=embedding_dim)
    model = RAGModel(embedder, use_cosine=True)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Evaluate before training
    print("\n=== Before Training ===")
    evaluate_rag_model(model, test_dataset, regularization_strength=regularization_strength)

    # Train
    print("\n=== Training ===")
    train_rag_model(
        model, train_dataset,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        regularization_strength=regularization_strength
    )

    # Evaluate after training
    print("\n=== After Training ===")
    evaluate_rag_model(model, test_dataset, regularization_strength=regularization_strength)

    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
