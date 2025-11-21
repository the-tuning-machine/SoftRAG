"""
RAG Benchmark with Real Embeddings, Pre-trained Reranker, and Soft Sorting.

Uses:
- Sentence-Transformers for embeddings
- Pre-trained cross-encoder for reranking
- Soft sorting/ranking for differentiability
- MultiHopRAG dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from fast_soft_sort import pytorch_ops
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
import json


@dataclass
class RAGExample:
    """A single RAG example with query, relevant docs, and corpus."""
    query: str
    relevant_doc_ids: List[int]
    documents: List[str]


class RealEmbedder(nn.Module):
    """Real embedding model using Sentence-Transformers (CPU-compatible)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Embed texts using sentence-transformers."""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
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
            query_embed = nn.functional.normalize(query_embed, p=2, dim=-1)
            doc_embeds = nn.functional.normalize(doc_embeds, p=2, dim=-1)

        scores = torch.bmm(
            query_embed.unsqueeze(1),
            doc_embeds.transpose(1, 2)
        ).squeeze(1)

        return scores


class DifferentiableReRanker(nn.Module):
    """
    Differentiable reranker using a cross-encoder-style architecture with soft sorting.

    This uses a pre-trained cross-encoder backbone and adds soft sorting on top
    to make the ranking differentiable.
    """

    def __init__(self, cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__()
        print(f"Loading cross-encoder: {cross_encoder_name}...")

        # Load the cross-encoder model
        self.cross_encoder = CrossEncoder(cross_encoder_name)

        # We'll also create a learnable refinement layer on top of cross-encoder scores
        self.refinement = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, query: str, documents: List[str],
                initial_scores: torch.Tensor,
                regularization_strength: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rerank documents using cross-encoder + soft sorting.

        Args:
            query: Query string
            documents: List of document strings
            initial_scores: (batch_size, num_docs) - Initial ranking scores
            regularization_strength: Strength for soft sorting

        Returns:
            soft_sorted_scores: (batch_size, num_docs) - Soft sorted scores
            soft_ranks: (batch_size, num_docs) - Soft ranks
        """
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get cross-encoder scores (no gradients through the cross-encoder itself)
        with torch.no_grad():
            cross_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            cross_scores = torch.tensor(cross_scores, dtype=torch.float32).unsqueeze(0)

        # Apply learnable refinement (this is where gradients flow)
        refined_scores = self.refinement(cross_scores.unsqueeze(-1)).squeeze(-1)

        # Add residual connection with initial scores
        combined_scores = refined_scores + 0.1 * initial_scores

        # Apply soft sorting to get differentiable sorted scores
        soft_sorted_scores = pytorch_ops.soft_sort(
            combined_scores,
            direction="DESCENDING",
            regularization_strength=regularization_strength,
            regularization="l2"
        )

        # Also compute soft ranks
        soft_ranks = pytorch_ops.soft_rank(
            combined_scores,
            direction="DESCENDING",
            regularization_strength=regularization_strength,
            regularization="l2"
        )

        return soft_sorted_scores, soft_ranks


class RAGModel(nn.Module):
    """Complete differentiable RAG model with real embeddings and reranker."""

    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_cosine: bool = True):
        super().__init__()
        self.embedder = RealEmbedder(model_name=embedding_model)
        self.ranker = SimpleRanker(use_cosine=use_cosine)
        self.reranker = DifferentiableReRanker(cross_encoder_name=reranker_model)

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
            query, documents, initial_scores,
            regularization_strength=regularization_strength
        )

        return initial_scores, soft_sorted_scores, soft_ranks


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


def load_multihop_rag_dataset(split: str = "train", max_examples: int = 100):
    """Load and process the MultiHopRAG dataset."""
    print(f"Loading MultiHopRAG dataset ({split})...")
    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split=split)

    examples = []
    for i, item in enumerate(ds):
        if i >= max_examples:
            break

        query = item["query"]

        # Get all documents from evidence_list
        documents = []
        for evidence_item in item["evidence_list"]:
            # Each evidence item has a 'fact' field
            fact = evidence_item.get("fact", "")
            if fact:
                documents.append(fact)

        # Get answer/relevant info
        answer = item["answer"]

        # For simplicity, mark first few documents as relevant
        # In a real scenario, you'd compute this based on answer overlap
        relevant_doc_ids = list(range(min(3, len(documents))))

        if len(documents) > 0:
            examples.append(RAGExample(
                query=query,
                relevant_doc_ids=relevant_doc_ids,
                documents=documents
            ))

    print(f"Loaded {len(examples)} examples from {split}")
    return examples


def train_rag_model(model: RAGModel, train_dataset: List[RAGExample],
                    num_epochs: int = 5, learning_rate: float = 0.001,
                    regularization_strength: float = 0.1):
    """Train the RAG model."""
    # Only train the refinement layer in the reranker
    optimizer = optim.Adam(model.reranker.refinement.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, example in enumerate(train_dataset):
            if i % 10 == 0:
                print(f"  Example {i+1}/{len(train_dataset)}")

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

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def evaluate_rag_model(model: RAGModel, dataset: List[RAGExample],
                       regularization_strength: float = 0.1,
                       k: int = 10):
    """Evaluate the RAG model on test set."""
    model.eval()

    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0

    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i % 10 == 0:
                print(f"  Evaluating {i+1}/{len(dataset)}")

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
    print("=== RAG Benchmark with Real Embeddings & Reranker ===\n")

    # Hyperparameters
    num_train_examples = 50
    num_test_examples = 20
    num_epochs = 3
    learning_rate = 0.001
    regularization_strength = 0.1

    # Load dataset (only train split available, so we split it manually)
    full_dataset = load_multihop_rag_dataset(split="train", max_examples=num_train_examples + num_test_examples)
    train_dataset = full_dataset[:num_train_examples]
    test_dataset = full_dataset[num_train_examples:num_train_examples + num_test_examples]

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    # Create model
    print("\nCreating RAG model...")
    model = RAGModel(
        embedding_model="all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cosine=True
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.reranker.refinement.parameters())
    print(f"Trainable parameters (refinement layer): {trainable_params}")

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
