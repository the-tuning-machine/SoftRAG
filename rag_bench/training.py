import torch
import torch.optim as optim
import numpy as np
from typing import List, Set
from .data import RAGExample, DocumentCorpus
from .models import RAGModel

def ranking_loss_sparse(final_scores: torch.Tensor, top_k_indices: torch.Tensor,
                        relevant_doc_ids: Set[int]) -> torch.Tensor:
    """
    Ranking loss using Binary Cross Entropy (BCE).
    
    Args:
        final_scores: (1, k) - Final reranking scores (logits) of top-k docs
        top_k_indices: (1, k) - Global corpus IDs of retrieved docs
        relevant_doc_ids: Set of relevant document IDs

    Returns:
        loss: Scalar loss
    """
    # Create labels: 1.0 if doc is relevant, 0.0 otherwise
    device = final_scores.device
    k = final_scores.size(1)
    top_k_ids = top_k_indices[0].tolist()
    
    labels = torch.zeros((1, k), device=device)
    for i, doc_id in enumerate(top_k_ids):
        if doc_id in relevant_doc_ids:
            labels[0, i] = 1.0
            
    # BCEWithLogitsLoss combines Sigmoid layer and the BCELoss in one single class.
    # This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(final_scores, labels)

    return loss


def train_rag_model(model: RAGModel, corpus: DocumentCorpus, train_examples: List[RAGExample],
                    num_epochs: int = 3, learning_rate: float = 0.001, k: int = 100, batch_size: int = 16):
    """Train the RAG model with mini-batching."""

    # Diagnostic: Check which parameters are trainable
    print("\n=== Parameter Diagnostic ===")
    embedder_trainable = sum(p.numel() for p in model.embedder.parameters() if p.requires_grad)
    embedder_total = sum(p.numel() for p in model.embedder.parameters())
    reranker_trainable = sum(p.numel() for p in model.reranker.parameters() if p.requires_grad)
    reranker_total = sum(p.numel() for p in model.reranker.parameters())

    print(f"Embedder: {embedder_trainable:,}/{embedder_total:,} trainable weights ({100*embedder_trainable/embedder_total:.1f}%)")
    print(f"Reranker: {reranker_trainable:,}/{reranker_total:,} trainable weights ({100*reranker_trainable/reranker_total:.1f}%)")

    if embedder_trainable < embedder_total:
        print(f"⚠️  WARNING: {embedder_total - embedder_trainable:,} embedder weights are FROZEN!")
    if reranker_trainable < reranker_total:
        print(f"⚠️  WARNING: {reranker_total - reranker_trainable:,} reranker weights are FROZEN!")
    print()

    # Optimizer now trains ALL parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # === TRAINING PHASE ===
        model.train()
        total_loss = 0.0

        # Track gradient norms
        embedder_grad_norms = []
        reranker_grad_norms = []

        # Mini-batch processing
        num_batches = (len(train_examples) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(train_examples))
            batch_examples = train_examples[batch_start:batch_end]

            optimizer.zero_grad()
            batch_loss = 0.0

            # Process each example in the batch
            for i, example in enumerate(batch_examples):
                # Forward pass
                top_k_indices, final_scores, initial_indices, initial_scores = model(example.query, corpus, k=k)

                # Compute loss
                loss = ranking_loss_sparse(final_scores, top_k_indices, example.relevant_doc_ids)

                # Accumulate loss (normalize by batch size)
                batch_loss += loss / len(batch_examples)

            # Backward pass on accumulated batch loss
            batch_loss.backward()

            # Check gradients before optimizer step
            embedder_grad_norm = 0.0
            reranker_grad_norm = 0.0
            embedder_weights_with_grad = 0
            reranker_weights_with_grad = 0
            embedder_total_weights = 0
            reranker_total_weights = 0

            for param in model.embedder.parameters():
                embedder_total_weights += param.numel()
                if param.grad is not None:
                    embedder_grad_norm += param.grad.norm().item() ** 2
                    embedder_weights_with_grad += param.numel()

            for param in model.reranker.parameters():
                reranker_total_weights += param.numel()
                if param.grad is not None:
                    reranker_grad_norm += param.grad.norm().item() ** 2
                    reranker_weights_with_grad += param.numel()

            embedder_grad_norm = embedder_grad_norm ** 0.5
            reranker_grad_norm = reranker_grad_norm ** 0.5

            embedder_grad_norms.append(embedder_grad_norm)
            reranker_grad_norms.append(reranker_grad_norm)

            optimizer.step()

            total_loss += batch_loss.item() * len(batch_examples)

            if batch_idx % max(1, num_batches // 5) == 0:
                print(f"  Batch {batch_idx+1}/{num_batches} - Loss: {batch_loss.item():.4f} | "
                      f"Embedder grad: {embedder_grad_norm:.4f} ({embedder_weights_with_grad:,}/{embedder_total_weights:,} weights) | "
                      f"Reranker grad: {reranker_grad_norm:.4f} ({reranker_weights_with_grad:,}/{reranker_total_weights:,} weights)")

        avg_loss = total_loss / len(train_examples)

        # === EVALUATION PHASE (after each epoch) ===
        model.eval()
        num_relevant_retrieved = 0
        num_relevant_retrieved_initial = 0
        total_relevant = 0

        with torch.no_grad():
            for example in train_examples:
                top_k_indices, final_scores, initial_indices, _ = model(example.query, corpus, k=k)

                # Track retrieval stats (Final)
                top_k_ids = set(top_k_indices[0].tolist())
                num_relevant_retrieved += len(top_k_ids & example.relevant_doc_ids)

                # Track retrieval stats (Initial)
                initial_ids = set(initial_indices[0].tolist())
                num_relevant_retrieved_initial += len(initial_ids & example.relevant_doc_ids)
                total_relevant += len(example.relevant_doc_ids)

        recall_at_k = num_relevant_retrieved / total_relevant if total_relevant > 0 else 0
        recall_at_k_initial = num_relevant_retrieved_initial / total_relevant if total_relevant > 0 else 0

        # Note: Recall is now evaluated AFTER the epoch with model.eval()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Recall@{k} (1st Stage): {recall_at_k_initial:.4f}, Recall@{k} (Reranked): {recall_at_k:.4f}")

def evaluate_rag_model(model: RAGModel, corpus: DocumentCorpus, test_examples: List[RAGExample], k: int = 100):
    """Evaluate the RAG model."""
    model.eval()

    # Metrics accumulators
    metrics = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    initial_metrics = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}

    def calculate_single_example_metrics(retrieved_indices, relevant_set, k):
        m = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
        top_k_ids = set(retrieved_indices)
        
        tp = len(top_k_ids & relevant_set)
        m["precision"] = tp / k if k > 0 else 0.0
        m["recall"] = tp / len(relevant_set) if len(relevant_set) > 0 else 0.0
        
        # MRR
        for rank, doc_id in enumerate(retrieved_indices, 1):
            if doc_id in relevant_set:
                m["mrr"] = 1.0 / rank
                break
        
        # NDCG
        dcg = 0.0
        idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(relevant_set), k)))
        for rank, doc_id in enumerate(retrieved_indices[:k], 1):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(rank + 1)
        m["ndcg"] = dcg / idcg if idcg > 0 else 0.0
        return m

    with torch.no_grad():
        for i, example in enumerate(test_examples):
            final_indices, _, initial_indices_tensor, _ = model(example.query, corpus, k=k)
            
            # Get lists
            final_list = final_indices[0].tolist()
            initial_list = initial_indices_tensor[0].tolist()
            
            # Compute metrics for this example
            final_m = calculate_single_example_metrics(final_list, example.relevant_doc_ids, k)
            initial_m = calculate_single_example_metrics(initial_list, example.relevant_doc_ids, k)
            
            # Accumulate
            for key in metrics:
                metrics[key] += final_m[key]
                initial_metrics[key] += initial_m[key]

            if (i + 1) % 10 == 0:
                # Calculate running averages for display
                current_count = i + 1
                avg_recall = metrics["recall"] / current_count
                avg_ndcg = metrics["ndcg"] / current_count
                print(f"  Evaluating {i+1}/{len(test_examples)} - Recall@{k}: {avg_recall:.4f} - NDCG@{k}: {avg_ndcg:.4f}")

    # Final Averages
    num_examples = len(test_examples)
    for key in metrics:
        metrics[key] /= num_examples
        initial_metrics[key] /= num_examples

    print(f"\nEvaluation Results (k={k}):")
    print(f"{'Metric':<15} | {'First Stage':<15} | {'Reranked (Final)':<15}")
    print("-" * 50)
    print(f"{'Precision':<15} | {initial_metrics['precision']:.4f}          | {metrics['precision']:.4f}")
    print(f"{'Recall':<15} | {initial_metrics['recall']:.4f}          | {metrics['recall']:.4f}")
    print(f"{'MRR':<15} | {initial_metrics['mrr']:.4f}          | {metrics['mrr']:.4f}")
    print(f"{'NDCG':<15} | {initial_metrics['ndcg']:.4f}          | {metrics['ndcg']:.4f}")

    return metrics
