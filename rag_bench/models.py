import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from fast_soft_sort import pytorch_ops
from .data import DocumentCorpus

class EmbeddingModel(nn.Module):
    """Embedding model using Sentence-Transformers."""

    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2"):
        super().__init__()
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Embed texts using sentence-transformers (differentiable version)."""
        # Tokenize the texts
        features = self.model.tokenize(texts)

        # Move to device
        device = next(self.model.parameters()).device
        features = {k: v.to(device) for k, v in features.items()}

        # Access the underlying transformer model directly to preserve gradients
        transformer = self.model[0].auto_model

        # Forward through transformer
        model_output = transformer(**features, return_dict=True)

        # Mean pooling
        token_embeddings = model_output[0]
        attention_mask = features['attention_mask']

        # Mean pooling - take attention mask into account
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

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
            soft_ranks: (batch_size, num_docs) - Soft ranks (1 = best)
        """
        # Normalize for cosine similarity
        query_embed = F.normalize(query_embed, p=2, dim=-1)
        corpus_embeds = F.normalize(corpus_embeds, p=2, dim=-1)

        # Compute similarities: (batch_size, num_docs)
        similarities = torch.mm(query_embed, corpus_embeds.t())

        # Use soft sorting to get differentiable top-k
        soft_ranks = pytorch_ops.soft_rank(
            similarities,
            direction="DESCENDING",
            regularization_strength=self.regularization_strength,
            regularization="l2"
        )

        return similarities, soft_ranks


class CrossEncoderReRanker(nn.Module):
    """Reranker using cross-encoder with learnable refinement."""

    def __init__(self, cross_encoder_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"):
        super().__init__()
        print(f"Loading cross-encoder: {cross_encoder_name}...")
        self.cross_encoder = CrossEncoder(cross_encoder_name)

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

        # Tokenize
        features = self.cross_encoder.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.cross_encoder.model.device)

        # Forward pass through the model (AutoModelForSequenceClassification)
        # This preserves gradients!
        outputs = self.cross_encoder.model(**features)
        logits = outputs.logits
        
        # Squeeze if necessary (depending on num_labels, usually 1 for regression)
        cross_scores = logits.squeeze(-1).unsqueeze(0)

        # Residual connection with initial scores
        final_scores = cross_scores + 0.1 * initial_scores

        return final_scores


class RAGModel(nn.Module):
    """Complete RAG model with corpus retrieval."""

    def __init__(self,
                 embedding_model: str = "paraphrase-MiniLM-L3-v2",
                 reranker_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                 regularization_strength: float = 0.1):
        super().__init__()
        self.embedder = EmbeddingModel(model_name=embedding_model)
        self.retriever = FirstStageRetriever(regularization_strength=regularization_strength)
        self.reranker = CrossEncoderReRanker(cross_encoder_name=reranker_model)
        self.regularization_strength = regularization_strength

    def forward(self, query: str, corpus: DocumentCorpus, k: int = 5, k_retrieval: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full retrieval pipeline with soft top-k selection.

        Args:
            query: Query string
            corpus: Document corpus with embeddings
            k: Number of documents to return (final top-k)
            k_retrieval: Number of documents to retrieve in first stage

        Returns:
            top_k_indices: (1, k) - Global corpus IDs of final top-k docs (after reranking)
            final_scores: (1, k) - Final reranking scores for top-k docs
            initial_top_k_indices: (1, k) - Global corpus IDs of top-k docs (before reranking)
            initial_top_k_scores: (1, k) - Scores from first stage for top-k docs
        """
        # Embed query
        query_embed = self.embedder([query])

        # First-stage retrieval with soft sorting (differentiable)
        # We retrieve k_retrieval (50) documents first
        similarities, soft_ranks = self.retriever(
            query_embed, corpus.embeddings, k=k_retrieval
        )

        # --- Intermediate: Get top-k from first stage directly (for comparison) ---
        initial_top_k_indices = torch.argsort(soft_ranks[0])[:k].unsqueeze(0)
        initial_top_k_scores = torch.gather(similarities, 1, initial_top_k_indices)

        # --- Reranking Pipeline ---
        # Select top-50 based on soft ranks for reranking
        top_retrieval_indices = torch.argsort(soft_ranks[0])[:k_retrieval].unsqueeze(0)

        # Get scores for these 50 from original similarities
        top_retrieval_initial_scores = torch.gather(similarities, 1, top_retrieval_indices)

        # Get documents for reranking
        top_retrieval_docs = [corpus.documents[idx] for idx in top_retrieval_indices[0]]

        # Reranking with cross-encoder on the top-50
        reranked_scores = self.reranker(query, top_retrieval_docs, top_retrieval_initial_scores)

        # Select final top-k from the reranked 50
        top_k_in_retrieved_indices = torch.argsort(reranked_scores[0], descending=True)[:k]

        # Map back to global corpus indices
        final_top_k_indices = top_retrieval_indices[0][top_k_in_retrieved_indices].unsqueeze(0)

        # Get final scores for top-k
        final_top_k_scores = reranked_scores[0][top_k_in_retrieved_indices].unsqueeze(0)

        return final_top_k_indices, final_top_k_scores, initial_top_k_indices, initial_top_k_scores
