from .data import load_multihop_rag_dataset
from .models import RAGModel
from .training import train_rag_model, evaluate_rag_model

def main():
    """Main benchmark function."""
    print("=== RAG Benchmark with Full Corpus Retrieval ===\n")

    # Hyperparameters
    max_examples = 1000  # Reduced for faster debugging
    train_ratio = 0.8
    # k will be set dynamically
    num_epochs = 5
    batch_size = 16
    learning_rate = 0.001
    regularization_strength = 0.1

    # Load dataset and build corpus
    corpus, all_examples = load_multihop_rag_dataset(split="train", max_examples=max_examples)

    # Calculate dynamic k
    max_relevant = 0
    for ex in all_examples:
        max_relevant = max(max_relevant, len(ex.relevant_doc_ids))
    
    k = max_relevant
    print(f"\n[Config] Max relevant docs per query: {max_relevant}")
    print(f"[Config] Setting k={k} for evaluation (Top-{k})")

    # print("\nSample Corpus Documents:")
    # print(corpus.documents[:5])
    # print(all_examples[:5])

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
        embedding_model="paraphrase-MiniLM-L3-v2",
        reranker_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        regularization_strength=regularization_strength
    )

    # Embed corpus
    corpus.embed_corpus(model.embedder)

    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_embedder_params = sum(p.numel() for p in model.embedder.parameters())
    total_reranker_params = sum(p.numel() for p in model.reranker.parameters())
    
    print(f"\nModel Parameters:")
    print(f"  - Embedder: {total_embedder_params:,}")
    print(f"  - Reranker: {total_reranker_params:,}")
    print(f"  - Total Trainable Parameters: {trainable_params:,}")
    print("  (Note: We are now training ALL parameters including Embedder and Reranker)")

    # Evaluate before training
    print(f"\n=== Before Training (k={k}) ===")
    evaluate_rag_model(model, corpus, test_examples, k=k)

    # Train
    print(f"\n=== Training (k={k}) ===")
    train_rag_model(
        model, corpus, train_examples,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        k=k,
        batch_size=batch_size
    )

    # Evaluate after training
    print(f"\n=== After Training (k={k}) ===")
    evaluate_rag_model(model, corpus, test_examples, k=k)

    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
