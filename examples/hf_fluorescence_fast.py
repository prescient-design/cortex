"""
Fast example using TAPE Fluorescence dataset with a tiny model.

This example demonstrates HuggingFace integration but uses:
- A tiny BERT model (5M params) instead of ProtBERT (420M params)
- Only 500 training samples
- 1 epoch
- Should complete in <60 seconds

Now with torch.compile support!
"""

import argparse
import time

import hydra
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cortex.config import NeuralTreeConfig
from cortex.model.leaf import RegressorLeaf
from cortex.model.root import HuggingFaceRoot
from cortex.model.tree import NeuralTreeLightningV2


def prepare_protein_data():
    """Load and prepare TAPE Fluorescence dataset from HuggingFace."""
    # Load dataset
    print("Loading TAPE Fluorescence dataset from HuggingFace...")
    dataset = load_dataset(
        "InstaDeepAI/true-cds-protein-tasks",
        name="fluorescence",
        trust_remote_code=True,
    )

    print("  Using subset: 500 train, 200 validation samples")

    # Initialize a small tokenizer (using bert-tiny for speed)
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    def tokenize_function(examples):
        # Space out amino acids for better tokenization
        # Convert "MKTVRQ..." to "M K T V R Q ..."
        spaced_sequences = [" ".join(seq) for seq in examples["sequence"]]

        return tokenizer(
            spaced_sequences,
            padding="max_length",
            truncation=True,
            max_length=256,  # Protein sequences need more tokens when spaced
            return_tensors="pt",
        )

    # Apply tokenization to small subsets
    train_subset = dataset["train"].select(range(500))
    val_subset = dataset["validation"].select(range(200))

    tokenized_train = train_subset.map(tokenize_function, batched=True, remove_columns=["sequence"])

    tokenized_val = val_subset.map(tokenize_function, batched=True, remove_columns=["sequence"])

    # Set format for PyTorch
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    return tokenized_train, tokenized_val


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fast TAPE Fluorescence Example")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--backend",
        type=str,
        default="inductor",
        choices=["inductor", "cudagraphs", "aot_eager", "eager"],
        help="torch.compile backend",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    args = parser.parse_args()

    print("=== Fast TAPE Fluorescence Example ===")
    print(f"torch.compile: {'ENABLED' if args.compile else 'DISABLED'}")
    if args.compile:
        print(f"  Backend: {args.backend}")
        print(f"  Mode: {args.mode}")
    print()

    # 1. Load and prepare data
    train_dataset, val_dataset = prepare_protein_data()

    # 2. Create configuration with tiny model
    print("\n2. Creating NeuralTree configuration with tiny BERT...")
    config = NeuralTreeConfig()

    # Add tiny BERT model (only 4.4M parameters)
    config.add_hf_root("protein", model_name_or_path="prajjwal1/bert-tiny")

    # Small architecture for speed
    config.trunk = {
        "_target_": "cortex.model.trunk.SumTrunk",
        "in_dims": [128],  # bert-tiny hidden size
        "out_dim": 64,
        "project_features": True,
    }

    config.branches["fluorescence_branch"] = {
        "_target_": "cortex.model.branch.TransformerBranch",
        "in_dim": 64,
        "out_dim": 32,
        "num_blocks": 1,  # Single block
        "num_heads": 4,
        "channel_dim": 64,
        "dropout_p": 0.1,
    }

    # 3. Initialize model components
    print("3. Initializing Neural Tree components...")

    # Create root node
    root_nodes = nn.ModuleDict(
        {
            "protein": HuggingFaceRoot(
                model_name_or_path="prajjwal1/bert-tiny",
                pooling_strategy="none",  # Return full sequence for Conv1d branches
            )
        }
    )

    # Create trunk node using hydra
    trunk_node = hydra.utils.instantiate(config.trunk)

    # Create branch nodes
    branch_nodes = nn.ModuleDict(
        {"fluorescence_branch": hydra.utils.instantiate(config.branches["fluorescence_branch"])}
    )

    # Create leaf nodes
    leaf_nodes = nn.ModuleDict(
        {
            "fluorescence": RegressorLeaf(
                branch_key="fluorescence_branch",
                in_dim=32,
                out_dim=1,
                num_layers=1,
            )
        }
    )

    # Create the tree model using NeuralTreeLightningV2
    model = NeuralTreeLightningV2(
        root_nodes=root_nodes,
        trunk_node=trunk_node,
        branch_nodes=branch_nodes,
        leaf_nodes=leaf_nodes,
    )

    print(f"   Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # 4. Create data loaders
    print("\n4. Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 5. Set up training
    print("\n5. Setting up training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Apply torch.compile if requested
    if args.compile:
        print(f"   Compiling model with backend={args.backend}, mode={args.mode}...")
        compile_start = time.time()
        model = torch.compile(model, backend=args.backend, mode=args.mode)
        print(f"   Compilation setup took {time.time() - compile_start:.2f}s")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # High LR for fast convergence
    criterion = nn.MSELoss()

    # 6. Quick training
    print("\n6. Training for 1 epoch...")
    model.train()

    total_loss = 0
    training_start = time.time()
    batch_times = []

    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].float().to(device)

        # Prepare inputs
        root_inputs = {"protein": {"input_ids": input_ids, "attention_mask": attention_mask}}

        # Forward pass
        outputs = model(root_inputs, leaf_keys=["fluorescence"])
        predictions = outputs.leaf_outputs["fluorescence"].loc.squeeze()

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_times.append(time.time() - batch_start)

        if batch_idx % 5 == 0:
            print(f"     Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Time: {batch_times[-1]:.3f}s")

    training_time = time.time() - training_start
    avg_loss = total_loss / len(train_loader)
    avg_batch_time = sum(batch_times) / len(batch_times)

    print(f"   Training Loss: {avg_loss:.4f}")
    print(f"   Total training time: {training_time:.2f}s")
    print(f"   Average batch time: {avg_batch_time:.3f}s")
    if args.compile and len(batch_times) > 5:
        # Skip first few batches for compilation overhead
        steady_state_avg = sum(batch_times[5:]) / len(batch_times[5:])
        print(f"   Steady-state batch time: {steady_state_avg:.3f}s")

    # 7. Quick validation
    print("\n7. Running validation...")
    model.eval()
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].float().to(device)

            root_inputs = {"protein": {"input_ids": input_ids, "attention_mask": attention_mask}}

            outputs = model(root_inputs, leaf_keys=["fluorescence"])
            predictions = outputs.leaf_outputs["fluorescence"].loc.squeeze()

            val_predictions.extend(predictions.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # Calculate correlation
    from scipy.stats import spearmanr

    val_correlation, _ = spearmanr(val_predictions, val_labels)

    print(f"   Validation Spearman ρ: {val_correlation:.4f}")

    print("\n✅ Fast example completed!")
    print("   - Used tiny BERT model (4.4M params vs 420M)")
    print("   - Trained on 500 samples for 1 epoch")
    print("   - Demonstrates HuggingFace dataset integration")
    if args.compile:
        print(f"   - torch.compile: {args.backend} backend with {args.mode} mode")


if __name__ == "__main__":
    main()
