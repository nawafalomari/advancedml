#!/usr/bin/env python3
"""
Script to train a contrastive learning model and save its weights.
Imports training functions from contrastive_learning.py
"""

import pandas as pd
import argparse
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch

# Import functions from contrastive_learning.py
from contrastive_learning import (
    generate_pairs,
    train_model,
    evaluate_docs,
    df_to_eval_docs,
    MODEL_NAME,
    EPOCHS,
    BATCH_SIZE,
    WARMUP_STEPS,
)


def train_and_save_model(
    train_data_path,
    val_data_path,
    test_data_path,
    output_dir="models",
    model_name=MODEL_NAME,
    pick_technique="next",
    num_pairs=5,
    lr=2e-5,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    optimizer="AdamW",
    warmup_steps=WARMUP_STEPS,
    threshold_method="median",
):
    """
    Train a contrastive learning model and save its weights.
    
    Args:
        train_data_path: Path to training CSV file
        val_data_path: Path to validation CSV file (optional, if None, will use train data)
        test_data_path: Path to test CSV file (optional, if None, will use train data)
        output_dir: Directory to save the trained model
        model_name: Base model name to use
        pick_technique: Technique for generating pairs ("next", "in_doc", "cross_doc")
        num_pairs: Number of pairs to generate per document
        lr: Learning rate
        batch_size: Batch size for training
        epochs: Number of training epochs
        optimizer: Optimizer name ("AdamW", "RMSprop", "SGD")
        warmup_steps: Number of warmup steps
        threshold_method: Threshold method for boundary prediction ("median", "mean", "p10", etc.)
    
    Returns:
        Tuple of (model_save_path, evaluation_summary)
    """
    print("=" * 80)
    print("TRAINING CONTRASTIVE LEARNING MODEL")
    print("=" * 80)
    
    # Load training data
    print(f"\nLoading training data from {train_data_path}...")
    df_train = pd.read_csv(train_data_path)
    print(f"  Training documents: {df_train['doc_id'].nunique()}")
    print(f"  Training sentences: {len(df_train)}")
    
    # Load validation and test data
    print(f"\nLoading validation data from {val_data_path}...")
    df_val = pd.read_csv(val_data_path)
    print(f"  Validation documents: {df_val['doc_id'].nunique()}")
    print(f"  Validation sentences: {len(df_val)}")

    print(f"\nLoading test data from {test_data_path}...")
    df_test = pd.read_csv(test_data_path)
    print(f"  Test documents: {df_test['doc_id'].nunique()}")
    print(f"  Test sentences: {len(df_test)}")
    
    #convert all sentences to strings
    df_train["sentence"] = df_train["sentence"].astype(str)
    df_val["sentence"] = df_val["sentence"].astype(str)
    df_test["sentence"] = df_test["sentence"].astype(str)
    
    train_pairs = generate_pairs(df_train, pick_technique=pick_technique, num_pairs=num_pairs)
    val_pairs = generate_pairs(df_val, pick_technique=pick_technique, num_pairs=num_pairs)
    print(f"  Generated {len(train_pairs)} training pairs")
    print(f"  Generated {len(val_pairs)} validation pairs")

    # Initialize model
    model = SentenceTransformer(model_name)
    
    # Train model
    print(f"\nTraining model with hyperparameters:")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Warmup steps: {warmup_steps}")
    
    model = train_model(
        model=model,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        optimizer_name=optimizer,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        early_stopping=False,
    )
    
    # Evaluate model on validation data
    print(f"\n{'='*80}")
    print("EVALUATING MODEL")
    print("="*80)
    
    docs_test = df_to_eval_docs(df_test)
    embed_fn = lambda sents: model.encode(sents, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    
    metrics_per_doc, summary = evaluate_docs(
        docs_test, 
        embed_fn, 
        name="Test Set",
        threshold_method=threshold_method
    )
    
    # Print detailed results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Threshold method: {threshold_method}")
    print(f"Documents evaluated: {summary['docs_evaluated']}")
    print(f"\nMean Metrics:")
    print(f"  IoU:      {summary['iou_mean']:.4f}")
    print(f"  Recall:   {summary['recall_mean']:.4f}")
    print(f"  Precision: {summary['precision_mean']:.4f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate model save path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_name = f"model_{df_val['doc_id'].nunique()}_{timestamp}"
    model_save_path = os.path.join(output_dir, model_save_name)
    
    # Save model
    print(f"\nSaving model to {model_save_path}...")
    model.save(model_save_path)
    
    print(f"\nModel saved successfully!")
    print(f"  Path: {model_save_path}")
    
    return model_save_path, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a contrastive learning model and save its weights")
    
    # Data paths
    parser.add_argument("--train_data", type=str,
                       help="Path to training CSV file")
    parser.add_argument("--val_data", type=str, 
                       help="Path to validation CSV file")
    parser.add_argument("--test_data", type=str,
                       help="Path to test CSV file")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                       help="Base model name to use")
    parser.add_argument("--output_dir", type=str, default="models",
                       help="Directory to save the trained model")
    
    # Hyperparameters
    parser.add_argument("--pick_technique", type=str, default="next",
                       choices=["next", "in_doc", "cross_doc"],
                       help="Technique for generating pairs")
    parser.add_argument("--num_pairs", type=int, default=5,
                       help="Number of pairs to generate per document")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                       choices=["AdamW", "RMSprop", "SGD"],
                       help="Optimizer to use")
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS,
                       help="Number of warmup steps")
    parser.add_argument("--threshold_method", type=str, default="mean",
                       choices=["median", "mean", "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"],
                       help="Threshold method for boundary prediction")
    
    args = parser.parse_args()
    
    # Train and save model
    model_path, eval_results = train_and_save_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pick_technique=args.pick_technique,
        num_pairs=args.num_pairs,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        optimizer=args.optimizer,
        warmup_steps=args.warmup_steps,
        threshold_method=args.threshold_method,
    )
    

