import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import os
import torch.optim as opt
import json
from datetime import datetime
import torch
import time

# include the default hyperparameters 
# random search hyperparameters
## ------------ choose best hyperparameters ------------
# Option B — Choose hyperparameters with the lowest mean inner-CV loss
# During nested CV, you already computed inner-CV scores for each hyperparameter set.
# You can aggregate and pick the overall best.

# train with the best hyperparameters
# Evaluate best model on the chromadb benchmark framework

# if we want to remove hyperparameters, remove SGD 



EPOCHS = 5
BATCH_SIZE = 64
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WARMUP_STEPS = 50
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0002

hyperparams = {
    "lr": [2e-5, 1e-4, 5e-4],
    "batch_size": [16, 64, 128, 192],
    "optimizer": ["AdamW", "RMSprop"],
    "num_pairs": [5, 25, 100],
    "pick_technique": [],
}


class EarlyStoppingException(Exception):
    """Raised when early stopping criteria are met."""

def generate_pairs(df: pd.DataFrame, pick_technique: str = "next", num_pairs: int = 5):
    """
    Generate pairs for contrastive learning.

    Args:
        df: DataFrame containing the data
        pick_technique: String indicating the pick technique to use (next, in_doc, cross_doc)
        num_pairs: Maximum number of pairs to generate for each document

    Returns:
        List of pairs
    """
    num_pairs = 1000000 if num_pairs is None else num_pairs
    
    pairs = []
    seen_pairs = set()  # Track pairs to avoid duplicates
    
    # Only sort for "next" technique which needs consecutive sentences
    if pick_technique == "next":
        all_docs = [(doc_id, group.sort_values("global_index").reset_index(drop=True)) 
                    for doc_id, group in df.groupby("doc_id")]
    else:
        all_docs = [(doc_id, group.reset_index(drop=True)) 
                    for doc_id, group in df.groupby("doc_id")]

    for doc_id, sents in all_docs:
        
        if pick_technique == "next":
            # Generate consecutive pairs, limited to num_pairs per document
            pair_count = 0
            for i in range(len(sents) - 1):
                if pair_count >= num_pairs:
                    break
                    
                row1 = sents.iloc[i]
                row2 = sents.iloc[i+1]
                
                # Create a unique identifier for the pair (order-independent)
                pair_key = tuple(sorted([row1["sentence"], row2["sentence"]]))
                
                if pair_key not in seen_pairs:
                    label = 1 if row1["paragraph_id"] == row2["paragraph_id"] else 0
                    pairs.append([row1["sentence"], row2["sentence"], label])
                    seen_pairs.add(pair_key)
                    pair_count += 1
        
        elif pick_technique == "in_doc" or pick_technique == "cross_doc":
            # Group sentences by paragraph within this document
            paragraphs = {}
            for idx, row in sents.iterrows():
                para_id = row["paragraph_id"]
                if para_id not in paragraphs:
                    paragraphs[para_id] = []
                paragraphs[para_id].append(row)
            
            # Sample positive pairs directly without creating all candidates
            positive_pairs = []
            positive_count = 0
            max_attempts = num_pairs * 10  # Limit attempts to avoid infinite loops
            attempts = 0
            
            # Get all paragraphs with at least 2 sentences
            valid_paragraphs = [(para_id, para_sents) for para_id, para_sents in paragraphs.items() if len(para_sents) >= 2]
            
            while positive_count < num_pairs and attempts < max_attempts and valid_paragraphs:
                attempts += 1
                
                # Pick a random paragraph
                para_id, para_sents = random.choice(valid_paragraphs)
                
                # Pick two random different sentences from that paragraph
                if len(para_sents) >= 2:
                    row1, row2 = random.sample(para_sents, 2)
                    pair_key = tuple(sorted([row1["sentence"], row2["sentence"]]))
                    
                    if pair_key not in seen_pairs:
                        positive_pairs.append((row1, row2, pair_key))
                        seen_pairs.add(pair_key)
                        positive_count += 1
            
            # Sample negative pairs directly
            negative_pairs = []
            negative_count = 0
            max_attempts = positive_count * 10
            attempts = 0
            
            if pick_technique == "in_doc":
                # In-document negatives: different paragraphs within same document
                para_ids = list(paragraphs.keys())
                
                if len(para_ids) >= 2:
                    while negative_count < positive_count and attempts < max_attempts:
                        attempts += 1
                        
                        # Pick two different paragraphs
                        para1_id, para2_id = random.sample(para_ids, 2)
                        
                        # Pick random sentence from each
                        row1 = random.choice(paragraphs[para1_id])
                        row2 = random.choice(paragraphs[para2_id])
                        
                        pair_key = tuple(sorted([row1["sentence"], row2["sentence"]]))
                        
                        if pair_key not in seen_pairs:
                            negative_pairs.append((row1, row2, pair_key))
                            seen_pairs.add(pair_key)
                            negative_count += 1
            
            elif pick_technique == "cross_doc":
                # Cross-document negatives: sentences from different documents
                if len(all_docs) > 1 and len(sents) > 0:
                    while negative_count < positive_count and attempts < max_attempts:
                        attempts += 1
                        
                        # Pick random sentence from current doc
                        current_row = sents.iloc[random.randint(0, len(sents) - 1)]
                        
                        # Pick random document from all_docs, skip if it's the current one
                        other_doc_id, other_sents = random.choice(all_docs)
                        if other_doc_id == doc_id:
                            continue  # Skip this iteration if we picked the same document
                        
                        if len(other_sents) > 0:
                            # Pick random sentence from other document
                            random_other_sent = other_sents.iloc[random.randint(0, len(other_sents) - 1)]
                            
                            pair_key = tuple(sorted([current_row["sentence"], random_other_sent["sentence"]]))
                            
                            if pair_key not in seen_pairs:
                                negative_pairs.append((current_row, random_other_sent, pair_key))
                                seen_pairs.add(pair_key)
                                negative_count += 1
            
            # Add positive pairs
            for row1, row2, pair_key in positive_pairs:
                pairs.append([row1["sentence"], row2["sentence"], 1])
            
            # Add negative pairs
            for row1, row2, pair_key in negative_pairs:
                pairs.append([row1["sentence"], row2["sentence"], 0])
        
        else:
            raise ValueError(f"Unknown pick_technique: {pick_technique}. Must be 'next', 'in_doc', or 'cross_doc'")
    
    return pairs

# ============================================================
#  Convert DF → {doc_id: {sentences, boundaries}}
# ============================================================
def df_to_eval_docs(df):
    """
    Convert your dataframe into a dict:
    {
      doc_id: {
         'sentences': [...],
         'boundaries': [0/1,...]
      }
    }
    """
    docs = {}

    for doc_id, group in df.groupby("doc_id"):
        g = group.sort_values("global_index")

        sentences = g["sentence"].tolist()
        paragraph_ids = g["paragraph_id"].tolist()

        if len(sentences) < 2:
            continue

        # gold boundaries (1 when paragraph changes)
        boundaries = []
        for i in range(len(paragraph_ids) - 1):
            if paragraph_ids[i] != paragraph_ids[i+1]:
                boundaries.append(1)
            else:
                boundaries.append(0)
        boundaries.append(1)  # last sentence closes a chunk

        docs[doc_id] = {"sentences": sentences, "boundaries": boundaries}

    return docs


# ============================================================
# ChromaDB boundary metrics: IoU, Recall, Precision
# ============================================================
def chroma_metrics(gold, pred):
    gold = np.asarray(gold)
    pred = np.asarray(pred)

    intersection = np.sum((gold == 1) & (pred == 1))
    union = np.sum((gold == 1) | (pred == 1))

    iou = intersection / union if union > 0 else 0.0
    recall = intersection / np.sum(gold == 1) if np.sum(gold == 1) > 0 else 0.0
    precision = intersection / np.sum(pred == 1) if np.sum(pred == 1) > 0 else 0.0

    return {
        "iou": float(iou),
        "recall": float(recall),
        "precision": float(precision),
        "intersection": int(intersection),
        "union": int(union),
        "gold_boundaries": int(np.sum(gold == 1)),
        "pred_boundaries": int(np.sum(pred == 1)),
    }


# ============================================================
# Predict boundaries from cosine similarity
# ============================================================
def predict_boundaries_from_embeddings(embeddings, method="median", value=None):
    """
    embeddings: np array (N, D)

    method options:
        - "median"
        - "mean"
        - "p10", "p20", "p30", "p40", ....
        - "fixed"  (requires value=some_number)

    Returns:
        pred_boundaries, threshold_value
    """

    emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = (emb[:-1] * emb[1:]).sum(axis=1)  # cosine similarities

    # Adaptive thresholds
    if method == "median":
        threshold = np.median(sims)

    elif method == "mean":
        threshold = float(np.mean(sims) - np.std(sims))  # mean - std

    elif method.startswith("p"):
        pct = int(method[1:])   # get number after p
        threshold = np.percentile(sims, pct)

    # Fixed manual threshold
    elif method == "fixed":
        if value is None:
            raise ValueError("For method='fixed', you must supply value=")
        threshold = float(value)

    else:
        raise ValueError(f"Unknown threshold method: {method}")

    # Predict: similarity < threshold => boundary
    pred_pairs = (sims < threshold).astype(int)

    pred_boundaries = list(pred_pairs) + [1] # +1 last sentence
    return pred_boundaries, threshold


# ============================================================
#     Evaluate all docs for a given embedding function
# ============================================================
def evaluate_docs(docs, embed_fn, name="model", threshold_method="median"):
    print(f"----- Evaluating: {name} -----")

    metrics_per_doc = []
    ious, recalls, precisions = [], [], []

    for doc_id, doc in docs.items():
        sentences = doc["sentences"]
        gold = doc["boundaries"]

        # --- embed ---
        embeddings = embed_fn(sentences)

        # --- predict boundaries ---
        pred_boundaries, threshold = predict_boundaries_from_embeddings(
            embeddings, method=threshold_method
        )

        # --- compute metrics ---
        m = chroma_metrics(gold, pred_boundaries)

        metrics_per_doc.append((doc_id, m))
        ious.append(m["iou"])
        recalls.append(m["recall"])
        precisions.append(m["precision"])

        # print(f"Doc {doc_id}: IoU={m['iou']:.3f}, Recall={m['recall']:.3f}, Precision={m['precision']:.3f}")

    summary = {
        "iou_mean": float(np.mean(ious)),
        "recall_mean": float(np.mean(recalls)),
        "precision_mean": float(np.mean(precisions)),
        "docs_evaluated": len(metrics_per_doc),
    }

    # print("\n------ Summary ------")
    print(summary)
    print()
    return metrics_per_doc, summary

def train_model(
    train_pairs,
    val_pairs,
    model: SentenceTransformer,
    optimizer_name="AdamW",
    lr=2e-5,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    early_stopping=EARLY_STOPPING_ENABLED,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
):
    train_examples = [InputExample(texts=[s1, s2], label=float(lb)) for (s1, s2, lb) in train_pairs]
    train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True, num_workers=8)

    val_examples = [InputExample(texts=[s1, s2], label=float(lb)) for (s1, s2, lb) in val_pairs]
    evaluator = EmbeddingSimilarityEvaluator(
        [ex.texts[0] for ex in val_examples],
        [ex.texts[1] for ex in val_examples],
        [ex.label for ex in val_examples],
        name="val",
        batch_size=128
    )

    # Select optimizer class
    if optimizer_name == "Muon":
        optimizer_class = opt.Muon
    elif optimizer_name == "AdamW":
        optimizer_class = opt.AdamW
    elif optimizer_name == "SGD":
        optimizer_class = opt.SGD
    elif optimizer_name == "RMSprop":
        optimizer_class = opt.RMSprop
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    train_loss = losses.ContrastiveLoss(model)

    fit_callback = None
    evaluations_without_improvement = 0
    best_score = -np.inf

    if early_stopping:
        if evaluator is None:
            raise ValueError("Early stopping requires an evaluator to compute validation scores.")

        def _early_stopping_callback(score, epoch, steps):
            nonlocal evaluations_without_improvement, best_score
            if score is None:
                return

            score = float(score)
            if score > best_score + early_stopping_min_delta:
                best_score = score
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1

            if evaluations_without_improvement >= early_stopping_patience :
                raise EarlyStoppingException(
                    f"Early stopping triggered after {evaluations_without_improvement} "
                    f"evaluations without improvement (best score: {best_score:.4f})."
                )

        fit_callback = _early_stopping_callback

    fit_kwargs = {
        "train_objectives": [(train_dataloader, train_loss)],
        "evaluator": evaluator,
        "evaluation_steps": int(len(train_dataloader) * 0.2), # evaluate every 20% of the training data
        # "logging_steps": min(int(len(train_dataloader) * 0.05), 10),
        "epochs": epochs,
        "warmup_steps": warmup_steps,
        "optimizer_class": optimizer_class,
        "optimizer_params": {'lr': lr},
        "show_progress_bar": False,
        "output_path": None,
        "save_best_model": False,
    }
    if fit_callback is not None:
        fit_kwargs["callback"] = fit_callback

    try:
        model.fit(**fit_kwargs)
    except EarlyStoppingException as exc:
        print(str(exc))
    return model

def train_evaluate_model(df_train, df_val, hyperparams):
    # Generate pairs using hyperparameters
    start_time = time.time()
    train_pairs = generate_pairs(df_train, pick_technique=hyperparams['pick_technique'], num_pairs=hyperparams['num_pairs'])
    val_pairs = generate_pairs(df_val, pick_technique=hyperparams['pick_technique'], num_pairs=hyperparams['num_pairs'])
    elapsed_time = time.time() - start_time
    print(f"    Generating pairs completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Train model with current hyperparameters
    model = SentenceTransformer(MODEL_NAME)
    start_time = time.time()
    model = train_model(
        train_pairs, 
        val_pairs, 
        model,
        optimizer_name=hyperparams['optimizer'],
        lr=hyperparams['lr'],
        epochs=EPOCHS,
        batch_size=hyperparams['batch_size'],
        warmup_steps=WARMUP_STEPS
    )
    elapsed_time = time.time() - start_time
    print(f"    Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Evaluate on inner validation set
    docs_val = df_to_eval_docs(df_val)
    embed_fn = lambda sents: model.encode(sents, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    _, summary = evaluate_docs(docs_val, embed_fn, 
                                name="Inner Fold", 
                                threshold_method="mean")
    
    # Clean up memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return summary

def nested_cross_validation(data, n_outer_folds, n_inner_folds, name, cv_seed):
    """
    Nested cross-validation:
    - Outer CV: for robust performance estimation
    - Inner CV: for hyperparameter tuning
    
    Args:
        data: Training data dataframe
        n_outer_folds: Number of outer CV folds
        n_inner_folds: Number of inner CV folds
    
    Returns:
        outer_results: List of results for each outer fold
        results_dict: Dictionary containing all nested CV results and hyperparameter search results
    """
    global hyperparam_grid
    
    # Create results directory if it doesn't exist
    results_dir = "results/cv"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up file paths
    intermediate_file = os.path.join(results_dir, f"hyperparameter_tuning_cv_{name}_intermediate.json")
    final_file = os.path.join(results_dir, f"hyperparameter_tuning_cv_{name}.json")
    
    # Get unique document IDs for stratified splitting
    doc_ids = data['doc_id'].unique()
    
    outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=cv_seed)
    outer_results = []
    # Store comprehensive results for JSON output
    results_dict = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "n_outer_folds": n_outer_folds,
            "n_inner_folds": n_inner_folds,
            "n_random_samples_per_fold": 10,
            "hyperparameter_grid_size": len(hyperparam_grid),
            "hyperparameter_grid": hyperparam_grid,
            "train_docs": len(doc_ids)
        },
        "outer_folds": [],
        "summary": {},
        "final_test": {}
    }
    
    print(f"\n{'='*80}")
    print(f"NESTED CROSS-VALIDATION: {n_outer_folds} outer folds, {n_inner_folds} inner folds")
    print(f"Total hyperparameter combinations: {len(hyperparam_grid)}")
    print(f"{'='*80}\n")
    
    # ========== OUTER CV LOOP ==========
    for outer_fold, (train_doc_idx, test_doc_idx) in enumerate(outer_cv.split(doc_ids), 1):
        print(f"\n{'='*80}")
        print(f"OUTER FOLD {outer_fold}/{n_outer_folds}")
        print(f"{'='*80}")
        
        train_docs = doc_ids[train_doc_idx]
        test_docs = doc_ids[test_doc_idx]
        
        df_outer_train = data[data['doc_id'].isin(train_docs)]
        df_outer_test = data[data['doc_id'].isin(test_docs)]
        
        print(f"Outer train docs: {len(train_docs)}, Outer test docs: {len(test_docs)}")
        
        # Store fold info
        fold_info = {
            "fold": outer_fold,
            "train_docs": int(len(train_docs)),
            "test_docs": int(len(test_docs)),
            "hyperparameter_search": [],
            "sampled_hyperparams": [],
            "best_hyperparams": None,
            "best_inner_score": None,
            "outer_test_results": {}
        }
        
        # ========== INNER CV LOOP (Hyperparameter Tuning) ==========
        print(f"\n--- Inner CV: Hyperparameter Tuning ---")
        inner_cv = KFold(n_splits=n_inner_folds, shuffle=True)
        best_score = -np.inf
        best_hyperparams = None
        
        # Random search: sample 10 hyperparameter sets without replacement for this outer fold
        n_random_samples = min(10, len(hyperparam_grid))
        sampled_hyperparams = random.sample(hyperparam_grid, n_random_samples)
        fold_info["sampled_hyperparams"] = sampled_hyperparams
        
        for hp_idx, hyperparams in enumerate(sampled_hyperparams, 1):
            print(f"\nTesting hyperparams {hp_idx}/{n_random_samples}: {hyperparams}")
            inner_scores = []
            inner_fold_results = []
            
            start_time = time.time()
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(train_docs), 1):
                inner_train_docs = train_docs[inner_train_idx]
                inner_val_docs = train_docs[inner_val_idx]
                
                df_inner_train = data[data['doc_id'].isin(inner_train_docs)]
                df_inner_val = data[data['doc_id'].isin(inner_val_docs)]
                
                print(f"\n===== Inner fold {inner_fold} ======")

                summary = train_evaluate_model(df_inner_train, df_inner_val, hyperparams)
                
                inner_scores.append(summary['iou_mean'])
                inner_fold_results.append({
                    "inner_fold": inner_fold,
                    "iou": float(summary['iou_mean']),
                    "recall": float(summary['recall_mean']),
                    "precision": float(summary['precision_mean'])
                })
            
            elapsed_time = time.time() - start_time
            print(f"    Inner fold {inner_fold} completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

            # Average score across inner folds
            avg_inner_score = np.mean(inner_scores)
            print(f"  Average IoU for hyperparams {hyperparams}: {avg_inner_score:.4f}")
            
            # Store hyperparameter search results
            fold_info["hyperparameter_search"].append({
                "hyperparams": hyperparams,
                "inner_fold_results": inner_fold_results,
                "mean_iou": float(avg_inner_score),
                "std_iou": float(np.std(inner_scores))
            })
            
            if avg_inner_score > best_score:
                best_score = avg_inner_score
                best_hyperparams = hyperparams
        
        fold_info["best_hyperparams"] = best_hyperparams
        fold_info["best_inner_score"] = float(best_score)
        
        print(f"\nBest hyperparameters for outer fold {outer_fold}: {best_hyperparams}")
        print(f"  with average IoU: {best_score:.4f}")
        
        # ========== TRAIN ON FULL OUTER TRAIN WITH BEST HYPERPARAMS ==========
        print(f"\n--- Training on full outer train set with best hyperparameters ---")
        
        summary_outer = train_evaluate_model(df_outer_train, df_outer_test, best_hyperparams)

        outer_results.append({
            'fold': outer_fold,
            'best_hyperparams': best_hyperparams,
            'iou': summary_outer['iou_mean'],
            'recall': summary_outer['recall_mean'],
            'precision': summary_outer['precision_mean']
        })
        
        fold_info["outer_test_results"] = {
            "iou": float(summary_outer['iou_mean']),
            "recall": float(summary_outer['recall_mean']),
            "precision": float(summary_outer['precision_mean']),
            "docs_evaluated": int(summary_outer['docs_evaluated'])
        }
        
        results_dict["outer_folds"].append(fold_info)
        
        # Calculate and update summary stats with current folds
        current_ious = [r['iou'] for r in outer_results]
        current_recalls = [r['recall'] for r in outer_results]
        current_precisions = [r['precision'] for r in outer_results]
        
        if len(current_ious) > 0:
            results_dict["summary"] = {
                "iou": {
                    "mean": float(np.mean(current_ious)),
                    "std": float(np.std(current_ious)),
                    "values": [float(x) for x in current_ious]
                },
                "recall": {
                    "mean": float(np.mean(current_recalls)),
                    "std": float(np.std(current_recalls)),
                    "values": [float(x) for x in current_recalls]
                },
                "precision": {
                    "mean": float(np.mean(current_precisions)),
                    "std": float(np.std(current_precisions)),
                    "values": [float(x) for x in current_precisions]
                }
            }
        
        # Save intermediate results after each outer fold
        with open(intermediate_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nIntermediate results saved to {intermediate_file}")
        
        print(f"\nOuter Fold {outer_fold} Results:")
        print(f"  IoU: {summary_outer['iou_mean']:.4f}")
        print(f"  Recall: {summary_outer['recall_mean']:.4f}")
        print(f"  Precision: {summary_outer['precision_mean']:.4f}")
    
    # ========== SUMMARY OF ALL OUTER FOLDS ==========
    print(f"\n{'='*80}")
    print(f"NESTED CV SUMMARY (Across {n_outer_folds} outer folds)")
    print(f"{'='*80}")
    
    ious = [r['iou'] for r in outer_results]
    recalls = [r['recall'] for r in outer_results]
    precisions = [r['precision'] for r in outer_results]
    
    summary_stats = {
        "iou": {
            "mean": float(np.mean(ious)),
            "std": float(np.std(ious)),
            "values": [float(x) for x in ious]
        },
        "recall": {
            "mean": float(np.mean(recalls)),
            "std": float(np.std(recalls)),
            "values": [float(x) for x in recalls]
        },
        "precision": {
            "mean": float(np.mean(precisions)),
            "std": float(np.std(precisions)),
            "values": [float(x) for x in precisions]
        }
    }
    
    results_dict["summary"] = summary_stats
    
    print(f"\nMean IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Mean Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"Mean Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    
    print("\nPer-fold results:")
    for r in outer_results:
        print(f"  Fold {r['fold']}: IoU={r['iou']:.4f}, Recall={r['recall']:.4f}, "
              f"Precision={r['precision']:.4f}, Hyperparams={r['best_hyperparams']}")
    
    # Save final results to JSON file
    print(f"\n{'='*80}")
    print(f"Saving final results to {final_file}")
    print(f"{'='*80}")
    
    with open(final_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Final results saved successfully to {final_file}")
    
    return outer_results, results_dict


if __name__ == "__main__":
    # read the pick techniques from terminal args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick_technique", type=str, default="next", choices=["next", "in_doc", "cross_doc"])
    parser.add_argument("--cv_seed", type=int, default=42) # mainly so I can parallelize the runs
    args = parser.parse_args()
    pick_technique = args.pick_technique
    hyperparams["pick_technique"].append(pick_technique)

    hyperparam_grid = []
    for lr in hyperparams["lr"]:
        for batch_size in hyperparams["batch_size"]:
            for num_pairs in hyperparams["num_pairs"]:
                for pick_technique in hyperparams["pick_technique"]:
                    for optimizer in hyperparams["optimizer"]:
                        if pick_technique == "next":
                            num_pairs = 1000000
                        new_set = {
                            "lr": lr,
                            "batch_size": batch_size,
                            "num_pairs": num_pairs,
                            "pick_technique": pick_technique,
                            "optimizer": optimizer
                        }
                        if new_set not in hyperparam_grid:
                            hyperparam_grid.append(new_set)
    print(f"Hyperparameter grid: {hyperparam_grid}")

    df_train = pd.read_csv('data/train_flattened_10000.csv')
    # df_val = pd.read_csv('data/val_flattened_5000.csv')
    # df_test = pd.read_csv('data/test_flattened_5000.csv')
    
    print(f"Original dataset sizes:")
    print(f"  Training documents: {df_train['doc_id'].nunique()}")
    # print(f"  Validation documents: {df_val['doc_id'].nunique()}")
    # print(f"  Test documents: {df_test['doc_id'].nunique()}")
    
    # Combine train and validation sets
    # df_combined = pd.concat([df_train, df_val], ignore_index=True)
    # print(f"\nCombined train+val documents: {df_combined['doc_id'].nunique()}")
    
    # Run nested cross-validation
    outer_results, results_dict = nested_cross_validation(
        df_train,    # df_combined,
        n_outer_folds=5,  # Outer CV for robust estimation
        n_inner_folds=5,   # Inner CV for hyperparameter tuning
        name=f"pick_{pick_technique}",
        cv_seed=args.cv_seed
    )
    