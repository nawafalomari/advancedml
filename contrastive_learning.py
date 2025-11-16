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

EPOCHS = 5
BATCH_SIZE = 32
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WARMUP_STEPS = 50

def generate_pairs(df: pd.DataFrame, pick_technique: str = "next", num_pairs: int = 5):
    num_pairs = 1000000 if num_pairs is None else num_pairs
    
    pairs = []
    seen_pairs = set()  # Track pairs to avoid duplicates
    
    all_docs = [(doc_id, group) for doc_id, group in df.groupby("doc_id")]

    for doc_id, group in df.groupby("doc_id"):
        sents = group.sort_values("global_index").reset_index(drop=True)
        
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
            
            # Collect positive pairs (same paragraph)
            positive_candidates = []
            for para_id, para_sents in paragraphs.items():
                if len(para_sents) >= 2:
                    # Generate all possible pairs within this paragraph
                    for i in range(len(para_sents)):
                        for j in range(i + 1, len(para_sents)):
                            row1 = para_sents[i]
                            row2 = para_sents[j]
                            pair_key = tuple(sorted([row1["sentence"], row2["sentence"]]))
                            if pair_key not in seen_pairs:
                                positive_candidates.append((row1, row2, pair_key))
            
            negative_candidates = []
            if pick_technique == "in_doc":
                # Collect negative pairs (different paragraphs)
                para_ids = list(paragraphs.keys())
                for i in range(len(para_ids)):
                    for j in range(i + 1, len(para_ids)):
                        para1_id = para_ids[i]
                        para2_id = para_ids[j]
                        for row1 in paragraphs[para1_id]:
                            for row2 in paragraphs[para2_id]:
                                pair_key = tuple(sorted([row1["sentence"], row2["sentence"]]))
                                if pair_key not in seen_pairs:
                                    negative_candidates.append((row1, row2, pair_key))
            
            elif pick_technique == "cross_doc":
                # Collect negative pairs from cross-document sampling
                # Get all other documents (excluding current doc)
                other_docs = [(other_doc_id, other_group) for other_doc_id, other_group in all_docs if other_doc_id != doc_id]
                
                if len(other_docs) == 0:
                    pass
                else:
                    # Determine how many random docs to pick (limit to available docs)
                    n_docs_to_pick = len(positive_candidates)
                    
                    # Randomly sample n documents
                    selected_other_docs = random.choices(other_docs, k=n_docs_to_pick)
                    
                    # For each selected document, pick one random sentence
                    for other_doc_id, other_group in selected_other_docs:
                        other_sents = other_group.sort_values("global_index").reset_index(drop=True)
                        if len(other_sents) == 0:
                            continue
                        
                        # Pick one random sentence from this other document
                        random_other_sent = other_sents.iloc[random.randint(0, len(other_sents) - 1)]
                        
                        # Pair this random sentence with every sentence in the current document
                        for idx, current_row in sents.iterrows():
                            pair_key = tuple(sorted([current_row["sentence"], random_other_sent["sentence"]]))
                            if pair_key not in seen_pairs:
                                negative_candidates.append((current_row, random_other_sent, pair_key))

            # Randomly sample up to num_pairs positive pairs
            np.random.shuffle(positive_candidates)
            positive_selected = positive_candidates[:min(num_pairs, len(positive_candidates))]
            
            # Randomly sample up to num_pairs negative pairs
            np.random.shuffle(negative_candidates)
            negative_selected = negative_candidates[:min(num_pairs, len(negative_candidates))]
            
            # Add positive pairs
            for row1, row2, pair_key in positive_selected:
                pairs.append([row1["sentence"], row2["sentence"], 1])
                seen_pairs.add(pair_key)
            
            # Add negative pairs
            for row1, row2, pair_key in negative_selected:
                pairs.append([row1["sentence"], row2["sentence"], 0])
                seen_pairs.add(pair_key)
        
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
    print(f"\n========== Evaluating: {name} ==========")

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
    # print(summary)
    return metrics_per_doc, summary

def train_model(train_pairs, val_pairs, model, epochs=EPOCHS, batch_size=BATCH_SIZE, warmup_steps=WARMUP_STEPS):
    train_examples = [InputExample(texts=[s1, s2], label=float(lb)) for (s1, s2, lb) in train_pairs]
    train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)

    val_examples = [InputExample(texts=[s1, s2], label=float(lb)) for (s1, s2, lb) in val_pairs]
    evaluator = EmbeddingSimilarityEvaluator(
        [ex.texts[0] for ex in val_examples],
        [ex.texts[1] for ex in val_examples],
        [ex.label for ex in val_examples],
        name="val"
    )

    train_loss = losses.ContrastiveLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        evaluation_steps=100,
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True
    )
    return model

def train_evaluate_model(df_train, df_val, hyperparams):
    # Generate pairs
    train_pairs = generate_pairs(df_train)
    val_pairs = generate_pairs(df_val)
    
    # Train model with current hyperparameters
    model = SentenceTransformer(MODEL_NAME)
    model = train_model(
        train_pairs, 
        val_pairs, 
        model,
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        warmup_steps=WARMUP_STEPS
    )
    
    # Evaluate on inner validation set
    docs_val = df_to_eval_docs(df_val)
    embed_fn = lambda sents: model.encode(sents, convert_to_numpy=True)
    _, summary = evaluate_docs(docs_val, embed_fn, 
                                name="Inner Fold", 
                                threshold_method="median")
    return summary

def nested_cross_validation(data, df_test, n_outer_folds=5, n_inner_folds=3, output_file="nested_cv_results.json"):
    """
    Nested cross-validation:
    - Outer CV: for robust performance estimation
    - Inner CV: for hyperparameter tuning
    
    Args:
        data: Training data dataframe
        df_test: Test data dataframe
        n_outer_folds: Number of outer CV folds
        n_inner_folds: Number of inner CV folds
        output_file: Path to JSON file to save results
    
    Returns:
        outer_results: List of results for each outer fold
        summary_test: Final test set evaluation summary
    """
    
    #? Hyperparameter grid to search i dont know what else to add here
    # hyperparams = {
    #     "lr": [1e-5, 2e-5, 5e-5],
    #     "batch_size": [16, 32, 64],
    #     "epochs": [3,5],
    #     "num_pairs": [3, 5, 10, 25, 100],
    #     "pick_technique": ["next", "in_doc", "cross_doc"],
    # }
    hyperparams = {
        "lr": [1e-5],
        "batch_size": [64],
        "epochs": [1],
        "num_pairs": [1, 100, 1000000],
        "pick_technique": ["next", "in_doc", "cross_doc"],
    }

    hyperparam_grid = []
    for lr in hyperparams["lr"]:
        for batch_size in hyperparams["batch_size"]:
            for epochs in hyperparams["epochs"]:
                for num_pairs in hyperparams["num_pairs"]:
                    for pick_technique in hyperparams["pick_technique"]:
                        hyperparam_grid.append({
                            "lr": lr,
                            "batch_size": batch_size,
                            "epochs": epochs,
                            "num_pairs": num_pairs,
                            "pick_technique": pick_technique
                        })

    
    # Get unique document IDs for stratified splitting
    doc_ids = data['doc_id'].unique()
    
    outer_cv = KFold(n_splits=n_outer_folds, shuffle=True)
    outer_results = []
    
    # Store comprehensive results for JSON output
    results_dict = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "n_outer_folds": n_outer_folds,
            "n_inner_folds": n_inner_folds,
            "hyperparameter_grid": hyperparam_grid,
            "train_docs": len(doc_ids),
            "test_docs": df_test['doc_id'].nunique()
        },
        "outer_folds": [],
        "summary": {},
        "final_test": {}
    }
    
    print(f"\n{'='*80}")
    print(f"NESTED CROSS-VALIDATION: {n_outer_folds} outer folds, {n_inner_folds} inner folds")
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
            "best_hyperparams": None,
            "best_inner_score": None,
            "outer_test_results": {}
        }
        
        # ========== INNER CV LOOP (Hyperparameter Tuning) ==========
        print(f"\n--- Inner CV: Hyperparameter Tuning ---")
        inner_cv = KFold(n_splits=n_inner_folds, shuffle=True)
        best_score = -np.inf
        best_hyperparams = None
        
        for hp_idx, hyperparams in enumerate(hyperparam_grid, 1):
            print(f"\nTesting hyperparams {hp_idx}/{len(hyperparam_grid)}: {hyperparams}")
            inner_scores = []
            inner_fold_results = []
            
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(train_docs), 1):
                inner_train_docs = train_docs[inner_train_idx]
                inner_val_docs = train_docs[inner_val_idx]
                
                df_inner_train = data[data['doc_id'].isin(inner_train_docs)]
                df_inner_val = data[data['doc_id'].isin(inner_val_docs)]
                
                print(f"  Inner fold {inner_fold}")

                summary = train_evaluate_model(df_inner_train, df_inner_val, hyperparams)
                
                inner_scores.append(summary['iou_mean'])
                inner_fold_results.append({
                    "inner_fold": inner_fold,
                    "iou": float(summary['iou_mean']),
                    "recall": float(summary['recall_mean']),
                    "precision": float(summary['precision_mean'])
                })
            
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
    
    # ========== FINAL EVALUATION ON HELD-OUT TEST SET ==========
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION ON HELD-OUT TEST SET")
    print(f"{'='*80}")
    
    # Find most common best hyperparameters
    from collections import Counter
    hyperparam_counts = Counter([str(r['best_hyperparams']) for r in outer_results])
    most_common_hp_str = hyperparam_counts.most_common(1)[0][0]
    
    # Convert back to dict (hacky but works)
    for r in outer_results:
        if str(r['best_hyperparams']) == most_common_hp_str:
            final_hyperparams = r['best_hyperparams']
            break
    
    print(f"\nTraining final model with most common hyperparameters: {final_hyperparams}")
        
    summary_test = train_evaluate_model(data, df_test, final_hyperparams)
    
    print("\n--- Final Test Set Evaluation ---")
    
    print(f"\n{'='*80}")
    print(f"FINAL TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"IoU: {summary_test['iou_mean']:.4f}")
    print(f"Recall: {summary_test['recall_mean']:.4f}")
    print(f"Precision: {summary_test['precision_mean']:.4f}")
    print(f"\nNested CV Mean (±std): IoU={np.mean(ious):.4f}±{np.std(ious):.4f}")
    
    # Store final test results
    results_dict["final_test"] = {
        "hyperparams_used": final_hyperparams,
        "iou": float(summary_test['iou_mean']),
        "recall": float(summary_test['recall_mean']),
        "precision": float(summary_test['precision_mean']),
        "docs_evaluated": int(summary_test['docs_evaluated']),
        "nested_cv_iou_mean": float(np.mean(ious)),
        "nested_cv_iou_std": float(np.std(ious))
    }
    
    # Save results to JSON file
    print(f"\n{'='*80}")
    print(f"Saving results to {output_file}")
    print(f"{'='*80}")
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved successfully to {output_file}")
    
    return outer_results, summary_test


if __name__ == "__main__":
    df_train = pd.read_csv('data/train_flattened.csv')
    df_val = pd.read_csv('data/val_flattened.csv')
    df_test = pd.read_csv('data/test_flattened.csv')
    
    print(f"Original dataset sizes:")
    print(f"  Training documents: {df_train['doc_id'].nunique()}")
    print(f"  Validation documents: {df_val['doc_id'].nunique()}")
    print(f"  Test documents: {df_test['doc_id'].nunique()}")
    
    # Combine train and validation sets
    df_combined = pd.concat([df_train, df_val], ignore_index=True)
    print(f"\nCombined train+val documents: {df_combined['doc_id'].nunique()}")
    
    # Run nested cross-validation
    outer_results, test_results = nested_cross_validation(
        df_combined, 
        df_test, 
        n_outer_folds=10,  # Outer CV for robust estimation
        n_inner_folds=10   # Inner CV for hyperparameter tuning
    )
    