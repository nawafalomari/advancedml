import pandas as pd
import argparse
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch

# Import functions from contrastive_learning.py
from contrastive_learning import (
    evaluate_docs,
    df_to_eval_docs
)

df_test = pd.read_csv("data/test_flattened_10000.csv")
df_test["sentence"] = df_test["sentence"].astype(str)
docs_test = df_to_eval_docs(df_test)

# load best model 
model = SentenceTransformer("models/best_model")

embed_fn = lambda sents: model.encode(sents, convert_to_numpy=True, batch_size=64, show_progress_bar=False)

metrics_per_doc, summary = evaluate_docs(
    docs_test, 
    embed_fn, 
    name="Test Set",
    threshold_method="mean"
)

print(f"\n{'='*80}")
print("EVALUATION RESULTS")
print(f"{'='*80}")
print(f"Threshold method: mean")
print(f"Documents evaluated: {summary['docs_evaluated']}")
print(f"\nMean Metrics:")
print(f"  IoU:      {summary['iou_mean']:.4f}")
print(f"  Recall:   {summary['recall_mean']:.4f}")
print(f"  Precision: {summary['precision_mean']:.4f}")
