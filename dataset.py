from datasets import load_dataset
import pandas as pd
import json

import os
os.makedirs('data', exist_ok=True)

if not os.path.exists('data/train.csv'):
    untitled_dataset = load_dataset(
        "saeedabc/wiki727k",
        drop_titles=True,         # custom config flag defined by the dataset script
        num_proc=8,
        trust_remote_code=True,
    )

    df_train = untitled_dataset['train'].to_pandas().to_csv('data/train.csv', index=False)
    df_val = untitled_dataset['validation'].to_pandas().to_csv('data/val.csv', index=False)
    df_test = untitled_dataset['test'].to_pandas().to_csv('data/test.csv', index=False)
else:
    df_train = pd.read_csv('data/train.csv')
    df_val = pd.read_csv('data/val.csv')
    df_test = pd.read_csv('data/test.csv')

def flatten_dataset(df, num_samples):
    rows = []
    i = 0
    for _, row in df.iterrows():
        if i >= num_samples:
            break
        i += 1
        print(f"Processing sample {i} of {num_samples}")
        doc_id = row["id"]
        ids = json.loads(row["ids"])
        sents = json.loads(row["sentences"])
        for idx, (sid, sent) in enumerate(zip(ids, sents)):
            paragraph_id, sent_id = sid.split("_")
            paragraph_id = int(paragraph_id)
            sent_id = int(sent_id)

            rows.append({
                "doc_id": doc_id,
                "paragraph_id": paragraph_id,
                "sent_id": sent_id,
                "global_index": idx,
                "sentence": sent,
            })

    return pd.DataFrame(rows)

df_train = flatten_dataset(df_train, 5500)
df_val = flatten_dataset(df_val, 500)
df_test = flatten_dataset(df_test, 500)

df_train.to_csv('data/train_flattened.csv', index=False)
df_val.to_csv('data/val_flattened.csv', index=False)
df_test.to_csv('data/test_flattened.csv', index=False)