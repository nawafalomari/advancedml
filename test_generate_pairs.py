#!/usr/bin/env python3
"""
Simple test script for generate_pairs function.
Tests all three pick_technique options with a sample from test_flattened.csv
"""
import time
import pandas as pd
from contrastive_learning import generate_pairs

def test_generate_pairs():
    # Load test data
    print("Loading test_flattened_10000.csv...")
    df = pd.read_csv('data/test_flattened_10000.csv')
    
    # Take a sample - get first few documents
    print(f"\nTotal rows in dataset: {len(df)}")
    print(f"Total documents: {df['doc_id'].nunique()}")
    
    # Sample a few documents for testing
    sample_doc_ids = df['doc_id'].unique()  # First 5 documents
    df_sample = df[df['doc_id'].isin(sample_doc_ids)].copy()
    
    print(f"\nSample: {len(df_sample)} rows from {len(sample_doc_ids)} documents")
    # print(f"Document IDs: {sample_doc_ids.tolist()}")
    
    # Test each pick_technique
    techniques = ["next", "in_doc", "cross_doc"]
    
    for technique in techniques:
        print(f"\n{'='*60}")
        print(f"Testing pick_technique: '{technique}'")
        print(f"{'='*60}")
        
        try:
            # Generate pairs with small num_pairs for testing
            start_time = time.time()
            pairs = generate_pairs(df_sample, pick_technique=technique, num_pairs=10)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
            print(f"Generated {len(pairs)} pairs")
            
            if len(pairs) > 0:
                # Count positive and negative pairs
                positive_count = sum(1 for p in pairs if p[2] == 1)
                negative_count = sum(1 for p in pairs if p[2] == 0)
                
                print(f"  Positive pairs (label=1): {positive_count}")
                print(f"  Negative pairs (label=0): {negative_count}")
                
                # Show first few examples
                print(f"\nFirst 3 examples:")
                for i, (sent1, sent2, label) in enumerate(pairs[:3]):
                    print(f"\n  Pair {i+1} (label={label}):")
                    print(f"    Sent1: {sent1[:80]}..." if len(sent1) > 80 else f"    Sent1: {sent1}")
                    print(f"    Sent2: {sent2[:80]}..." if len(sent2) > 80 else f"    Sent2: {sent2}")
            else:
                print("  No pairs generated!")
                
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_generate_pairs()

