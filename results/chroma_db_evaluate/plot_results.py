#!/usr/bin/env python3
"""
Script to plot chunking evaluation results as bar charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os

# Read the CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'chunking_results.csv')
df = pd.read_csv(csv_path)

def parse_chunker_name(name):
    """Parse chunker name and return all relevant information."""
    is_trained = name.endswith('_trained')
    model_type = 'Ours' if is_trained else 'MiniLM'
    
    if 'ClusterSemanticChunker' in name:
        # Extract size from ClusterSemanticChunker_400 or ClusterSemanticChunker_400_trained
        parts = name.replace('ClusterSemanticChunker_', '').replace('_trained', '').split('_')
        size = int(parts[0]) if parts and parts[0].isdigit() else None
        clustering_type = 'ClusterSemantic'
        
        clean_name = f'ClusterSemantic ({size})' if size else 'ClusterSemantic'
        visual_group = f'ClusterSemantic_{size}' if size else 'ClusterSemantic'
        sort_key = (0, -size if size else 0, 0 if model_type == 'MiniLM' else 1)
        
    elif 'KamradtModifiedChunker' in name:
        clustering_type = 'KamradtModified'
        size = None
        clean_name = 'KamradtModified'
        visual_group = 'KamradtModified'
        sort_key = (1, 0 if model_type == 'MiniLM' else 1, 0)
        
    else:
        clustering_type = 'Other'
        size = None
        clean_name = name
        visual_group = 'Other'
        sort_key = (2, 0, 0)
    
    return {
        'chunker_clean': clean_name,
        'clustering_type': clustering_type,
        'model_type': model_type,
        'size': size,
        'visual_group': visual_group,
        'sort_key': sort_key
    }

# Parse all chunker names and add columns
parsed_data = df['chunker'].apply(parse_chunker_name)
parsed_df = pd.DataFrame(parsed_data.tolist())
df = pd.concat([df, parsed_df], axis=1)

# Sort by clustering type, size, and model type
df = df.sort_values('sort_key').reset_index(drop=True)

# Color mapping by model type
COLOR_MAP = {'Ours': 'steelblue', 'MiniLM': 'coral'}

# Set up plotting
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Chunking Evaluation Results', fontsize=16, fontweight='bold')

metrics = [
    ('iou_mean', 'IoU Mean', axes[0, 0]),
    ('recall_mean', 'Recall Mean', axes[0, 1]),
    ('precision_mean', 'Precision Mean', axes[1, 0]),
    ('precision_omega_mean', 'Precision Omega Mean', axes[1, 1]),
]

def calculate_x_positions(visual_groups):
    """Calculate x positions with spacing between visual groups."""
    x_positions = []
    current_pos = 0
    prev_group = None
    
    for group in visual_groups:
        if prev_group is not None and group != prev_group:
            current_pos += 0.5
        x_positions.append(current_pos)
        current_pos += 1
        prev_group = group
    
    return np.array(x_positions)

# Create legend elements
legend_elements = [
    Patch(facecolor=COLOR_MAP['Ours'], alpha=0.7, edgecolor='black', linewidth=1.2, label='Ours'),
    Patch(facecolor=COLOR_MAP['MiniLM'], alpha=0.7, edgecolor='black', linewidth=1.2, label='MiniLM')
]

# Plot each metric
for mean_col, title, ax in metrics:
    x_pos = calculate_x_positions(df['visual_group'])
    bar_colors = [COLOR_MAP[mt] for mt in df['model_type']]
    
    bars = ax.bar(x_pos, df[mean_col], alpha=0.7, edgecolor='black', 
                  linewidth=1.2, color=bar_colors)
    
    ax.set_xlabel('Chunker', fontsize=11, fontweight='bold')
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['chunker_clean'], rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend to each subplot
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, df[mean_col]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'chunking_results_barplot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Bar chart saved to: {output_path}")

