#!/usr/bin/env python3
"""
Script to plot chunking evaluation results as bar charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'chunking_results.csv')
df = pd.read_csv(csv_path)

# Clean up chunker names for better readability
def clean_chunker_name(name):
    """Simplify chunker names for display."""
    # Extract key information
    if 'ClusterSemanticChunker' in name:
        model = name.replace('ClusterSemanticChunker', '')
        if 'all-MiniLM-L6-v2' in model:
            # Format: all-MiniLM-L6-v2_400_0
            parts = model.split('_')
            if len(parts) >= 2:
                size = parts[1]  # e.g., "400"
                return f'ClusterSemantic (MiniLM, {size})'
            return 'ClusterSemantic (MiniLM)'
        elif 'models/best_model' in model:
            # Format: models/best_model_400_0 (note: contains underscore in "best_model")
            parts = model.split('_')
            if len(parts) >= 3:
                size = parts[2]  # Size is at index 2 because "best_model" contains an underscore
                return f'ClusterSemantic (Ours, {size})'
            return 'ClusterSemantic (Ours)'
    elif 'KamradtModifiedChunker' in name:
        model = name.replace('KamradtModifiedChunker', '')
        if 'all-MiniLM-L6-v2' in model:
            return 'KamradtModified (MiniLM)'
        elif 'models/best_model' in model:
            return 'KamradtModified (Ours)'
    return name

df['chunker_clean'] = df['chunker'].apply(clean_chunker_name)

# Extract configuration identifier (ignoring model type) for consistent coloring
def get_config_id(name):
    """Extract configuration identifier that matches across model types."""
    if 'ClusterSemanticChunker' in name:
        model = name.replace('ClusterSemanticChunker', '')
        # Extract size regardless of whether it's MiniLM or Ours
        if 'all-MiniLM-L6-v2' in model:
            # Format: all-MiniLM-L6-v2_400_0
            parts = model.split('_')
            if len(parts) >= 2:
                size = parts[1]  # e.g., "400" or "200"
                return f'ClusterSemantic_{size}'
            return 'ClusterSemantic'
        elif 'models/best_model' in model:
            # Format: models/best_model_400_0 (note: contains underscore in "best_model")
            parts = model.split('_')
            if len(parts) >= 3:
                size = parts[2]  # Size is at index 2 because "best_model" contains an underscore
                return f'ClusterSemantic_{size}'
            return 'ClusterSemantic'
    elif 'KamradtModifiedChunker' in name:
        return 'KamradtModified'
    return name

df['config_id'] = df['chunker'].apply(get_config_id)

# Extract model type for grouping
def get_model_type(name):
    """Extract model type (MiniLM vs Ours)."""
    if 'all-MiniLM-L6-v2' in name:
        return 'MiniLM'
    elif 'models/best_model' in name:
        return 'Ours'
    return 'Unknown'

df['model_type'] = df['chunker'].apply(get_model_type)

# Group and reorder: MiniLM methods first, then Ours
def get_group_order(name):
    """Get grouping and ordering for bars."""
    model_type = get_model_type(name)
    
    if model_type == 'MiniLM':
        group = 0  # MiniLM group
    else:  # Ours
        group = 1  # Ours group
    
    # Within each group, order by configuration
    if 'ClusterSemanticChunker' in name:
        model = name.replace('ClusterSemanticChunker', '')
        parts = model.split('_')
        if 'all-MiniLM-L6-v2' in model:
            # Format: all-MiniLM-L6-v2_400_0
            if len(parts) >= 2:
                size = int(parts[1]) if parts[1].isdigit() else 0
            else:
                size = 0
        elif 'models/best_model' in model:
            # Format: models/best_model_400_0 (size is at index 2)
            if len(parts) >= 3:
                size = int(parts[2]) if parts[2].isdigit() else 0
            else:
                size = 0
        else:
            size = 0
        return (group, 0, -size)  # ClusterSemantic first, then by size (larger first)
    else:  # KamradtModified
        return (group, 1, 0)  # KamradtModified second

df['sort_key'] = df['chunker'].apply(get_group_order)
df = df.sort_values('sort_key').reset_index(drop=True)

# Identify group boundaries for visual separation (by model type)
df['model_group'] = df['model_type']

# Create color mapping for consistent colors across all plots
# Use config_id so matching configurations get the same color
unique_configs = df['config_id'].unique()
# Use a colormap to assign colors
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_configs)))
color_map = dict(zip(unique_configs, colors))

# Set up the plotting style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Chunking Evaluation Results', fontsize=16, fontweight='bold')

# Define metrics to plot
metrics = [
    ('iou_mean', 'iou_std', 'IoU Mean', axes[0, 0]),
    ('recall_mean', 'recall_std', 'Recall Mean', axes[0, 1]),
    ('precision_mean', 'precision_std', 'Precision Mean', axes[1, 0]),
    ('precision_omega_mean', 'precision_omega_std', 'Precision Omega Mean', axes[1, 1]),
]

# Plot each metric
for mean_col, std_col, title, ax in metrics:
    # Create x positions with spacing between groups
    x_positions = []
    current_pos = 0
    group_boundaries = []
    prev_group = None
    
    for i, group in enumerate(df['model_group']):
        if prev_group is not None and group != prev_group:
            current_pos += 0.5  # Add spacing between groups
            group_boundaries.append(current_pos - 0.25)
        x_positions.append(current_pos)
        current_pos += 1
        prev_group = group
    
    x_pos = np.array(x_positions)
    
    bars = ax.bar(x_pos, df[mean_col], yerr=df[std_col], 
                  capsize=5, alpha=0.7, edgecolor='black', linewidth=1.2,
                  color=[color_map[config_id] for config_id in df['config_id']])
    
    # Add vertical lines to separate groups
    for boundary in group_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Chunker', fontsize=11, fontweight='bold')
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['chunker_clean'], rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for i, (bar, mean_val) in enumerate(zip(bars, df[mean_col])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + df[std_col].iloc[i],
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# Save the plot
output_path = os.path.join(os.path.dirname(__file__), 'chunking_results_barplot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Bar chart saved to: {output_path}")

# Also create individual plots for each metric
# output_dir = os.path.dirname(__file__)
# for mean_col, std_col, title, _ in metrics:
#     fig, ax = plt.subplots(figsize=(10, 6))
#     x_pos = np.arange(len(df))
#     bars = ax.bar(x_pos, df[mean_col], yerr=df[std_col], 
#                   capsize=5, alpha=0.7, edgecolor='black', linewidth=1.2,
#                   color=plt.cm.viridis(np.linspace(0, 1, len(df))))
    
#     ax.set_xlabel('Chunker', fontsize=12, fontweight='bold')
#     ax.set_ylabel(title, fontsize=12, fontweight='bold')
#     ax.set_title(f'{title} by Chunker', fontsize=14, fontweight='bold')
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(df['chunker_clean'], rotation=45, ha='right', fontsize=10)
#     ax.grid(axis='y', alpha=0.3, linestyle='--')
    
#     # Add value labels on top of bars
#     for i, (bar, mean_val) in enumerate(zip(bars, df[mean_col])):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + df[std_col].iloc[i],
#                 f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
#     plt.tight_layout()
#     filename = title.lower().replace(' ', '_') + '_barplot.png'
#     filepath = os.path.join(output_dir, filename)
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Individual plot saved to: {filepath}")

# print("\nAll plots generated successfully!")

