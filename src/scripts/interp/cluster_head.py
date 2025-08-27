import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from ICL.interp.extract import (
    ClusteringModule,
)


def cluster_selected_layers(all_features, layer_spec_scores, selected_metric="memorization", n_clusters=4):
    """Cluster heads in top layers based on layer specialization scores.

    Args:
        all_features (dict): Output of FeatureExtractor, keyed by seq_id -> layer -> head
        layer_spec_scores (dict): Layer specialization scores per task, e.g., {'memorization': {...}, ...}
        selected_metric (str): Which metric to select top layers, e.g., 'memorization'
        n_clusters (int): Number of clusters for clustering module

    Returns:
        dict: {
            'selected_layers': [...],
            'cluster_labels': {layer: {head: label, ...}, ...},
            'cluster_stats': {...}  # optional stats per cluster
        }

    """
    # 1. Select top layers based on specialization score
    scores = layer_spec_scores[selected_metric]
    top_layers = sorted(scores, key=lambda l: scores[l], reverse=True)[:2]  # top 2 layers
    print(f"Selected layers for clustering: {top_layers}")

    # 2. Extract features for only top layers
    features_to_cluster = {
        seq_id: {layer: heads for layer, heads in layer_data.items() if layer in top_layers}
        for seq_id, layer_data in all_features.items()
    }

    # 3. Initialize ClusterAnalyzer (reuse existing)
    cluster_analyzer = ClusteringModule(n_clusters=n_clusters)

    # 4. Flatten heads across selected layers into a single feature matrix
    flattened_features = []
    head_map = []  # keep track of (seq_id, layer, head) for assignment
    for seq_id, layer_data in features_to_cluster.items():
        for layer, heads in layer_data.items():
            for head_idx, head_feat in heads.items():
                flattened_features.append(head_feat)
                head_map.append((seq_id, layer, head_idx))

    # 5. Run clustering
    flattened_features = np.stack(flattened_features)
    labels = cluster_analyzer.fit_predict(flattened_features)

    # 6. Map labels back to layers/heads
    cluster_labels = {layer: {} for layer in top_layers}
    for idx, (seq_id, layer, head_idx) in enumerate(head_map):
        cluster_labels[layer][head_idx] = int(labels[idx])

    # 7. Compute cluster stats (optional)
    cluster_stats = {}
    for layer in top_layers:
        layer_labels = cluster_labels[layer]
        stats = {}
        for label in range(n_clusters):
            assigned_heads = [h for h, l in layer_labels.items() if l == label]
            stats[label] = {
                "num_heads": len(assigned_heads),
                # optionally add avg specialization, avg entropy, etc.
            }
        cluster_stats[layer] = stats

    # 8. Return everything in a dict (ready to save as JSON)
    result_dict = {"selected_layers": top_layers, "cluster_labels": cluster_labels, "cluster_stats": cluster_stats}
    return result_dict


# ========================
# Phase-2: Geometry Analysis
# ========================


def extract_head_features(all_features, selected_layers):
    """Collect features from Phase-1 for selected layers.

    Args:
        all_features (dict): seq_id -> layer -> head -> feature vector
        selected_layers (list[int]): layers to include
    Returns:
        feature_matrix (np.array): N x D array of features
        head_map (list[tuple]): list of (seq_id, layer, head) mapping to rows

    """
    flattened_features = []
    head_map = []

    for seq_id, layer_data in all_features.items():
        for layer, heads in layer_data.items():
            if layer not in selected_layers:
                continue
            for head_idx, head_feat in heads.items():
                flattened_features.append(head_feat)
                head_map.append((seq_id, layer, head_idx))

    feature_matrix = np.stack(flattened_features)
    return feature_matrix, head_map


def run_pca(feature_matrix, n_components=2):
    """Reduce feature dimensionality using PCA.

    Returns:
        pca_features: N x n_components
        pca_model: PCA object

    """
    pca_model = PCA(n_components=n_components)
    pca_features = pca_model.fit_transform(feature_matrix)
    return pca_features, pca_model


def visualize_pca(pca_features, head_map, cluster_labels=None, title="PCA of Selected Heads"):
    """Simple scatter plot of PCA results."""
    plt.figure(figsize=(8, 6))
    colors = None
    if cluster_labels is not None:
        colors = []
        for seq_id, layer, head_idx in head_map:
            colors.append(cluster_labels.get(layer, {}).get(head_idx, -1))
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=colors, cmap="tab10", alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.colorbar(label="Cluster Label")
    plt.show()


# ========================
# Example Usage
# ========================

if __name__ == "__main__":
    # Load Phase-1 outputs
    with open("selected_layer_clusters.json") as f:
        phase1_data = json.load(f)

    selected_layers = phase1_data["selected_layers"]
    cluster_labels = phase1_data["cluster_labels"]
    # Suppose all_features comes from FeatureExtractor
    all_features = ...  # seq_id -> layer -> head -> feature vector

    # Extract features for selected layers
    feature_matrix, head_map = extract_head_features(all_features, selected_layers)

    # Run PCA
    pca_features, pca_model = run_pca(feature_matrix, n_components=2)

    # Visualize PCA with cluster labels
    visualize_pca(pca_features, head_map, cluster_labels)
