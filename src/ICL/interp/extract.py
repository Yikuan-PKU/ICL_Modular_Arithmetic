import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import entropy, pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

########################################
# load data
########################################


class AttentionDataLoaderBatch:
    """Batch generator for attention matrices to feed streaming Phase-1 pipeline"""

    def __init__(
        self,
        data_dir,
        n_layers=6,
        n_heads=8,
        tokens_per_example=9,
        batch_size=50,
        max_shots=None,
        max_sequences=None,
        step_range=None,
    ):
        """Args:
        data_dir (str | Path): Directory containing .npz files.
        n_layers (int): Number of transformer layers.
        n_heads (int): Number of attention heads.
        tokens_per_example (int): Tokens per example.
        batch_size (int): Number of attention matrices per batch.
        max_shots (int, optional): Skip sequences with n_shots > max_shots.
        max_sequences (int, optional): Stop after loading this many unique sequences.
        step_range (tuple[int,int], optional): Keep only steps in [min_step, max_step].

        """
        self.data_dir = Path(data_dir)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.tokens_per_example = tokens_per_example
        self.batch_size = batch_size
        self.max_shots = max_shots
        self.max_sequences = max_sequences
        self.step_range = step_range

    def parse_filename(self, filename):
        """Parse metadata from filename"""
        pattern = r".*step(\d+)_layer(\d+)_head(\d+)_k(\d+)_seq(\d+)\.npz"

        match = re.match(r".*" + pattern, filename)
        if match:
            return {
                "step": int(match.group(1)),
                "layer": int(match.group(2)),
                "head": int(match.group(3)),
                "n_shots": int(match.group(4)),
                "seq_id": int(match.group(5)),
            }
        return None

    def compute_context_boundaries(self, n_shots):
        return [(i * self.tokens_per_example, (i + 1) * self.tokens_per_example) for i in range(n_shots)]

    def compute_query_position(self, n_shots):
        return n_shots * self.tokens_per_example

    def infer_rule_complexity(self, n_shots):
        if n_shots <= 2:
            return 1
        if n_shots <= 4:
            return 2
        return 3

    def batch_generator(self):
        """Yield batches of files_data and sequence_metadata"""
        seq_data_store = defaultdict(lambda: defaultdict(dict))
        seq_metadata_store = {}
        batch_counter = 0
        seen_sequences = set()

        npz_files = sorted(self.data_dir.glob("*.npz"))
        for file_path in npz_files:
            metadata = self.parse_filename(file_path.name)
            if not metadata:
                continue

            step = metadata["step"]
            n_shots = metadata["n_shots"]
            seq_id = metadata["seq_id"]

            # Filter by step_range if provided
            if self.step_range:
                min_step, max_step = self.step_range
                if not (min_step <= step <= max_step):
                    continue

            # Filter by max_shots
            if self.max_shots is not None and n_shots >= self.max_shots:
                continue

            # Stop if max_sequences reached
            if self.max_sequences is not None and seq_id >= self.max_sequences:
                continue

            # Load attention matrix
            try:
                data = np.load(file_path)
                attn_matrix = data["attention_matrix"]
            except:
                continue

            # Store attention matrix
            seq_data_store[seq_id][metadata["layer"]][metadata["head"]] = attn_matrix

            # Store sequence-level metadata once
            if seq_id not in seq_metadata_store:
                seq_metadata_store[seq_id] = {
                    "n_shots": n_shots,
                    "context_boundaries": self.compute_context_boundaries(n_shots),
                    "query_position": self.compute_query_position(n_shots),
                    "rule_complexity": self.infer_rule_complexity(n_shots),
                    "step": step,
                }
                seen_sequences.add(seq_id)

            # Yield batch if batch size reached
            batch_counter += 1
            if batch_counter >= self.batch_size:
                yield dict(seq_data_store), dict(seq_metadata_store)
                seq_data_store.clear()
                seq_metadata_store.clear()
                batch_counter = 0

        # Yield any remaining sequences
        if seq_data_store:
            yield dict(seq_data_store), dict(seq_metadata_store)


########################################
# extract patterns by batch
########################################


class FeatureExtractor:
    """Extracts attention features batch-wise and accumulates globally."""

    def __init__(self):
        self.global_features = []

    def extract(self, seq_data_batch, seq_meta_batch):
        """Extract features from a batch of sequences.
        seq_data_batch: dict[seq_id][layer][head] = attention matrix
        seq_meta_batch: dict[seq_id] = metadata
        """
        batch_features = []

        for seq_id, layer_dict in seq_data_batch.items():
            meta = seq_meta_batch[seq_id]
            for layer, head_dict in layer_dict.items():
                for head, attn_matrix in head_dict.items():
                    features = self._compute_features(attn_matrix, meta, layer, head)
                    batch_features.append(features)

        # Accumulate globally
        self.global_features.extend(batch_features)
        return batch_features

    def _compute_features(self, attn_matrix, meta, layer, head):
        """Compute basic attention features for a single head."""
        return {
            "seq_id": meta["step"],
            "layer": layer,
            "head": head,
            "n_shots": meta["n_shots"],
            "entropy": self._attention_entropy(attn_matrix),
            "locality": self._locality_score(attn_matrix, meta["context_boundaries"]),
            "cross_attention": self._cross_attention(attn_matrix, meta["context_boundaries"]),
            "query_to_context": self._query_to_context(attn_matrix, meta["context_boundaries"], meta["query_position"]),
        }

    def _attention_entropy(self, attn):
        # compute entropy row-wise
        ent = entropy(attn + 1e-12, base=np.e, axis=-1)  # add small eps
        return float(ent.mean())

    def _locality_score(self, attn, context_boundaries):
        intra, total = 0.0, 0.0
        for start, end in context_boundaries:
            if start >= attn.shape[0] or end > attn.shape[1]:
                continue
            block = attn[start:end, start:end]
            intra += block.sum()
            total += attn[start:end, :].sum()
        return float(intra / (total + 1e-8))

    def _cross_attention(self, attn, context_boundaries):
        cross, total = 0.0, 0.0
        for i, (s1, e1) in enumerate(context_boundaries):
            for j, (s2, e2) in enumerate(context_boundaries):
                if i != j and s2 < attn.shape[1] and e2 <= attn.shape[1]:
                    cross += attn[s1:e1, s2:e2].sum()
            total += attn[s1:e1, :].sum()
        return float(cross / (total + 1e-8))

    def _query_to_context(self, attn, context_boundaries, query_pos):
        total = 0.0
        for start, end in context_boundaries:
            if start < attn.shape[0] and end <= attn.shape[1]:
                total += attn[start:end, query_pos].sum()
        return float(total)

    def get_all_features(self):
        return self.global_features


########################################
# compute global stat
########################################


class PreClusteringAnalyzer:
    """Compute aggregated statistics BEFORE clustering."""

    def __init__(self):
        self.summary_stats = {}

    def run(self, features, seq_meta_batch=None):
        entropies = [f["entropy"] for f in features]
        localities = [f["locality"] for f in features]
        cross_attn = [f["cross_attention"] for f in features]
        query2ctx = [f["query_to_context"] for f in features]

        self.summary_stats = {
            "entropy_mean": np.mean(entropies),
            "entropy_std": np.std(entropies),
            "locality_mean": np.mean(localities),
            "cross_attention_mean": np.mean(cross_attn),
            "query2ctx_mean": np.mean(query2ctx),
            "total_heads": len(features),
        }
        return self.summary_stats

    def get_summary(self):
        return self.summary_stats


########################################
# clustering module
########################################


class ClusteringModule:
    """Cluster heads based on extracted attention features."""

    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.cluster_labels = None
        self.cluster_centers = None

    def run(self, features):
        # Use behavioral features only
        X = np.array([[f["entropy"], f["locality"], f["cross_attention"], f["query_to_context"]] for f in features])
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(X_scaled)
        self.cluster_centers = kmeans.cluster_centers_
        return self.cluster_labels, self.cluster_centers


########################################
# compute stat after clustering
########################################


class PostClusteringAnalyzer:
    """Compute cluster-level statistics after clustering."""

    def __init__(self):
        self.cluster_summary = {}

    def run(self, features, cluster_labels):
        cluster_map = defaultdict(list)
        for label, f in zip(cluster_labels, features, strict=False):
            cluster_map[label].append(f)

        self.cluster_summary = {
            c: {
                "size": len(flist),
                "entropy_mean": np.mean([f["entropy"] for f in flist]),
                "locality_mean": np.mean([f["locality"] for f in flist]),
                "cross_attention_mean": np.mean([f["cross_attention"] for f in flist]),
                "query2ctx_mean": np.mean([f["query_to_context"] for f in flist]),
            }
            for c, flist in cluster_map.items()
        }
        return self.cluster_summary

    def get_summary(self):
        return self.cluster_summary


########################################
# Layer-Level Stat & Specialization Score
########################################


class LayerStatsAnalyzer:
    """Compute per-layer statistics, specialization scores, and inter-layer comparisons."""

    def __init__(self, n_layers, n_heads):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.layer_stats = {}  # Per-layer feature stats
        self.specialization_scores = {}  # Variance-based scores

    def run(self, features):
        """Compute per-layer statistics and specialization scores.
        features: list of dicts, each containing 'layer', 'head', and behavioral features
        """
        # --- 1. Aggregate features per layer ---
        layer_map = defaultdict(list)
        for f in features:
            layer_map[f["layer"]].append([f["entropy"], f["locality"], f["cross_attention"], f["query_to_context"]])

        self.layer_stats = {}
        for layer, feats in layer_map.items():
            feats_arr = np.array(feats)
            self.layer_stats[layer] = {
                "mean": np.mean(feats_arr, axis=0),
                "std": np.std(feats_arr, axis=0),
                "min": np.min(feats_arr, axis=0),
                "max": np.max(feats_arr, axis=0),
            }

        # --- 2. Compute specialization score ---
        # Normalized variance across features: higher = more specialized
        all_means = np.stack([v["mean"] for v in self.layer_stats.values()])
        feature_variances = np.var(all_means, axis=0)
        norm_var = feature_variances / (np.sum(feature_variances) + 1e-8)

        for idx, layer in enumerate(sorted(self.layer_stats.keys())):
            self.specialization_scores[layer] = float(np.sum((all_means[idx] - np.mean(all_means, axis=0)) ** 2))

        return self.layer_stats, self.specialization_scores

    def get_layer_stats(self):
        return self.layer_stats

    def get_specialization_scores(self):
        return self.specialization_scores


########################################
# Inter-Layer Comparative Metrics
########################################


class InterLayerAnalyzer:
    """Compare layers using correlation and functional diversity metrics."""

    def __init__(self):
        self.layer_correlations = {}
        self.functional_diversity = {}

    def run(self, features, cluster_labels=None):
        """features: list of dicts
        cluster_labels: optional, used to compute fraction of heads with extreme behavior
        """
        # --- 1. Build layer-feature map ---
        layer_map = defaultdict(list)
        for idx, f in enumerate(features):
            feats = [f["entropy"], f["locality"], f["cross_attention"], f["query_to_context"]]
            layer_map[f["layer"]].append(feats)

        # --- 2. Compute pairwise correlations ---
        self.layer_correlations = {}
        layers = sorted(layer_map.keys())
        for l1, l2 in combinations(layers, 2):
            feats1 = np.mean(np.array(layer_map[l1]), axis=0)
            feats2 = np.mean(np.array(layer_map[l2]), axis=0)
            # Pearson correlation
            corr = np.mean([pearsonr(feats1, feats2)[0]])
            self.layer_correlations[f"{l1}_{l2}"] = float(corr)

        # --- 3. Functional diversity (optional) ---
        if cluster_labels is not None:
            self.functional_diversity = defaultdict(float)
            for idx, f in enumerate(features):
                layer = f["layer"]
                self.functional_diversity[layer] += 1  # count heads per layer
            total_heads = len(features)
            for layer in self.functional_diversity:
                self.functional_diversity[layer] /= total_heads

        return self.layer_correlations, self.functional_diversity
