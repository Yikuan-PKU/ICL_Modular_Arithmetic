import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import entropy, spearmanr
from sklearn.cluster import MiniBatchKMeans
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
            if self.max_shots is not None and n_shots <= self.max_shots:
                continue

            # Stop if max_sequences reached
            if self.max_sequences is not None and seq_id <= self.max_sequences:
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


# batch loader
class AttentionPatternExtractorStreaming:
    """Streaming, batch-safe feature extractor for Phase-1 analysis"""

    def __init__(self, n_layers=6, n_heads=8, missing_data_strategy="skip"):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.missing_data_strategy = missing_data_strategy  # 'skip', 'nan', 'zero'

    def _handle_missing_data(self, value, default_value=0.0):
        if self.missing_data_strategy == "skip":
            return None
        if self.missing_data_strategy == "nan":
            return np.nan
        if self.missing_data_strategy == "zero":
            return default_value
        return value

    def compute_attention_entropy(self, attention_matrix):
        if attention_matrix is None:
            return self._handle_missing_data(None, 0.0)
        attention_matrix += 1e-8
        try:
            return float(np.mean([entropy(row) for row in attention_matrix]))
        except:
            return self._handle_missing_data(None, 0.0)

    def compute_locality_score(self, attention_matrix, context_boundaries):
        if attention_matrix is None:
            return self._handle_missing_data(None, 0.0)
        try:
            intra_attention = 0.0
            total_attention = 0.0
            for start, end in context_boundaries:
                if start >= attention_matrix.shape[0] or end > attention_matrix.shape[1]:
                    continue
                block = attention_matrix[start:end, start:end]
                intra_attention += np.sum(block)
                total_attention += np.sum(attention_matrix[start:end, :])
            return intra_attention / (total_attention + 1e-8)
        except:
            return self._handle_missing_data(None, 0.0)

    def compute_cross_example_attention(self, attention_matrix, context_boundaries):
        if attention_matrix is None:
            return self._handle_missing_data(None, 0.0)
        try:
            cross_attention = 0.0
            total_attention = 0.0
            for i, (start1, end1) in enumerate(context_boundaries):
                if start1 >= attention_matrix.shape[0] or end1 > attention_matrix.shape[1]:
                    continue
                for j, (start2, end2) in enumerate(context_boundaries):
                    if i != j and start2 < attention_matrix.shape[1] and end2 <= attention_matrix.shape[1]:
                        cross_attention += np.sum(attention_matrix[start1:end1, start2:end2])
                total_attention += np.sum(attention_matrix[start1:end1, :])
            return cross_attention / (total_attention + 1e-8)
        except:
            return self._handle_missing_data(None, 0.0)

    def compute_icl_patterns(self, attention_matrix, context_boundaries, query_position):
        if attention_matrix is None:
            return {
                "context_to_query": self._handle_missing_data(None, 0.0),
                "query_to_context": self._handle_missing_data(None, 0.0),
            }
        if query_position is None or not context_boundaries:
            return {"context_to_query": 0.0, "query_to_context": 0.0}
        try:
            context_to_query = sum(
                np.sum(attention_matrix[query_position, start:end])
                for start, end in context_boundaries
                if start < attention_matrix.shape[1] and end <= attention_matrix.shape[1]
            )
            query_to_context = 0.0
            for start, end in context_boundaries:
                if query_position < start < attention_matrix.shape[0] and end <= attention_matrix.shape[0]:
                    query_to_context += np.sum(attention_matrix[start:end, query_position])
            return {"context_to_query": context_to_query, "query_to_context": query_to_context}
        except:
            return {
                "context_to_query": self._handle_missing_data(None, 0.0),
                "query_to_context": self._handle_missing_data(None, 0.0),
            }

    def extract_head_features_streaming(self, files_data, sequence_metadata):
        """Generator that yields features per head, per sequence"""
        for seq_id, seq_attention in files_data.items():
            context_boundaries = sequence_metadata[seq_id]["context_boundaries"]
            query_position = sequence_metadata[seq_id]["query_position"]
            rule_complexity = sequence_metadata[seq_id]["rule_complexity"]

            for layer_idx in range(self.n_layers):
                for head_idx in range(self.n_heads):
                    attention_matrix = seq_attention.get(layer_idx, {}).get(head_idx, None)

                    entropy_score = self.compute_attention_entropy(attention_matrix)
                    locality_score = self.compute_locality_score(attention_matrix, context_boundaries)
                    cross_score = self.compute_cross_example_attention(attention_matrix, context_boundaries)
                    icl_patterns = self.compute_icl_patterns(attention_matrix, context_boundaries, query_position)

                    feature_vector = [
                        entropy_score,
                        locality_score,
                        cross_score,
                        icl_patterns["context_to_query"],
                        icl_patterns["query_to_context"],
                        rule_complexity,
                        layer_idx,
                        head_idx,
                    ]

                    if self.missing_data_strategy == "skip" and any(f is None for f in feature_vector[:5]):
                        continue

                    yield np.array(feature_vector, dtype=np.float32), (seq_id, layer_idx, head_idx)
                    # Discard matrix
                    del attention_matrix


class Phase1AnalysisPipeline:
    """Phase-1 pipeline: batch feature extraction, clustering, and layer specialization mapping"""

    def __init__(self, n_layers=6, n_heads=8, n_clusters=4, missing_data_strategy="skip", batch_size=50):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.extractor = AttentionPatternExtractorStreaming(n_layers, n_heads, missing_data_strategy)
        self.batch_size = batch_size
        self.n_clusters = n_clusters

    def extract_features_from_batches(self, batch_generator):
        """Streaming feature extraction and incremental clustering"""
        all_features = []
        all_head_ids = []
        all_sequence_metadata = {}

        for files_data, seq_meta in batch_generator:
            all_sequence_metadata.update(seq_meta)

            for feature_vector, head_id in self.extractor.extract_head_features_streaming(files_data, seq_meta):
                all_features.append(feature_vector)
                all_head_ids.append(head_id)

            # Clean up
            del files_data, seq_meta

        if all_features:
            features_array = np.stack(all_features)
        else:
            features_array = np.zeros((0, 8), dtype=np.float32)

        return features_array, all_head_ids, all_sequence_metadata

    def cluster_heads(self, features_array):
        """Cluster heads based on attention features only (exclude layer/head indices)"""
        if features_array.shape[0] == 0:
            return np.array([]), None

        # Use only behavioral features (first 6 columns)
        clustering_features = features_array[:, :6]
        scaler = StandardScaler()
        clustering_features_scaled = scaler.fit_transform(clustering_features)

        clusterer = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size)
        cluster_labels = clusterer.fit_predict(clustering_features_scaled)

        return cluster_labels, clusterer

    def compute_layer_specialization(self, features_array, head_identifiers, cluster_labels):
        """Map cluster labels to layers and compute mean features per cluster per layer"""
        layer_cluster_map = defaultdict(lambda: defaultdict(list))

        for i, (seq_id, layer_idx, head_idx) in enumerate(head_identifiers):
            cluster = cluster_labels[i]
            layer_cluster_map[layer_idx][cluster].append(i)

        layer_specialization = {}
        for layer_idx, cluster_dict in layer_cluster_map.items():
            layer_specialization[layer_idx] = {}
            for cluster_label, head_indices in cluster_dict.items():
                mean_vector = np.mean(features_array[head_indices, :6], axis=0)
                layer_specialization[layer_idx][cluster_label] = mean_vector

        return layer_specialization

    def run_pipeline(self, batch_generator):
        # Step 1: extract features
        features_array, head_identifiers, sequence_metadata = self.extract_features_from_batches(batch_generator)

        # Step 2: cluster heads
        cluster_labels, clusterer_model = self.cluster_heads(features_array)

        # Step 3: layer specialization mapping
        if len(features_array) > 0:
            layer_specialization = self.compute_layer_specialization(features_array, head_identifiers, cluster_labels)
        else:
            layer_specialization = {}

        return {
            "features_array": features_array,
            "head_identifiers": head_identifiers,
            "sequence_metadata": sequence_metadata,
            "cluster_labels": cluster_labels,
            "clusterer_model": clusterer_model,
            "layer_specialization": layer_specialization,
        }


########################################
# compute stat
########################################


class Phase1Stat:
    """Phase-1 analysis: compute mandatory stats for attention heads.
    Features must match head_identifiers.
    """

    def __init__(self, features_array, head_identifiers, sequence_metadata):
        self.features_array = features_array  # shape: [num_heads, num_features]
        self.head_identifiers = head_identifiers  # list of (seq_id, layer_idx, head_idx)
        self.sequence_metadata = sequence_metadata

    # -----------------------------------
    # Main entry
    # -----------------------------------
    def run_phase1(self, clusters):
        print("Computing Layer Specialization...")
        layer_stats = self.compute_layer_specialization(clusters)

        print("Computing Complexity Trends...")
        complexity_stats = self.compute_complexity_trends(clusters)

        print("Computing Cluster-Layer Enrichment...")
        cluster_layer_stats = self.compute_cluster_layer_enrichment(clusters)

        print("Computing Cluster-Performance Mapping...")
        cluster_perf_stats = self.compute_cluster_performance(clusters)

        return {
            "layer_specialization": layer_stats,
            "complexity_trends": complexity_stats,
            "cluster_layer_enrichment": cluster_layer_stats,
            "cluster_performance": cluster_perf_stats,
        }

    # -----------------------------------
    # Layer specialization stats
    # -----------------------------------
    def compute_layer_specialization(self, clusters):
        """Compute mean attention features per layer and cluster.
        Only use actual feature dimensions (exclude layer/head indices)
        Returns dict: layer_idx -> cluster -> mean_feature_vector
        """
        layer_cluster_features = defaultdict(lambda: defaultdict(list))
        for head_idx, (seq_id, layer_idx, head_id) in enumerate(self.head_identifiers):
            cluster_label = clusters[head_idx]
            # slice features to exclude last two columns (layer_idx, head_idx)
            feat_vector = self.features_array[head_idx, :6]  # adjust if you add/remove features
            layer_cluster_features[layer_idx][cluster_label].append(feat_vector)

        # Compute mean vectors
        layer_stats = {}
        for layer_idx, cluster_dict in layer_cluster_features.items():
            layer_stats[layer_idx] = {}
            for cluster_label, feats in cluster_dict.items():
                layer_stats[layer_idx][cluster_label] = np.mean(feats, axis=0)
        return layer_stats

    # -----------------------------------
    # Complexity trends: n_shots
    # -----------------------------------
    def compute_complexity_trends(self, clusters):
        cluster_trends = defaultdict(dict)
        for cluster_label in np.unique(clusters):
            feats = []
            n_shots = []
            for head_idx, (seq_id, layer_idx, head_id) in enumerate(self.head_identifiers):
                if clusters[head_idx] != cluster_label:
                    continue
                feats.append(self.features_array[head_idx, 0])  # primary feature (entropy)
                n_shots.append(self.sequence_metadata[seq_id]["n_shots"])
            if len(feats) > 1:
                slope = np.polyfit(n_shots, feats, 1)[0]
                rho, _ = spearmanr(n_shots, feats)
                low_feats = [f for f, n in zip(feats, n_shots, strict=False) if n <= np.median(n_shots)]
                high_feats = [f for f, n in zip(feats, n_shots, strict=False) if n > np.median(n_shots)]
                effect_size = (np.mean(high_feats) - np.mean(low_feats)) / np.sqrt(
                    0.5 * (np.var(high_feats) + np.var(low_feats) + 1e-8)
                )
                cluster_trends[cluster_label] = {"slope": slope, "spearman_r": rho, "effect_size": effect_size}
        return cluster_trends

    # -----------------------------------
    # Cluster-Layer enrichment
    # -----------------------------------
    def compute_cluster_layer_enrichment(self, clusters):
        enrichment = defaultdict(lambda: defaultdict(int))
        for head_idx, (seq_id, layer_idx, head_id) in enumerate(self.head_identifiers):
            cluster_label = clusters[head_idx]
            enrichment[layer_idx][cluster_label] += 1
        return enrichment

    # -----------------------------------
    # Cluster-performance mapping
    # -----------------------------------
    def compute_cluster_performance(self, clusters):
        cluster_perf = defaultdict(lambda: defaultdict(list))
        for head_idx, (seq_id, layer_idx, head_id) in enumerate(self.head_identifiers):
            cluster_label = clusters[head_idx]
            # use outcome from metadata if available, fallback to "unknown"
            outcome = self.sequence_metadata[seq_id].get("outcome", "unknown")
            cluster_perf[cluster_label][outcome].append(self.features_array[head_idx])
        return cluster_perf
