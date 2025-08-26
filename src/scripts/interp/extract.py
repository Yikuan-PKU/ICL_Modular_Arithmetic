from ICL.interp.extract import (
    AttentionDataLoaderBatch,
    ClusteringModule,
    FeatureExtractor,
    InterLayerAnalyzer,
    LayerStatsAnalyzer,
    PostClusteringAnalyzer,
    PreClusteringAnalyzer,
)
from ICL.load_util import JsonProcessor
from ICL.settings import PATH

data_dir = (
    PATH.result_dir
    / "uniform_1000_L3_M3/clm_noshuffle_seedbalanced/id_generalization/collection/raw_evaluations/attention_data/clm_noshuffle_seedbalanced_step36880"
)

output_file = PATH.interp_dir / "uniform_1000_L3_M3/clm_noshuffle_seedbalanced/layer"


n_layers, n_heads = 6, 8
batch_size = 50
n_clusters = 4
max_shots = 10
max_sequences = 10

# --- Initialize modules ---
dataloader = AttentionDataLoaderBatch(
    data_dir=data_dir,
    n_layers=n_layers,
    n_heads=n_heads,
    batch_size=batch_size,
    max_shots=max_shots,
    max_sequences=max_sequences,
)
extractor = FeatureExtractor()
pre_analyzer = PreClusteringAnalyzer()
clustering_module = ClusteringModule(n_clusters=n_clusters)
post_analyzer = PostClusteringAnalyzer()
layer_analyzer = LayerStatsAnalyzer(n_layers=n_layers, n_heads=n_heads)
inter_layer_analyzer = InterLayerAnalyzer()

# --- Feature extraction & pre-cluster analysis ---
print("[INFO] Extracting features and pre-clustering stats...")
for seq_data_batch, seq_meta_batch in dataloader.batch_generator():
    extractor.extract(seq_data_batch, seq_meta_batch)
    pre_analyzer.run(extractor.get_all_features())

features = extractor.get_all_features()
print("Pre-clustering summary:", pre_analyzer.get_summary())

# --- Layer stats & specialization ---
layer_stats, spec_scores = layer_analyzer.run(features)
print("Layer stats:", layer_stats)
print("Specialization scores:", spec_scores)

# --- Clustering ---
cluster_labels, _ = clustering_module.run(features)
print("[INFO] Clustering done.")

# --- Post-clustering analysis ---
post_summary = post_analyzer.run(features, cluster_labels)
print("Post-clustering summary:", post_summary)

# --- Inter-layer comparison ---
corr, diversity = inter_layer_analyzer.run(features, cluster_labels)
print("Inter-layer correlations:", corr)
print("Functional diversity:", diversity)

phase1_json = {
    "metadata": {
        "num_layers": n_layers,
        "num_heads": n_heads,
        "num_sequences": max_sequences,
        "n_clusters": n_clusters,
        "batch_size": batch_size,
    },
    "layer_stats": layer_stats,
    "layer_spec": spec_scores,
    "inter_layer_corr": corr,
    "inter_layer_div": diversity,
    "cluster_labels": cluster_labels.tolist() if cluster_labels is not None else [],
    "features": features,
}

if output_file is not None:
    JsonProcessor.save_json(phase1_json, output_file / "id_generalization.json")


def extract_stat():
    n_layers, n_heads = 6, 8
    batch_size = 50
    n_clusters = 4
    max_shots = 10
    max_sequences = 10

    # --- Initialize modules ---
    dataloader = AttentionDataLoaderBatch(
        data_dir=data_dir,
        n_layers=n_layers,
        n_heads=n_heads,
        batch_size=batch_size,
        max_shots=max_shots,
        max_sequences=max_sequences,
    )
    extractor = FeatureExtractor()
    pre_analyzer = PreClusteringAnalyzer()
    clustering_module = ClusteringModule(n_clusters=n_clusters)
    post_analyzer = PostClusteringAnalyzer()
    layer_analyzer = LayerStatsAnalyzer(n_layers=n_layers, n_heads=n_heads)
    inter_layer_analyzer = InterLayerAnalyzer()

    # --- Feature extraction & pre-cluster analysis ---
    print("[INFO] Extracting features and pre-clustering stats...")
    for seq_data_batch, seq_meta_batch in dataloader.batch_generator():
        extractor.extract(seq_data_batch, seq_meta_batch)
        pre_analyzer.run(extractor.get_all_features())

    features = extractor.get_all_features()
    print("Pre-clustering summary:", pre_analyzer.get_summary())

    # --- Layer stats & specialization ---
    layer_stats, spec_scores = layer_analyzer.run(features)
    print("Layer stats:", layer_stats)
    print("Specialization scores:", spec_scores)

    # --- Clustering ---
    cluster_labels, _ = clustering_module.run(features)
    print("[INFO] Clustering done.")

    # --- Post-clustering analysis ---
    post_summary = post_analyzer.run(features, cluster_labels)
    print("Post-clustering summary:", post_summary)

    # --- Inter-layer comparison ---
    corr, diversity = inter_layer_analyzer.run(features, cluster_labels)
    print("Inter-layer correlations:", corr)
    print("Functional diversity:", diversity)

    phase1_json = {
        "metadata": {
            "num_layers": n_layers,
            "num_heads": n_heads,
            "num_sequences": max_sequences,
            "n_clusters": n_clusters,
            "batch_size": batch_size,
        },
        "layer_stats": layer_stats,
        "layer_spec": spec_scores,
        "inter_layer_corr": corr,
        "inter_layer_div": diversity,
        "cluster_labels": cluster_labels.tolist() if cluster_labels is not None else [],
        "features": features,
    }

    if output_file is not None:
        JsonProcessor.save_json(phase1_json, output_file / "id_generalization.json")


if __name__ == "__main__":
    main()
