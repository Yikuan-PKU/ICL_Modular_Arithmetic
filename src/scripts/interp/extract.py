import logging

from ICL.interp.extract import AttentionDataLoaderBatch, Phase1AnalysisPipeline

logger = logging.getLogger(__name__)


def run_phase1_pipeline(
    data_dir,
    batch_size=16,
    n_layers=6,
    n_heads=8,
    tokens_per_example=3,
    missing_data_strategy="skip",
    step_range=None,
    max_shots=None,
    max_sequences=None,
    n_clusters=4,
):
    """Refactored Phase-1 pipeline:
    - Streaming feature extraction
    - Clustering attention heads based on behavioral features
    - Layer specialization mapping
    """
    # ----------------------------
    # 1️⃣ Data Loading & Batching
    # ----------------------------
    loader = AttentionDataLoaderBatch(
        data_dir=data_dir,
        n_layers=n_layers,
        n_heads=n_heads,
        tokens_per_example=tokens_per_example,
        batch_size=batch_size,
        max_shots=max_shots,
        max_sequences=max_sequences,
        step_range=step_range,
    )
    batches = loader.batch_generator()

    # ----------------------------
    # 2️⃣ Feature Extraction & Clustering
    # ----------------------------
    pipeline = Phase1AnalysisPipeline(
        n_layers=n_layers,
        n_heads=n_heads,
        n_clusters=n_clusters,
        missing_data_strategy=missing_data_strategy,
        batch_size=batch_size,
    )

    result = pipeline.run_pipeline(batches)

    features_array = result["features_array"]
    head_identifiers = result["head_identifiers"]
    sequence_metadata = result["sequence_metadata"]
    cluster_labels = result["cluster_labels"]
    clusterer_model = result["clusterer_model"]
    layer_specialization = result["layer_specialization"]

    # ----------------------------
    # 3️⃣ Package outputs
    # ----------------------------
    basic_stats = {
        "features_array": features_array,
        "head_identifiers": head_identifiers,
        "sequence_metadata": sequence_metadata,
        "cluster_labels": cluster_labels,
        "clusterer_model": clusterer_model,
        "completeness_info": {
            "total_heads_processed": len(head_identifiers),
            "unique_sequences": len(sequence_metadata),
            "n_clusters": n_clusters,
        },
    }

    inspection_stats = {"layer_specialization": layer_specialization}

    return basic_stats, inspection_stats
