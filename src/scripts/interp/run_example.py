import argparse
import logging
from pathlib import Path

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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Seach neuron groups across different training steps.")
    parser.add_argument("--model", type=str, choices=["last", "clm", "mlm"], default="clm", help="model type")
    parser.add_argument("--seed_num", type=int, default=1000, help="model type")
    parser.add_argument("--n_clusters", type=int, default=3, help="model type")
    parser.add_argument(
        "--task",
        type=str,
        choices=["memorization", "id_generalization", "ood_same_rule", "ood_transfer"],
        default="memorization",
        help="task type",
    )
    parser.add_argument("--max_shots", type=int, default=None, help="model type")
    parser.add_argument("--max_seq", type=int, default=None, help="model type")
    parser.add_argument("--step", type=str, default=None, help="model type")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if available.")
    return parser.parse_args()


def extract_stat(data_dir: Path, n_clusters: int, max_shots=None, max_sequences=None):
    n_layers, n_heads = 6, 8
    batch_size = 50

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
    # print("Pre-clustering summary:", pre_analyzer.get_summary())

    # --- Layer stats & specialization ---
    layer_stats, spec_scores = layer_analyzer.run(features)
    # print("Layer stats:", layer_stats)
    # print("Specialization scores:", spec_scores)

    # --- Clustering ---
    # cluster_labels, _ = clustering_module.run(features)
    # print("[INFO] Clustering done.")

    # --- Post-clustering analysis ---
    # post_summary = post_analyzer.run(features, cluster_labels)
    # print("Post-clustering summary:", post_summary)

    # --- Inter-layer comparison ---
    # corr, diversity = inter_layer_analyzer.run(features, cluster_labels)
    # print("Inter-layer correlations:", corr)
    # print("Functional diversity:", diversity)

    step = seq_meta_batch[0]["step"]

    phase1_json = {
        step: {
            "metadata": {
                "num_layers": n_layers,
                "num_heads": n_heads,
                "num_sequences": max_sequences,
                "n_clusters": n_clusters,
                "batch_size": batch_size,
                "max_shots": max_shots,
                "max_sequences": max_sequences,
            },
            "layer_stats": layer_stats,
            "layer_spec": spec_scores,
            "features": features,
        }
    }

    # phase1_json = {
    #     step: {
    #         "metadata": {
    #             "num_layers": n_layers,
    #             "num_heads": n_heads,
    #             "num_sequences": max_sequences,
    #             "n_clusters": n_clusters,
    #             "batch_size": batch_size,
    #             "max_shots": max_shots,
    #             "max_sequences": max_sequences,
    #         },
    #         "layer_stats": layer_stats,
    #         "layer_spec": spec_scores,
    #         "inter_layer_corr": corr,
    #         "inter_layer_div": diversity,
    #         "cluster_labels": cluster_labels.tolist() if cluster_labels is not None else [],
    #         "post_summary": post_summary,
    #         "features": features,
    #     }
    # }
    return phase1_json


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    prefix = f"uniform_{args.seed_num}_L3_M3/{args.model}_noshuffle_seedbalanced"
    suffix = "collection/raw_evaluations/attention_data"
    output_file = PATH.interp_dir / prefix / "layer" / f"{args.task}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # configure save_dir
    data_dir = PATH.result_dir / prefix / args.task / suffix
    if args.resume and output_file.exists():
        logger.info(f"Reume mode and file exists: {output_file}")
        exit()

    if data_dir.exists():
        # loop over different steps
        result_dict = {}
        for step_dir in data_dir.iterdir():
            # try:
            logger.info(f"Loading file from: {step_dir}")
            phase1_json = extract_stat(
                step_dir,
                n_clusters=args.n_clusters,
                max_shots=args.max_shots,
                max_sequences=args.max_seq,
            )
            result_dict.update(phase1_json)
            JsonProcessor.save_json(result_dict, output_file)
            logger.info(f"Save the rest to: {output_file}/{args.task}.json")
    # except:
    #     logger.info(f"Fail to extract stat from: {step_dir}")
    else:
        logger.info(f"Does NOT exist: {data_dir}")


if __name__ == "__main__":
    main()
