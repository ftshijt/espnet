#!/usr/bin/env python3
"""
Visualization script for features and metrics using t-SNE.

This script supports:
1. Feature-based visualization from npz files
2. Metric-based visualization from JSON files
"""

import numpy as np
import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_domains_csv(csv_path):
    """
    Load domain and language information from CSV file.
    
    Args:
        csv_path: Path to domains.csv file
    
    Returns:
        dataset_to_domain: dict mapping dataset name to domain
        dataset_to_language: dict mapping dataset name to language_group
    """
    dataset_to_domain = {}
    dataset_to_language = {}
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Warning: CSV file not found at {csv_path}")
        return dataset_to_domain, dataset_to_language
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset']
            domain = row['domain']
            language = row['language_group']
            dataset_to_domain[dataset] = domain
            dataset_to_language[dataset] = language
    
    print(f"Loaded domain/language info for {len(dataset_to_domain)} datasets")
    return dataset_to_domain, dataset_to_language


def load_features_from_npz(feature_dir, aggregation='mean', max_samples_per_dataset=None):
    """
    Load features from npz files and aggregate (L, D) -> (D).
    
    Args:
        feature_dir: Path to feature directory with dataset subdirectories
        aggregation: Method to aggregate (L, D) -> (D)
            - 'mean': Average over time dimension
            - 'max': Max pooling over time dimension
            - 'last': Take last frame
            - 'first': Take first frame
            - 'raw': Keep raw (L, D) and flatten to (L*D,)
        max_samples_per_dataset: Maximum number of samples to load per dataset (None for no limit)
    
    Returns:
        features: numpy array of shape (N, D) or (N, L*D) for raw
        labels: list of dataset names
        utterance_ids: list of utterance IDs
    """
    feature_dir = Path(feature_dir)
    features = []
    labels = []
    utterance_ids = []
    
    # Find all dataset subdirectories
    dataset_dirs = [d for d in feature_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(dataset_dirs)} datasets")
    
    for dataset_dir in tqdm(dataset_dirs, desc="Loading features"):
        dataset_name = dataset_dir.name
        npz_files = list(dataset_dir.glob("*_encoded_feat.npz"))
        
        # Apply limit if specified
        if max_samples_per_dataset is not None and len(npz_files) > max_samples_per_dataset:
            npz_files = npz_files[:max_samples_per_dataset]
            print(f"  Limited {dataset_name} to {max_samples_per_dataset} samples")
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file)
                feat = data['encoded_feat']  # Shape: (L, D)
                
                if aggregation == 'mean':
                    feat_agg = np.mean(feat, axis=0)  # (D,)
                elif aggregation == 'max':
                    feat_agg = np.max(feat, axis=0)  # (D,)
                elif aggregation == 'last':
                    feat_agg = feat[-1]  # (D,)
                elif aggregation == 'first':
                    feat_agg = feat[0]  # (D,)
                elif aggregation == 'raw':
                    feat_agg = feat.flatten()  # (L*D,)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation}")
                
                features.append(feat_agg)
                labels.append(dataset_name)
                utterance_ids.append(npz_file.stem.replace('_encoded_feat', ''))
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
                continue
    
    if len(features) == 0:
        raise ValueError("No features loaded. Check that npz files exist and contain 'encoded_feat' key.")
    
    features_array = np.array(features)
    print(f"Loaded {len(features_array)} feature vectors with dimension {features_array.shape[1]}")
    
    return features_array, labels, utterance_ids


def calculate_dataset_statistics(features, labels):
    """
    Calculate statistics per dataset and return aggregated features.
    
    Args:
        features: numpy array of shape (N, D)
        labels: list of dataset names
    
    Returns:
        aggregated_features: numpy array of shape (num_datasets, D*num_stats)
        aggregated_labels: list of dataset names
    """
    dataset_features = defaultdict(list)
    
    for feat, label in zip(features, labels):
        dataset_features[label].append(feat)
    
    aggregated_features = []
    aggregated_labels = []
    
    for dataset_name, feat_list in dataset_features.items():
        feat_array = np.array(feat_list)  # (num_samples, D)
        
        # Calculate statistics: mean, std, min, max
        mean_feat = np.mean(feat_array, axis=0)
        std_feat = np.std(feat_array, axis=0)
        min_feat = np.min(feat_array, axis=0)
        max_feat = np.max(feat_array, axis=0)
        
        # Concatenate statistics
        stats_feat = np.concatenate([mean_feat, std_feat, min_feat, max_feat])
        aggregated_features.append(stats_feat)
        aggregated_labels.append(dataset_name)
    
    return np.array(aggregated_features), aggregated_labels


def load_metrics_from_json(feature_dir, use_classification=False, normalize=False, max_samples_per_dataset=None):
    """
    Load metrics from JSON files.
    
    Args:
        feature_dir: Path to feature directory with dataset subdirectories
        use_classification: If True, encode categorical metrics as integers
        normalize: If True, normalize numerical metrics
        max_samples_per_dataset: Maximum number of samples to load per dataset (None for no limit)
    
    Returns:
        metrics: numpy array of shape (N, M) where M is number of metrics
        labels: list of dataset names
        utterance_ids: list of utterance IDs
        metric_names: list of metric names
    """
    feature_dir = Path(feature_dir)
    all_metrics = []
    labels = []
    utterance_ids = []
    metric_names = None
    
    # Find all dataset subdirectories
    dataset_dirs = [d for d in feature_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(dataset_dirs)} datasets")
    
    # First pass: collect all metric names and determine types
    metric_types = {}  # 'numerical' or 'categorical'
    sample_json = None
    
    for dataset_dir in dataset_dirs:
        json_files = list(dataset_dir.glob("*.json"))
        if json_files:
            with open(json_files[0], 'r') as f:
                sample_json = json.load(f)
            break
    
    if sample_json is None:
        raise ValueError("No JSON files found")
    
    # Determine metric types
    for key, value in sample_json.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], (int, float)):
                metric_types[key] = 'numerical'
            else:
                metric_types[key] = 'categorical'
    
    # Filter metric names based on use_classification
    if use_classification:
        metric_names = sorted(metric_types.keys())  # Use all metrics
    else:
        metric_names = sorted([m for m, t in metric_types.items() if t == 'numerical'])  # Only numerical
    
    print(f"Found {len(metric_names)} metrics: {len([m for m, t in metric_types.items() if t == 'numerical'])} numerical, "
          f"{len([m for m, t in metric_types.items() if t == 'categorical'])} categorical")
    print(f"Using {len(metric_names)} metrics for visualization")
    
    # Create label encoders for categorical metrics
    label_encoders = {}
    if use_classification:
        # Collect all unique values for each categorical metric
        categorical_values = defaultdict(set)
        for dataset_dir in dataset_dirs:
            json_files = list(dataset_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    for key, value in data.items():
                        if metric_types.get(key) == 'categorical' and isinstance(value, list) and len(value) > 0:
                            categorical_values[key].add(str(value[0]))
                except:
                    continue
        
        # Create label encoders
        for key, values in categorical_values.items():
            le = LabelEncoder()
            le.fit(sorted(list(values)))
            label_encoders[key] = le
    
    # Load all metrics
    for dataset_dir in tqdm(dataset_dirs, desc="Loading metrics"):
        dataset_name = dataset_dir.name
        json_files = list(dataset_dir.glob("*.json"))
        
        # Apply limit if specified
        if max_samples_per_dataset is not None and len(json_files) > max_samples_per_dataset:
            json_files = json_files[:max_samples_per_dataset]
            print(f"  Limited {dataset_name} to {max_samples_per_dataset} samples")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                metric_vector = []
                for metric_name in metric_names:
                    value = data.get(metric_name, [None])[0] if metric_name in data else None
                    
                    if metric_types[metric_name] == 'numerical':
                        if value is None:
                            metric_vector.append(0.0)
                        else:
                            metric_vector.append(float(value))
                    else:  # categorical (only included if use_classification=True)
                        if value is None:
                            metric_vector.append(0)
                        else:
                            le = label_encoders[metric_name]
                            metric_vector.append(le.transform([str(value)])[0])
                
                all_metrics.append(metric_vector)
                labels.append(dataset_name)
                utterance_ids.append(json_file.stem)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
    
    if len(all_metrics) == 0:
        raise ValueError("No metrics loaded. Check that JSON files exist and contain valid data.")
    
    metrics = np.array(all_metrics)
    print(f"Loaded {len(metrics)} metric vectors with {metrics.shape[1]} dimensions")
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        metrics = scaler.fit_transform(metrics)
        print("Metrics normalized using StandardScaler")
    
    return metrics, labels, utterance_ids, metric_names


def calculate_metric_statistics(metrics, labels):
    """
    Calculate statistics per dataset for metrics.
    
    Args:
        metrics: numpy array of shape (N, M)
        labels: list of dataset names
    
    Returns:
        aggregated_metrics: numpy array of shape (num_datasets, M*num_stats)
        aggregated_labels: list of dataset names
    """
    dataset_metrics = defaultdict(list)
    
    for metric, label in zip(metrics, labels):
        dataset_metrics[label].append(metric)
    
    aggregated_metrics = []
    aggregated_labels = []
    
    for dataset_name, metric_list in dataset_metrics.items():
        metric_array = np.array(metric_list)  # (num_samples, M)
        
        # Calculate statistics: mean, std, min, max
        mean_metric = np.mean(metric_array, axis=0)
        std_metric = np.std(metric_array, axis=0)
        min_metric = np.min(metric_array, axis=0)
        max_metric = np.max(metric_array, axis=0)
        
        # Concatenate statistics
        stats_metric = np.concatenate([mean_metric, std_metric, min_metric, max_metric])
        aggregated_metrics.append(stats_metric)
        aggregated_labels.append(dataset_name)
    
    return np.array(aggregated_metrics), aggregated_labels


def plot_tsne(features, labels, output_path, title="t-SNE Visualization", perplexity=30, n_iter=1000, 
              color_by='dataset', dataset_to_domain=None, dataset_to_language=None):
    """
    Plot t-SNE visualization with different color schemes.
    
    Args:
        features: numpy array of shape (N, D)
        labels: list of labels (dataset names)
        output_path: path to save the plot
        title: plot title
        perplexity: t-SNE perplexity parameter
        n_iter: number of iterations
        color_by: 'dataset', 'domain', or 'language' - how to colorize points
        dataset_to_domain: dict mapping dataset name to domain (for color_by='domain')
        dataset_to_language: dict mapping dataset name to language (for color_by='language')
    """
    print(f"Running t-SNE on {len(features)} samples with dimension {features.shape[1]}...")
    print(f"Perplexity: {perplexity}, Iterations: {n_iter}")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=1)
    features_2d = tsne.fit_transform(features)
    
    # Determine color labels based on color_by parameter
    if color_by == 'dataset':
        color_labels = labels
    elif color_by == 'domain':
        if dataset_to_domain is None:
            raise ValueError("dataset_to_domain must be provided when color_by='domain'")
        color_labels = [dataset_to_domain.get(label, 'unknown') for label in labels]
    elif color_by == 'language':
        if dataset_to_language is None:
            raise ValueError("dataset_to_language must be provided when color_by='language'")
        color_labels = [dataset_to_language.get(label, 'unknown') for label in labels]
    else:
        raise ValueError(f"color_by must be 'dataset', 'domain', or 'language', got '{color_by}'")
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and assign colors
    unique_labels = sorted(set(color_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    if len(unique_labels) > 20:
        # Use a different colormap if we have more than 20 unique labels
        colors = plt.cm.tab20b(np.linspace(0, 1, min(len(unique_labels), 40)))
        if len(unique_labels) > 40:
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Plot each group
    for label in unique_labels:
        mask = np.array(color_labels) == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[label_to_color[label]], label=label, alpha=0.6, s=50)
    
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_tsne_multiple(features, labels, output_base_path, title_base="t-SNE Visualization", 
                       perplexity=30, n_iter=1000, dataset_to_domain=None, dataset_to_language=None):
    """
    Create three t-SNE plots: one colored by dataset, one by domain, one by language.
    
    Args:
        features: numpy array of shape (N, D)
        labels: list of labels (dataset names)
        output_base_path: base path for output files (will append _datasets.png, _domain.png, _languages.png)
        title_base: base title for plots
        perplexity: t-SNE perplexity parameter
        n_iter: number of iterations
        dataset_to_domain: dict mapping dataset name to domain
        dataset_to_language: dict mapping dataset name to language
    """
    # Run t-SNE once (it's expensive)
    print(f"Running t-SNE on {len(features)} samples with dimension {features.shape[1]}...")
    print(f"Perplexity: {perplexity}, Iterations: {n_iter}")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=1)
    features_2d = tsne.fit_transform(features)
    
    # Create three plots
    output_base = Path(output_base_path)
    output_base_stem = output_base.stem
    output_base_dir = output_base.parent
    
    # Plot 1: By dataset
    plot_tsne_from_2d(features_2d, labels, 
                      str(output_base_dir / f"{output_base_stem}_datasets.png"),
                      f"{title_base} (colored by dataset)",
                      color_by='dataset', dataset_to_domain=dataset_to_domain, 
                      dataset_to_language=dataset_to_language)
    
    # Plot 2: By domain
    if dataset_to_domain:
        plot_tsne_from_2d(features_2d, labels,
                          str(output_base_dir / f"{output_base_stem}_domain.png"),
                          f"{title_base} (colored by domain)",
                          color_by='domain', dataset_to_domain=dataset_to_domain,
                          dataset_to_language=dataset_to_language)
    
    # Plot 3: By language
    if dataset_to_language:
        plot_tsne_from_2d(features_2d, labels,
                          str(output_base_dir / f"{output_base_stem}_languages.png"),
                          f"{title_base} (colored by language)",
                          color_by='language', dataset_to_domain=dataset_to_domain,
                          dataset_to_language=dataset_to_language)


def plot_tsne_from_2d(features_2d, labels, output_path, title="t-SNE Visualization",
                      color_by='dataset', dataset_to_domain=None, dataset_to_language=None):
    """
    Plot t-SNE visualization from pre-computed 2D coordinates.
    
    Args:
        features_2d: numpy array of shape (N, 2) - pre-computed t-SNE coordinates
        labels: list of labels (dataset names)
        output_path: path to save the plot
        title: plot title
        color_by: 'dataset', 'domain', or 'language' - how to colorize points
        dataset_to_domain: dict mapping dataset name to domain (for color_by='domain')
        dataset_to_language: dict mapping dataset name to language (for color_by='language')
    """
    # Determine color labels based on color_by parameter
    if color_by == 'dataset':
        color_labels = labels
    elif color_by == 'domain':
        if dataset_to_domain is None:
            raise ValueError("dataset_to_domain must be provided when color_by='domain'")
        color_labels = [dataset_to_domain.get(label, 'unknown') for label in labels]
    elif color_by == 'language':
        if dataset_to_language is None:
            raise ValueError("dataset_to_language must be provided when color_by='language'")
        color_labels = [dataset_to_language.get(label, 'unknown') for label in labels]
    else:
        raise ValueError(f"color_by must be 'dataset', 'domain', or 'language', got '{color_by}'")
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and assign colors
    unique_labels = sorted(set(color_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    if len(unique_labels) > 20:
        # Use a different colormap if we have more than 20 unique labels
        colors = plt.cm.tab20b(np.linspace(0, 1, min(len(unique_labels), 40)))
        if len(unique_labels) > 40:
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Plot each group
    for label in unique_labels:
        mask = np.array(color_labels) == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[label_to_color[label]], label=label, alpha=0.6, s=50)
    
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize features or metrics using t-SNE")
    parser.add_argument("--feature_dir", type=str, required=True,
                       help="Path to feature directory with dataset subdirectories")
    parser.add_argument("--mode", type=str, choices=['features', 'metrics'], required=True,
                       help="Visualization mode: 'features' or 'metrics'")
    
    # Feature-specific arguments
    parser.add_argument("--aggregation", type=str, 
                       choices=['mean', 'max', 'last', 'first', 'raw'],
                       default='mean',
                       help="How to aggregate (L, D) -> (D) for features")
    parser.add_argument("--use_statistics", action='store_true',
                       help="Calculate statistics per dataset instead of per utterance")
    
    # Metric-specific arguments
    parser.add_argument("--use_classification", action='store_true',
                       help="Include categorical metrics as classification labels (0, 1, 2, ...)")
    parser.add_argument("--normalize", action='store_true',
                       help="Normalize metrics using StandardScaler")
    parser.add_argument("--metric_statistics", action='store_true',
                       help="Calculate statistics per dataset for metrics")
    
    # t-SNE arguments
    parser.add_argument("--perplexity", type=int, default=30,
                       help="t-SNE perplexity parameter")
    parser.add_argument("--n_iter", type=int, default=1000,
                       help="Number of t-SNE iterations")
    parser.add_argument("--output", type=str, default="tsne_visualization.png",
                       help="Output path for the plot")
    parser.add_argument("--max_samples_per_dataset", type=int, default=1000,
                       help="Maximum number of samples to load per dataset (None for no limit)")
    parser.add_argument("--domains_csv", type=str, default="/work/nvme/bbjs/shi3/evaluation/espnet/arecho_app/source_data/domains.csv",
                       help="Path to domains.csv file for domain/language coloring")
    
    args = parser.parse_args()
    
    # Load domain/language information if CSV is provided
    dataset_to_domain = None
    dataset_to_language = None
    if args.domains_csv:
        dataset_to_domain, dataset_to_language = load_domains_csv(args.domains_csv)
    
    if args.mode == 'features':
        print("=" * 60)
        print("FEATURE-BASED VISUALIZATION")
        print("=" * 60)
        
        # Load features
        features, labels, utterance_ids = load_features_from_npz(
            args.feature_dir, 
            aggregation=args.aggregation,
            max_samples_per_dataset=args.max_samples_per_dataset
        )
        print(f"Loaded {len(features)} feature vectors")
        print(f"Feature dimension: {features.shape[1]}")
        
        # Calculate statistics if requested
        if args.use_statistics:
            print("\nCalculating dataset statistics...")
            features, labels = calculate_dataset_statistics(features, labels)
            print(f"Aggregated to {len(features)} dataset-level features")
            print(f"Feature dimension: {features.shape[1]}")
        
        # Create title
        title = f"t-SNE: Features (aggregation={args.aggregation}"
        if args.use_statistics:
            title += ", dataset statistics"
        if args.max_samples_per_dataset:
            title += f", max {args.max_samples_per_dataset} per dataset"
        title += ")"
        
        # Plot - create three figures if domains CSV is provided
        if args.domains_csv and dataset_to_domain and dataset_to_language:
            plot_tsne_multiple(features, labels, args.output, title_base=title,
                              perplexity=args.perplexity, n_iter=args.n_iter,
                              dataset_to_domain=dataset_to_domain, 
                              dataset_to_language=dataset_to_language)
        else:
            # Fallback to single plot by dataset
            plot_tsne(features, labels, args.output, title=title,
                     perplexity=args.perplexity, n_iter=args.n_iter)
    
    else:  # metrics
        print("=" * 60)
        print("METRIC-BASED VISUALIZATION")
        print("=" * 60)
        
        # Load metrics
        metrics, labels, utterance_ids, metric_names = load_metrics_from_json(
            args.feature_dir,
            use_classification=args.use_classification,
            normalize=args.normalize,
            max_samples_per_dataset=args.max_samples_per_dataset
        )
        print(f"Loaded {len(metrics)} metric vectors")
        print(f"Number of metrics: {len(metric_names)}")
        print(f"Metric names: {', '.join(metric_names[:10])}{'...' if len(metric_names) > 10 else ''}")
        
        # Calculate statistics if requested
        if args.metric_statistics:
            print("\nCalculating dataset statistics for metrics...")
            metrics, labels = calculate_metric_statistics(metrics, labels)
            print(f"Aggregated to {len(metrics)} dataset-level metric vectors")
            print(f"Metric dimension: {metrics.shape[1]}")
        
        # Create title
        title = "t-SNE: Metrics"
        if args.use_classification:
            title += " (with classification)"
        if args.normalize:
            title += " (normalized)"
        if args.metric_statistics:
            title += " (dataset statistics)"
        if args.max_samples_per_dataset:
            title += f" (max {args.max_samples_per_dataset} per dataset)"
        
        # Plot - create three figures if domains CSV is provided
        if args.domains_csv and dataset_to_domain and dataset_to_language:
            plot_tsne_multiple(metrics, labels, args.output, title_base=title,
                              perplexity=args.perplexity, n_iter=args.n_iter,
                              dataset_to_domain=dataset_to_domain, 
                              dataset_to_language=dataset_to_language)
        else:
            # Fallback to single plot by dataset
            plot_tsne(metrics, labels, args.output, title=title,
                     perplexity=args.perplexity, n_iter=args.n_iter)


if __name__ == "__main__":
    main()

