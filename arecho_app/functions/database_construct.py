#!/usr/bin/env python3
"""
Database construction script for training set statistics.

This script:
1. Splits datasets into train/test (maritime datasets in test + a few others)
2. Collects comprehensive statistics from training set:
   - Feature statistics (mean/std of aggregated features)
   - Metric statistics (with/without classification, with/without normalization)
3. Saves database as binary dict file (pickle format)
"""

import numpy as np
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm


def get_train_test_split(all_datasets):
    """
    Split datasets into train and test sets.
    Maritime datasets go to test set, plus a few others for diversity.
    
    Args:
        all_datasets: list of dataset names
    
    Returns:
        train_datasets: list of training dataset names
        test_datasets: list of test dataset names
    """
    # Maritime datasets always in test
    test_datasets = [d for d in all_datasets if 'maritime' in d.lower()]
    
    # Add a few other diverse datasets to test set for better evaluation
    # Choose some that are different from maritime (e.g., different languages, domains)
    additional_test = []
    for dataset in all_datasets:
        if dataset not in test_datasets:
            # Add a few diverse datasets (e.g., non-English, different domains)
            if any(x in dataset.lower() for x in ['openslr_bengali', 'openslr_javanese', 'fleurs', 'covost2']):
                additional_test.append(dataset)
                if len(additional_test) >= 3:  # Add 3-4 additional test datasets
                    break
    
    test_datasets.extend(additional_test)
    train_datasets = [d for d in all_datasets if d not in test_datasets]
    
    return train_datasets, test_datasets


def load_features_from_npz(feature_dir, dataset_name, aggregation='mean', max_utterances=None):
    """
    Load features from npz files for a specific dataset.
    
    Args:
        feature_dir: Path to feature directory
        dataset_name: Name of dataset folder
        aggregation: Method to aggregate (L, D) -> (D): 'mean', 'max', 'last', 'first'
        max_utterances: Maximum number of utterances to load (None for all)
    
    Returns:
        features: numpy array of shape (N, D)
        utterance_ids: list of utterance IDs
    """
    dataset_dir = Path(feature_dir) / dataset_name
    if not dataset_dir.exists():
        return None, []
    
    features = []
    utterance_ids = []
    npz_files = list(dataset_dir.glob("*_encoded_feat.npz"))
    
    # Limit number of files if max_utterances is specified
    if max_utterances is not None and max_utterances > 0:
        npz_files = npz_files[:max_utterances]
    
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
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            features.append(feat_agg)
            utterance_ids.append(npz_file.stem.replace('_encoded_feat', ''))
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue
    
    if len(features) == 0:
        return None, []
    
    return np.array(features), utterance_ids


def load_metrics_from_json(feature_dir, dataset_name, use_classification=False):
    """
    Load metrics from JSON files for a specific dataset.
    
    Args:
        feature_dir: Path to feature directory
        dataset_name: Name of dataset folder
        use_classification: If True, encode categorical metrics as integers
    
    Returns:
        metrics: numpy array of shape (N, M)
        utterance_ids: list of utterance IDs
        metric_names: list of metric names
        metric_types: dict mapping metric names to 'numerical' or 'categorical'
    """
    dataset_dir = Path(feature_dir) / dataset_name
    if not dataset_dir.exists():
        return None, [], [], {}
    
    # First, determine metric types from a sample file
    json_files = list(dataset_dir.glob("*.json"))
    if not json_files:
        return None, [], [], {}
    
    # Load sample to determine structure
    with open(json_files[0], 'r') as f:
        sample_json = json.load(f)
    
    metric_types = {}
    for key, value in sample_json.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], (int, float)):
                metric_types[key] = 'numerical'
            else:
                metric_types[key] = 'categorical'
    
    # Filter metric names
    if use_classification:
        metric_names = sorted(metric_types.keys())
    else:
        metric_names = sorted([m for m, t in metric_types.items() if t == 'numerical'])
    
    # Create label encoders for categorical metrics if needed
    label_encoders = {}
    if use_classification:
        categorical_values = defaultdict(set)
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for key, value in data.items():
                    if metric_types.get(key) == 'categorical' and isinstance(value, list) and len(value) > 0:
                        categorical_values[key].add(str(value[0]))
            except:
                continue
        
        for key, values in categorical_values.items():
            le = LabelEncoder()
            le.fit(sorted(list(values)))
            label_encoders[key] = le
    
    # Load all metrics
    all_metrics = []
    utterance_ids = []
    
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
                else:  # categorical
                    if value is None:
                        metric_vector.append(0)
                    else:
                        le = label_encoders[metric_name]
                        metric_vector.append(le.transform([str(value)])[0])
            
            all_metrics.append(metric_vector)
            utterance_ids.append(json_file.stem)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    if len(all_metrics) == 0:
        return None, [], [], {}
    
    return np.array(all_metrics), utterance_ids, metric_names, metric_types


def calculate_statistics(data_array):
    """
    Calculate comprehensive statistics for an array.
    
    Args:
        data_array: numpy array of shape (N, D)
    
    Returns:
        stats: dict with mean, std, min, max, median, q25, q75
    """
    if data_array is None or len(data_array) == 0:
        return None
    
    return {
        'mean': np.mean(data_array, axis=0).astype(np.float32),
        'std': np.std(data_array, axis=0).astype(np.float32),
        'min': np.min(data_array, axis=0).astype(np.float32),
        'max': np.max(data_array, axis=0).astype(np.float32),
        'median': np.median(data_array, axis=0).astype(np.float32),
        'q25': np.percentile(data_array, 25, axis=0).astype(np.float32),
        'q75': np.percentile(data_array, 75, axis=0).astype(np.float32),
        'count': len(data_array)
    }


def build_database(feature_dir, output_path, train_datasets=None, max_utterances=None):
    """
    Build database from training datasets.
    
    Args:
        feature_dir: Path to feature directory
        output_path: Path to save database (pickle file)
        train_datasets: Optional list of training dataset names. If None, auto-split.
        max_utterances: Maximum number of utterances per dataset to use (None for all)
    """
    feature_dir = Path(feature_dir)
    
    # Get all datasets
    all_datasets = [d.name for d in feature_dir.iterdir() if d.is_dir()]
    print(f"Found {len(all_datasets)} datasets")
    
    # Split into train/test if not provided
    if train_datasets is None:
        train_datasets, test_datasets = get_train_test_split(all_datasets)
    else:
        test_datasets = [d for d in all_datasets if d not in train_datasets]
    
    print(f"\nTrain datasets ({len(train_datasets)}): {sorted(train_datasets)}")
    print(f"Test datasets ({len(test_datasets)}): {sorted(test_datasets)}")
    
    # Save train/test split info
    split_info = {
        'train_datasets': train_datasets,
        'test_datasets': test_datasets
    }
    
    # Build database from training set
    database = {
        'split_info': split_info,
        'feature_stats': {},
        'utterance_features': {},  # Store individual utterance mean features for detection
        'metric_stats': {
            'numerical_only': {},
            'with_classification': {},
            'numerical_only_normalized': {},
            'with_classification_normalized': {}
        },
        'metadata': {
            'feature_aggregation': 'mean',
            'feature_dim': None,
            'metric_names': {},
            'metric_types': {},
            'max_utterances': max_utterances
        }
    }
    
    # First pass: collect all metrics to compute global scalers and label encoders
    print("\nFirst pass: Collecting all metrics for global normalization...")
    all_metrics_num = []
    all_metrics_cls = []
    num_metric_names = None
    num_metric_types = None
    cls_metric_names = None
    cls_metric_types = None
    
    # Collect categorical values for label encoders
    categorical_values = defaultdict(set)
    
    for dataset_name in train_datasets:
        # Load numerical metrics
        metrics_num, _, num_names, num_types = load_metrics_from_json(
            feature_dir, dataset_name, use_classification=False
        )
        if metrics_num is not None and len(metrics_num) > 0:
            all_metrics_num.append(metrics_num)
            if num_metric_names is None:
                num_metric_names = num_names
                num_metric_types = num_types
        
        # Load classification metrics and collect categorical values
        metrics_cls, _, cls_names, cls_types = load_metrics_from_json(
            feature_dir, dataset_name, use_classification=True
        )
        if metrics_cls is not None and len(metrics_cls) > 0:
            all_metrics_cls.append(metrics_cls)
            if cls_metric_names is None:
                cls_metric_names = cls_names
                cls_metric_types = cls_types
            
            # Collect categorical values for label encoders
            dataset_dir = Path(feature_dir) / dataset_name
            json_files = list(dataset_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    for key, value in data.items():
                        if cls_types.get(key) == 'categorical' and isinstance(value, list) and len(value) > 0:
                            categorical_values[key].add(str(value[0]))
                except:
                    continue
    
    # Compute global scalers
    scaler_num = None
    scaler_cls = None
    if all_metrics_num:
        all_metrics_num_concat = np.vstack(all_metrics_num)
        scaler_num = StandardScaler()
        scaler_num.fit(all_metrics_num_concat)
        database['metadata']['numerical_only_scaler'] = {
            'mean': scaler_num.mean_.astype(np.float32),
            'scale': scaler_num.scale_.astype(np.float32)
        }
        database['metadata']['metric_names']['numerical_only'] = num_metric_names
        database['metadata']['metric_types']['numerical_only'] = num_metric_types
    
    if all_metrics_cls:
        all_metrics_cls_concat = np.vstack(all_metrics_cls)
        scaler_cls = StandardScaler()
        scaler_cls.fit(all_metrics_cls_concat)
        database['metadata']['with_classification_scaler'] = {
            'mean': scaler_cls.mean_.astype(np.float32),
            'scale': scaler_cls.scale_.astype(np.float32)
        }
        database['metadata']['metric_names']['with_classification'] = cls_metric_names
        database['metadata']['metric_types']['with_classification'] = cls_metric_types
        
        # Build and store label encoders for categorical metrics
        label_encoders_dict = {}
        for key, values in categorical_values.items():
            if cls_metric_types.get(key) == 'categorical':
                le = LabelEncoder()
                le.fit(sorted(list(values)))
                # Store as dict with classes for reconstruction
                label_encoders_dict[key] = {
                    'classes': le.classes_.tolist()
                }
        database['metadata']['label_encoders'] = label_encoders_dict
    
    # Second pass: process each training dataset and compute statistics
    print("\nSecond pass: Computing statistics per dataset...")
    for dataset_name in tqdm(train_datasets, desc="Building database"):
        dataset_stats = {}
        
        # Load features
        features, feat_utt_ids = load_features_from_npz(
            feature_dir, dataset_name, aggregation='mean', max_utterances=max_utterances
        )
        if features is not None and len(features) > 0:
            feat_stats = calculate_statistics(features)
            dataset_stats['features'] = feat_stats
            if database['metadata']['feature_dim'] is None:
                database['metadata']['feature_dim'] = features.shape[1]
            
            # Store individual utterance mean features for detection purposes
            # Format: {dataset_name: {'features': array(N, D), 'utterance_ids': list}}
            database['utterance_features'][dataset_name] = {
                'features': features.astype(np.float32),  # Store as float32 to save memory
                'utterance_ids': feat_utt_ids
            }
        
        # Load metrics with different configurations
        # 1. Numerical only (no classification)
        metrics_num, num_utt_ids, _, _ = load_metrics_from_json(
            feature_dir, dataset_name, use_classification=False
        )
        if metrics_num is not None and len(metrics_num) > 0:
            database['metric_stats']['numerical_only'][dataset_name] = calculate_statistics(metrics_num)
        
        # 2. Numerical only normalized (using global scaler)
        if metrics_num is not None and len(metrics_num) > 0 and scaler_num is not None:
            metrics_num_norm = scaler_num.transform(metrics_num)
            database['metric_stats']['numerical_only_normalized'][dataset_name] = calculate_statistics(metrics_num_norm)
        
        # 3. With classification (categorical + numerical)
        metrics_cls, cls_utt_ids, _, _ = load_metrics_from_json(
            feature_dir, dataset_name, use_classification=True
        )
        if metrics_cls is not None and len(metrics_cls) > 0:
            database['metric_stats']['with_classification'][dataset_name] = calculate_statistics(metrics_cls)
        
        # 4. With classification normalized (using global scaler)
        if metrics_cls is not None and len(metrics_cls) > 0 and scaler_cls is not None:
            metrics_cls_norm = scaler_cls.transform(metrics_cls)
            database['metric_stats']['with_classification_normalized'][dataset_name] = calculate_statistics(metrics_cls_norm)
        
        # Store feature stats
        if 'features' in dataset_stats:
            database['feature_stats'][dataset_name] = dataset_stats['features']
    
    # Save database
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"\nDatabase saved to {output_path}")
    print(f"Database contains:")
    print(f"  - {len(database['feature_stats'])} datasets with feature statistics")
    print(f"  - {len(database['utterance_features'])} datasets with individual utterance features")
    total_utterances = sum(len(v['utterance_ids']) for v in database['utterance_features'].values())
    print(f"  - {total_utterances} total utterances with stored mean features")
    print(f"  - {len(database['metric_stats']['numerical_only'])} datasets with numerical-only metrics")
    print(f"  - {len(database['metric_stats']['with_classification'])} datasets with classification metrics")
    print(f"  - Feature dimension: {database['metadata']['feature_dim']}")
    if max_utterances is not None:
        print(f"  - Max utterances per dataset: {max_utterances}")
    
    return database


def main():
    parser = argparse.ArgumentParser(description="Build database from training datasets")
    parser.add_argument("--feature_dir", type=str, default="feature",
                       help="Path to feature directory with dataset subdirectories")
    parser.add_argument("--output", type=str, default="database.pkl",
                       help="Output path for database (pickle file)")
    parser.add_argument("--train_datasets", type=str, nargs='+', default=None,
                       help="Optional: explicitly specify training datasets (otherwise auto-split)")
    parser.add_argument("--max_utterances", type=int, default=None,
                       help="Maximum number of utterances per dataset to use (None for all)")
    
    args = parser.parse_args()
    
    database = build_database(args.feature_dir, args.output, args.train_datasets, args.max_utterances)
    
    print("\nDatabase construction completed!")


if __name__ == "__main__":
    main()

