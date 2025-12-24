#!/usr/bin/env python3
"""
Detection script for out-of-domain and similarity detection.

This script:
1. Loads the database built from training set
2. Iteratively tests all samples in test set
3. Finds the most similar training dataset (by metric or feature distribution)
4. Decides whether it's out-of-domain or similar to a specific dataset
5. Detects the most relevant metrics that are different
"""

import numpy as np
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm


def load_database(db_path):
    """Load the database from pickle file."""
    with open(db_path, 'rb') as f:
        database = pickle.load(f)
    return database


def load_sample_features(feature_dir, dataset_name, utterance_id, aggregation='mean'):
    """Load features for a single sample."""
    dataset_dir = Path(feature_dir) / dataset_name
    npz_file = dataset_dir / f"{utterance_id}_encoded_feat.npz"
    
    if not npz_file.exists():
        return None
    
    try:
        data = np.load(npz_file)
        feat = data['encoded_feat']  # Shape: (L, D)
        
        if aggregation == 'mean':
            feat_agg = np.mean(feat, axis=0)
        elif aggregation == 'max':
            feat_agg = np.max(feat, axis=0)
        elif aggregation == 'last':
            feat_agg = feat[-1]
        elif aggregation == 'first':
            feat_agg = feat[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return feat_agg
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return None


def load_sample_metrics(feature_dir, dataset_name, utterance_id, use_classification=False, 
                        metric_names=None, metric_types=None, label_encoders_dict=None):
    """Load metrics for a single sample."""
    dataset_dir = Path(feature_dir) / dataset_name
    json_file = dataset_dir / f"{utterance_id}.json"
    
    if not json_file.exists():
        return None
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if metric_names is None:
            # Determine metric names from data
            metric_names = sorted(data.keys())
            metric_types = {}
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        metric_types[key] = 'numerical'
                    else:
                        metric_types[key] = 'categorical'
        
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
                    if label_encoders_dict and metric_name in label_encoders_dict:
                        # Reconstruct label encoder from stored classes
                        le_info = label_encoders_dict[metric_name]
                        classes = le_info['classes']
                        try:
                            # Find index of value in classes
                            idx = classes.index(str(value))
                            metric_vector.append(idx)
                        except (ValueError, KeyError):
                            # Value not in training classes, use 0
                            metric_vector.append(0)
                    else:
                        metric_vector.append(0)
        
        return np.array(metric_vector)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def compute_similarity_by_features(sample_feat, database, similarity_metric='cosine'):
    """
    Compute similarity between sample features and training datasets.
    
    Args:
        sample_feat: numpy array of shape (D,)
        database: database dict
        similarity_metric: 'cosine' or 'euclidean'
    
    Returns:
        similarities: dict mapping dataset_name to similarity score
    """
    similarities = {}
    
    for dataset_name, feat_stats in database['feature_stats'].items():
        train_mean = feat_stats['mean']
        train_std = feat_stats['std']
        
        if similarity_metric == 'cosine':
            # For cosine similarity, compare sample directly to training mean
            # Normalize both to unit vectors for proper cosine similarity
            sample_norm = sample_feat / (np.linalg.norm(sample_feat) + 1e-8)
            train_mean_norm = train_mean / (np.linalg.norm(train_mean) + 1e-8)
            sim = cosine_similarity([sample_norm], [train_mean_norm])[0, 0]
            
            # Also consider distance from mean in original space (as a penalty)
            # This helps distinguish samples that are far from the mean even if direction is similar
            normalized_sample = (sample_feat - train_mean) / (train_std + 1e-8)
            distance = np.linalg.norm(normalized_sample)
            # Combine: high cosine similarity and low distance = high similarity
            sim = sim / (1 + distance * 0.1)  # Penalize large distances
        else:  # euclidean
            # For euclidean, use normalized space (distance from mean)
            normalized_sample = (sample_feat - train_mean) / (train_std + 1e-8)
            normalized_train_mean = np.zeros_like(train_mean)  # After normalization, mean is 0
            # Euclidean distance in normalized space (lower is better)
            dist = np.linalg.norm(normalized_sample - normalized_train_mean)
            sim = 1 / (1 + dist)  # Convert distance to similarity
        
        similarities[dataset_name] = sim
    
    return similarities


def compute_similarity_by_metrics(sample_metrics, database, metric_mode='numerical_only', 
                                   similarity_metric='cosine'):
    """
    Compute similarity between sample metrics and training datasets.
    
    Args:
        sample_metrics: numpy array of shape (M,)
        database: database dict
        metric_mode: 'numerical_only', 'with_classification', 'numerical_only_normalized', 'with_classification_normalized'
        similarity_metric: 'cosine' or 'euclidean'
    
    Returns:
        similarities: dict mapping dataset_name to similarity score
    """
    similarities = {}
    
    metric_stats = database['metric_stats'].get(metric_mode, {})
    if not metric_stats:
        return similarities
    
    for dataset_name, stats_dict in metric_stats.items():
        train_mean = stats_dict['mean']
        
        if similarity_metric == 'cosine':
            sim = cosine_similarity([sample_metrics], [train_mean])[0, 0]
            # Also consider distance
            distance = np.linalg.norm(sample_metrics - train_mean)
            sim = sim / (1 + distance * 0.1)
        else:  # euclidean
            dist = np.linalg.norm(sample_metrics - train_mean)
            sim = 1 / (1 + dist)
        
        similarities[dataset_name] = sim
    
    return similarities


def aggregate_test_samples_features(sample_features_list, aggregation='mean'):
    """
    Aggregate multiple test sample features into a single representation.
    
    Args:
        sample_features_list: list of numpy arrays, each of shape (D,)
        aggregation: 'mean', 'max', 'min', 'median', or 'concat'
    
    Returns:
        aggregated_feat: numpy array of shape (D,) or (D*k,) if concat
    """
    if len(sample_features_list) == 0:
        return None
    
    sample_features_array = np.array(sample_features_list)  # Shape: (k, D)
    
    if aggregation == 'mean':
        return np.mean(sample_features_array, axis=0)
    elif aggregation == 'max':
        return np.max(sample_features_array, axis=0)
    elif aggregation == 'min':
        return np.min(sample_features_array, axis=0)
    elif aggregation == 'median':
        return np.median(sample_features_array, axis=0)
    elif aggregation == 'concat':
        return sample_features_array.flatten()  # Shape: (k*D,)
    else:
        return np.mean(sample_features_array, axis=0)


def aggregate_test_samples_metrics(sample_metrics_list, aggregation='mean'):
    """
    Aggregate multiple test sample metrics into a single representation.
    
    Args:
        sample_metrics_list: list of numpy arrays, each of shape (M,)
        aggregation: 'mean', 'max', 'min', 'median', or 'concat'
    
    Returns:
        aggregated_metrics: numpy array of shape (M,) or (M*k,) if concat
    """
    if len(sample_metrics_list) == 0:
        return None
    
    sample_metrics_array = np.array(sample_metrics_list)  # Shape: (k, M)
    
    if aggregation == 'mean':
        return np.mean(sample_metrics_array, axis=0)
    elif aggregation == 'max':
        return np.max(sample_metrics_array, axis=0)
    elif aggregation == 'min':
        return np.min(sample_metrics_array, axis=0)
    elif aggregation == 'median':
        return np.median(sample_metrics_array, axis=0)
    elif aggregation == 'concat':
        return sample_metrics_array.flatten()  # Shape: (k*M,)
    else:
        return np.mean(sample_metrics_array, axis=0)


def compute_similarity_by_raw_features_direct(sample_feat, database, similarity_metric='cosine',
                                               aggregation='mean'):
    """
    Compute similarity using raw features directly (all training examples).
    
    Args:
        sample_feat: numpy array of shape (D,)
        database: database dict
        similarity_metric: 'cosine' or 'euclidean'
        aggregation: how to aggregate similarities across examples ('mean', 'max', 'top_k_mean')
    
    Returns:
        similarities: dict mapping dataset_name to similarity score
    """
    similarities = {}
    
    utterance_features = database.get('utterance_features', {})
    if not utterance_features:
        return similarities
    
    for dataset_name, feat_data in utterance_features.items():
        train_features = feat_data['features']  # Shape: (N, D)
        
        if len(train_features) == 0:
            continue
        
        # Compute similarity to all training examples
        if similarity_metric == 'cosine':
            sims = cosine_similarity([sample_feat], train_features)[0]  # Shape: (N,)
        else:  # euclidean
            distances = euclidean_distances([sample_feat], train_features)[0]  # Shape: (N,)
            sims = 1 / (1 + distances)
        
        # Aggregate similarities
        if aggregation == 'mean':
            similarity = np.mean(sims)
        elif aggregation == 'max':
            similarity = np.max(sims)
        elif aggregation == 'top_k_mean':
            k = min(10, len(sims))
            top_k_sims = np.sort(sims)[-k:]
            similarity = np.mean(top_k_sims)
        else:
            similarity = np.mean(sims)
        
        similarities[dataset_name] = float(similarity)
    
    return similarities


def detect_metric_differences(sample_metrics, database, metric_mode='numerical_only', 
                              top_k=5, metric_names=None, best_match_dataset=None):
    """
    Detect the most relevant metrics that are different from training datasets.
    If best_match_dataset is provided, also identify metrics most similar to the match.
    
    Args:
        sample_metrics: numpy array of shape (M,)
        database: database dict
        metric_mode: metric mode to use
        top_k: number of top different metrics to return
        metric_names: list of metric names (optional)
        best_match_dataset: name of best matching dataset (optional, for similarity analysis)
    
    Returns:
        differences: list of tuples (metric_name, difference_score, train_mean, sample_value)
    """
    metric_stats = database['metric_stats'].get(metric_mode, {})
    if not metric_stats:
        return []
    
    if metric_names is None:
        metric_names = database['metadata']['metric_names'].get(metric_mode, [])
    
    # Compute average statistics across all training datasets
    all_means = []
    all_stds = []
    for stats_dict in metric_stats.values():
        all_means.append(stats_dict['mean'])
        all_stds.append(stats_dict['std'])
    
    if len(all_means) == 0:
        return []
    
    avg_mean = np.mean(all_means, axis=0)
    avg_std = np.mean(all_stds, axis=0)
    
    # If best_match_dataset is provided, use its statistics for comparison
    # Otherwise use average across all datasets
    if best_match_dataset and best_match_dataset in metric_stats:
        match_stats = metric_stats[best_match_dataset]
        comparison_mean = match_stats['mean']
        comparison_std = match_stats.get('std', avg_std)
    else:
        comparison_mean = avg_mean
        comparison_std = avg_std
    
    # Compute normalized differences (z-scores)
    differences = []
    for i, metric_name in enumerate(metric_names):
        if i >= len(sample_metrics):
            continue
        
        sample_val = sample_metrics[i]
        comp_mean = comparison_mean[i]
        comp_std = comparison_std[i] + 1e-8
        
        # Z-score: how many standard deviations away
        z_score = abs((sample_val - comp_mean) / comp_std)
        
        differences.append((metric_name, z_score, comp_mean, sample_val))
    
    # Sort by difference score and return top_k
    differences.sort(key=lambda x: x[1], reverse=True)
    return differences[:top_k]


def detect_metric_similarities(sample_metrics, database, metric_mode='numerical_only',
                               top_k=5, metric_names=None, best_match_dataset=None):
    """
    Detect the most similar metrics to the best match dataset (for in-domain samples).
    This helps identify which metrics are leading to the match.
    
    Args:
        sample_metrics: numpy array of shape (M,)
        database: database dict
        metric_mode: metric mode to use
        top_k: number of top similar metrics to return
        metric_names: list of metric names (optional)
        best_match_dataset: name of best matching dataset (required)
    
    Returns:
        similarities: list of tuples (metric_name, similarity_score, match_mean, sample_value)
    """
    metric_stats = database['metric_stats'].get(metric_mode, {})
    if not metric_stats:
        return []
    
    if metric_names is None:
        metric_names = database['metadata']['metric_names'].get(metric_mode, [])
    
    if not best_match_dataset or best_match_dataset not in metric_stats:
        return []
    
    match_stats = metric_stats[best_match_dataset]
    match_mean = match_stats['mean']
    match_std = match_stats.get('std', np.ones_like(match_mean))
    
    # Compute similarities (inverse of normalized distance)
    similarities = []
    for i, metric_name in enumerate(metric_names):
        if i >= len(sample_metrics):
            continue
        
        sample_val = sample_metrics[i]
        match_val = match_mean[i]
        std_val = match_std[i] + 1e-8
        
        # Normalized distance (lower is more similar)
        normalized_distance = abs((sample_val - match_val) / std_val)
        # Convert to similarity score (higher is more similar)
        # Use inverse with smoothing: similarity = 1 / (1 + distance)
        similarity_score = 1.0 / (1.0 + normalized_distance)
        
        similarities.append((metric_name, similarity_score, match_val, sample_val))
    
    # Sort by similarity score (highest first) and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def detect_out_of_domain(similarities, threshold=0.5):
    """
    Decide if sample is out-of-domain based on similarity scores.
    
    Args:
        similarities: dict mapping dataset_name to similarity score
        threshold: similarity threshold below which considered out-of-domain
    
    Returns:
        is_ood: bool, whether out-of-domain
        best_match: tuple (dataset_name, similarity_score) or None
    """
    if not similarities:
        return True, None
    
    best_dataset = max(similarities.items(), key=lambda x: x[1])
    best_score = best_dataset[1]
    
    is_ood = bool(best_score < threshold)  # Ensure Python bool, not numpy bool_
    return is_ood, best_dataset


def process_test_samples(feature_dir, database, test_datasets, detection_mode='metric', 
                         metric_mode='numerical_only', similarity_metric='cosine', 
                         ood_threshold=0.5, top_k_metrics=5, k_shot=5, 
                         raw_features_aggregation='mean', fewshot_aggregation='mean'):
    """
    Process all test samples and detect similarities/differences.
    
    Args:
        feature_dir: Path to feature directory
        database: database dict
        test_datasets: list of test dataset names
        detection_mode: 'metric', 'feature', 'fewshot', or 'raw_features'
        metric_mode: metric mode to use (if detection_mode='metric')
        similarity_metric: 'cosine' or 'euclidean'
        ood_threshold: threshold for out-of-domain detection
        top_k_metrics: number of top different metrics to report
        k_shot: number of test samples to aggregate together for few-shot learning
        raw_features_aggregation: how to aggregate similarities for raw_features mode
        fewshot_aggregation: how to aggregate multiple test samples ('mean', 'max', 'min', 'median', 'concat')
    
    Returns:
        results: list of dicts with detection results
    """
    feature_dir = Path(feature_dir)
    results = []
    
    # Get label encoders from database if needed for classification metrics
    label_encoders_dict = None
    if 'with_classification' in metric_mode:
        label_encoders_dict = database['metadata'].get('label_encoders', None)
    
    # Get metric names and types
    metric_names = database['metadata']['metric_names'].get(metric_mode, None)
    metric_types = database['metadata']['metric_types'].get(metric_mode, None)
    
    # Process each test dataset
    for dataset_name in test_datasets:
        dataset_dir = feature_dir / dataset_name
        if not dataset_dir.exists():
            continue
        
        # Get all samples in this dataset
        json_files = list(dataset_dir.glob("*.json"))
        
        print(f"\nProcessing {dataset_name} ({len(json_files)} samples)...")
        
        # For few-shot mode, process in batches
        if detection_mode == 'fewshot':
            # Group samples into batches of k_shot
            for batch_start in tqdm(range(0, len(json_files), k_shot), desc=f"Processing {dataset_name} (few-shot batches)"):
                batch_files = json_files[batch_start:batch_start + k_shot]
                batch_utterance_ids = [f.stem for f in batch_files]
                
                # Collect features/metrics for all samples in batch
                batch_features = []
                batch_metrics = []
                valid_samples = []
                
                for json_file in batch_files:
                    utterance_id = json_file.stem
                    
                    # Load features for few-shot aggregation
                    sample_feat = load_sample_features(feature_dir, dataset_name, utterance_id)
                    if sample_feat is not None:
                        batch_features.append(sample_feat)
                        valid_samples.append(utterance_id)
                
                if len(batch_features) == 0:
                    # No valid samples in batch, skip
                    for utterance_id in batch_utterance_ids:
                        result = {
                            'dataset': dataset_name,
                            'utterance_id': utterance_id,
                            'detection_mode': detection_mode,
                            'error': 'Failed to load features for batch'
                        }
                        results.append(result)
                    continue
                
                # Aggregate batch features
                aggregated_feat = aggregate_test_samples_features(batch_features, aggregation=fewshot_aggregation)
                
                if aggregated_feat is None:
                    for utterance_id in batch_utterance_ids:
                        result = {
                            'dataset': dataset_name,
                            'utterance_id': utterance_id,
                            'detection_mode': detection_mode,
                            'error': 'Failed to aggregate features'
                        }
                        results.append(result)
                    continue
                
                # Compute similarities using aggregated representation
                # Try raw features first, fall back to feature statistics
                similarities = compute_similarity_by_raw_features_direct(
                    aggregated_feat, database, similarity_metric=similarity_metric,
                    aggregation=raw_features_aggregation
                )
                
                # If raw features not available, use feature statistics
                if not similarities:
                    similarities = compute_similarity_by_features(
                        aggregated_feat, database, similarity_metric=similarity_metric
                    )
                
                # Detect out-of-domain
                is_ood, best_match = detect_out_of_domain(similarities, threshold=ood_threshold)
                
                # Get top similar datasets
                sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                top_similar = [
                    {'dataset': name, 'similarity': float(score)}
                    for name, score in sorted_similarities[:5]
                ]
                
                # Assign same result to all samples in batch
                for utterance_id in valid_samples:
                    result = {
                        'dataset': dataset_name,
                        'utterance_id': utterance_id,
                        'detection_mode': detection_mode,
                        'metric_mode': None,
                        'similarity_metric': similarity_metric,
                        'k_shot': k_shot,
                        'fewshot_aggregation': fewshot_aggregation,
                        'batch_size': len(batch_features),
                        'is_out_of_domain': bool(is_ood),
                        'best_match': {
                            'dataset': best_match[0] if best_match else None,
                            'similarity': float(best_match[1]) if best_match else None
                        },
                        'top_similar_datasets': top_similar
                    }
                    results.append(result)
                
                # Handle any samples that couldn't be loaded
                for utterance_id in batch_utterance_ids:
                    if utterance_id not in valid_samples:
                        result = {
                            'dataset': dataset_name,
                            'utterance_id': utterance_id,
                            'detection_mode': detection_mode,
                            'error': 'Failed to load features'
                        }
                        results.append(result)
        
        else:
            # Process samples one by one (original behavior)
            for json_file in tqdm(json_files, desc=f"Processing {dataset_name}"):
                utterance_id = json_file.stem
                
                result = {
                    'dataset': dataset_name,
                    'utterance_id': utterance_id,
                    'detection_mode': detection_mode,
                    'metric_mode': metric_mode if detection_mode == 'metric' else None,
                    'similarity_metric': similarity_metric
                }
                
                # Load sample data
                if detection_mode == 'feature':
                    sample_feat = load_sample_features(feature_dir, dataset_name, utterance_id)
                    if sample_feat is None:
                        result['error'] = 'Failed to load features'
                        results.append(result)
                        continue
                    
                    # Compute similarities using dataset statistics
                    similarities = compute_similarity_by_features(
                        sample_feat, database, similarity_metric=similarity_metric
                    )
                    
                    # Detect out-of-domain
                    is_ood, best_match = detect_out_of_domain(similarities, threshold=ood_threshold)
                    best_match_dataset = best_match[0] if best_match else None
                    
                elif detection_mode == 'raw_features':
                    # Direct comparison with raw features
                    sample_feat = load_sample_features(feature_dir, dataset_name, utterance_id)
                    if sample_feat is None:
                        result['error'] = 'Failed to load features'
                        results.append(result)
                        continue
                    
                    # Compute similarities using raw features directly
                    similarities = compute_similarity_by_raw_features_direct(
                        sample_feat, database, similarity_metric=similarity_metric,
                        aggregation=raw_features_aggregation
                    )
                    result['raw_features_aggregation'] = raw_features_aggregation
                    
                    # Detect out-of-domain
                    is_ood, best_match = detect_out_of_domain(similarities, threshold=ood_threshold)
                    best_match_dataset = best_match[0] if best_match else None
                
                else:  # metric mode
                    sample_metrics = load_sample_metrics(
                        feature_dir, dataset_name, utterance_id,
                        use_classification='classification' in metric_mode,
                        metric_names=metric_names,
                        metric_types=metric_types,
                        label_encoders_dict=label_encoders_dict
                    )
                    if sample_metrics is None:
                        result['error'] = 'Failed to load metrics'
                        results.append(result)
                        continue
                    
                    # Normalize if needed
                    if 'normalized' in metric_mode:
                        if 'numerical_only' in metric_mode:
                            scaler_info = database['metadata'].get('numerical_only_scaler', None)
                        else:
                            scaler_info = database['metadata'].get('with_classification_scaler', None)
                        
                        if scaler_info:
                            sample_metrics = (sample_metrics - scaler_info['mean']) / (scaler_info['scale'] + 1e-8)
                    
                    # Compute similarities
                    similarities = compute_similarity_by_metrics(
                        sample_metrics, database, metric_mode=metric_mode,
                        similarity_metric=similarity_metric
                    )
                    
                    # Detect out-of-domain first to know if we should look for similarities or differences
                    is_ood, best_match = detect_out_of_domain(similarities, threshold=ood_threshold)
                    best_match_dataset = best_match[0] if best_match else None
                    
                    # For in-domain matches: find most similar metrics (leading to the match)
                    # For OOD: find most different metrics (leading to OOD classification)
                    if not is_ood and best_match_dataset:
                        # In-domain: identify metrics most similar to best match
                        metric_similarities = detect_metric_similarities(
                            sample_metrics, database, metric_mode=metric_mode,
                            top_k=top_k_metrics, metric_names=metric_names,
                            best_match_dataset=best_match_dataset
                        )
                        result['metric_similarities'] = [
                            {
                                'metric': name,
                                'similarity_score': float(score),
                                'match_mean': float(match_mean),
                                'sample_value': float(sample_val)
                            }
                            for name, score, match_mean, sample_val in metric_similarities
                        ]
                        
                        # Also include differences for context (compared to best match)
                        metric_differences = detect_metric_differences(
                            sample_metrics, database, metric_mode=metric_mode,
                            top_k=top_k_metrics, metric_names=metric_names,
                            best_match_dataset=best_match_dataset
                        )
                        result['metric_differences'] = [
                            {
                                'metric': name,
                                'z_score': float(score),
                                'match_mean': float(train_mean),
                                'sample_value': float(sample_val)
                            }
                            for name, score, train_mean, sample_val in metric_differences
                        ]
                    else:
                        # OOD: identify metrics most different from training data
                        metric_differences = detect_metric_differences(
                            sample_metrics, database, metric_mode=metric_mode,
                            top_k=top_k_metrics, metric_names=metric_names,
                            best_match_dataset=None  # Compare to average training data
                        )
                        result['metric_differences'] = [
                            {
                                'metric': name,
                                'z_score': float(score),
                                'train_mean': float(train_mean),
                                'sample_value': float(sample_val)
                            }
                            for name, score, train_mean, sample_val in metric_differences
                        ]
                
                # Store OOD detection result
                result['is_out_of_domain'] = bool(is_ood)  # Ensure Python bool, not numpy bool_
                result['best_match'] = {
                    'dataset': best_match_dataset,
                    'similarity': float(best_match[1]) if best_match else None
                }
                
                # Get top similar datasets
                sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                result['top_similar_datasets'] = [
                    {'dataset': name, 'similarity': float(score)}
                    for name, score in sorted_similarities[:5]
                ]
                
                results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Detect out-of-domain samples and similarities")
    parser.add_argument("--database", type=str, required=True,
                       help="Path to database pickle file")
    parser.add_argument("--feature_dir", type=str, default="feature",
                       help="Path to feature directory")
    parser.add_argument("--output", type=str, default="detection_results.json",
                       help="Output path for detection results (JSON)")
    parser.add_argument("--detection_mode", type=str, 
                       choices=['metric', 'feature', 'fewshot', 'raw_features'], default='metric',
                       help="Detection mode: 'metric', 'feature', 'fewshot', or 'raw_features'")
    parser.add_argument("--metric_mode", type=str, 
                       choices=['numerical_only', 'with_classification', 
                               'numerical_only_normalized', 'with_classification_normalized'],
                       default='numerical_only',
                       help="Metric mode to use (only if detection_mode='metric')")
    parser.add_argument("--similarity_metric", type=str, choices=['cosine', 'euclidean'], 
                       default='cosine',
                       help="Similarity metric to use")
    parser.add_argument("--ood_threshold", type=float, default=0.5,
                       help="Threshold for out-of-domain detection (0-1)")
    parser.add_argument("--top_k_metrics", type=int, default=5,
                       help="Number of top different metrics to report")
    parser.add_argument("--k_shot", type=int, default=5,
                       help="Number of test samples to aggregate together for few-shot learning (if detection_mode='fewshot')")
    parser.add_argument("--fewshot_aggregation", type=str,
                       choices=['mean', 'max', 'min', 'median', 'concat'], default='mean',
                       help="How to aggregate multiple test samples for few-shot learning")
    parser.add_argument("--raw_features_aggregation", type=str, 
                       choices=['mean', 'max', 'top_k_mean'], default='mean',
                       help="How to aggregate similarities for raw_features mode")
    parser.add_argument("--test_datasets", type=str, nargs='+', default=None,
                       help="Optional: explicitly specify test datasets (otherwise from database)")
    
    args = parser.parse_args()
    
    # Load database
    print(f"Loading database from {args.database}...")
    database = load_database(args.database)
    
    # Get test datasets
    if args.test_datasets:
        test_datasets = args.test_datasets
    else:
        test_datasets = database['split_info']['test_datasets']
    
    print(f"Test datasets: {test_datasets}")
    print(f"Detection mode: {args.detection_mode}")
    if args.detection_mode == 'metric':
        print(f"Metric mode: {args.metric_mode}")
    elif args.detection_mode == 'fewshot':
        print(f"K-shot: {args.k_shot}")
        print(f"Few-shot aggregation: {args.fewshot_aggregation}")
    elif args.detection_mode == 'raw_features':
        print(f"Raw features aggregation: {args.raw_features_aggregation}")
    print(f"Similarity metric: {args.similarity_metric}")
    print(f"OOD threshold: {args.ood_threshold}")
    
    # Process test samples
    results = process_test_samples(
        args.feature_dir, database, test_datasets,
        detection_mode=args.detection_mode,
        metric_mode=args.metric_mode,
        similarity_metric=args.similarity_metric,
        ood_threshold=args.ood_threshold,
        top_k_metrics=args.top_k_metrics,
        k_shot=args.k_shot,
        raw_features_aggregation=args.raw_features_aggregation,
        fewshot_aggregation=args.fewshot_aggregation
    )
    
    # Print summary statistics (before conversion)
    print(f"\nTotal samples processed: {len(results)}")
    
    ood_count = sum(1 for r in results if r.get('is_out_of_domain', False))
    print(f"Out-of-domain samples: {ood_count} ({ood_count/len(results)*100:.1f}%)")
    
    # Print most common best matches
    best_matches = defaultdict(int)
    for r in results:
        if r.get('best_match') and r['best_match']['dataset']:
            best_matches[r['best_match']['dataset']] += 1
    
    print("\nMost common best matches:")
    for dataset, count in sorted(best_matches.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {dataset}: {count} samples")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    # Convert all results to native types
    results = convert_to_native(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

