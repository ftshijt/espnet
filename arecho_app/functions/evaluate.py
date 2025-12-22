#!/usr/bin/env python3
"""
Evaluation script for detection results.

This script evaluates:
1. Classification consistency: Are samples from the same dataset classified similarly?
2. Dataset separation: Can different test datasets be distinguished?
3. Additional metrics: Top-k accuracy, confidence scores, confusion matrix, etc.
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_path):
    """Load detection results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def load_database(db_path):
    """Load database to get train/test split info."""
    with open(db_path, 'rb') as f:
        database = pickle.load(f)
    return database


def load_domains_csv(domains_csv_path):
    """Load domains.csv to get ground truth domain and language information."""
    df = pd.read_csv(domains_csv_path)
    # Create mappings: dataset -> (domain, language)
    dataset_to_domain = {}
    dataset_to_language = {}
    for _, row in df.iterrows():
        dataset = row['dataset']
        domain = row['domain']
        language = row['language_group']
        dataset_to_domain[dataset] = domain
        dataset_to_language[dataset] = language
    return dataset_to_domain, dataset_to_language


def evaluate_ood_estimation(results, database, domains_csv_path):
    """
    Evaluate if out-of-domain is estimated correctly by comparing against domains.csv.
    
    A test dataset is truly OOD if:
    - Its domain is not present in any training dataset, OR
    - Its language is not present in any training dataset
    
    Returns:
        metrics: dict with OOD evaluation metrics
    """
    # Load domain and language mappings
    dataset_to_domain, dataset_to_language = load_domains_csv(domains_csv_path)
    
    # Get train and test datasets
    train_datasets = set(database.get('split_info', {}).get('train_datasets', []))
    test_datasets = set(r['dataset'] for r in results if 'error' not in r)
    
    # Get domains and languages in training set
    train_domains = set()
    train_languages = set()
    for dataset in train_datasets:
        if dataset in dataset_to_domain:
            train_domains.add(dataset_to_domain[dataset])
        if dataset in dataset_to_language:
            train_languages.add(dataset_to_language[dataset])
    
    # Determine ground truth OOD for each test dataset
    # A test dataset is OOD if its domain is not present in any training dataset
    true_ood_datasets = set()
    for test_dataset in test_datasets:
        test_domain = dataset_to_domain.get(test_dataset)
        
        # OOD if domain not in training set
        if test_domain and test_domain not in train_domains:
            true_ood_datasets.add(test_dataset)
    
    # Evaluate predictions
    metrics = {
        'ground_truth': {
            'train_domains': sorted(list(train_domains)),
            'train_languages': sorted(list(train_languages)),
            'true_ood_datasets': sorted(list(true_ood_datasets)),
            'true_in_domain_datasets': sorted(list(test_datasets - true_ood_datasets))
        },
        'per_dataset': {},
        'overall': {}
    }
    
    # Per-dataset evaluation
    true_positives = 0  # Correctly identified as OOD
    false_positives = 0  # Incorrectly identified as OOD
    true_negatives = 0  # Correctly identified as in-domain
    false_negatives = 0  # Incorrectly identified as in-domain
    
    for test_dataset in test_datasets:
        true_is_ood = test_dataset in true_ood_datasets
        
        # Get predictions for this dataset
        dataset_results = [r for r in results if r.get('dataset') == test_dataset and 'error' not in r]
        if not dataset_results:
            continue
        
        # Count samples at individual level
        ood_predictions = [r.get('is_out_of_domain', False) for r in dataset_results]
        num_ood_samples = sum(ood_predictions)
        num_in_domain_samples = len(ood_predictions) - num_ood_samples
        
        # Majority vote for dataset-level prediction (for per-dataset metrics)
        predicted_is_ood = sum(ood_predictions) > len(ood_predictions) / 2
        
        metrics['per_dataset'][test_dataset] = {
            'true_is_ood': true_is_ood,
            'predicted_is_ood': predicted_is_ood,
            'correct': true_is_ood == predicted_is_ood,
            'domain': dataset_to_domain.get(test_dataset, 'unknown'),
            'language': dataset_to_language.get(test_dataset, 'unknown'),
            'num_samples': len(dataset_results),
            'num_ood_samples': num_ood_samples,
            'num_in_domain_samples': num_in_domain_samples,
            'ood_ratio': num_ood_samples / len(dataset_results) if dataset_results else 0.0
        }
        
        # Update confusion matrix counts at sample level
        for predicted_ood in ood_predictions:
            if true_is_ood and predicted_ood:
                true_positives += 1
            elif true_is_ood and not predicted_ood:
                false_negatives += 1
            elif not true_is_ood and predicted_ood:
                false_positives += 1
            else:  # not true_is_ood and not predicted_ood
                true_negatives += 1
    
    # Overall metrics
    total_samples = true_positives + false_positives + true_negatives + false_negatives
    if total_samples > 0:
        accuracy = (true_positives + true_negatives) / total_samples
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['overall'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives),
            'total_samples': int(total_samples)
        }
    
    return metrics


def find_closest_mappings(results, database, domains_csv_path):
    """
    Find the closest mapping for each dataset at:
    1. Dataset level: which training dataset is most similar
    2. Domain level: which domain is most similar (aggregate similarities by domain)
    3. Language level: which language is most similar (aggregate similarities by language)
    
    Returns:
        mappings: dict with closest mappings at each level
    """
    # Load domain and language mappings
    dataset_to_domain, dataset_to_language = load_domains_csv(domains_csv_path)
    
    # Get train datasets
    train_datasets = set(database.get('split_info', {}).get('train_datasets', []))
    
    # Group results by test dataset
    dataset_results = defaultdict(list)
    for result in results:
        if 'error' in result:
            continue
        dataset = result['dataset']
        top_similar = result.get('top_similar_datasets', [])
        dataset_results[dataset].append(top_similar)
    
    mappings = {
        'dataset_level': {},
        'domain_level': {},
        'language_level': {}
    }
    
    for test_dataset, top_matches_list in dataset_results.items():
        if not top_matches_list:
            continue
        
        # Aggregate similarities across all samples in the test dataset
        dataset_similarities = defaultdict(list)
        domain_similarities = defaultdict(list)
        language_similarities = defaultdict(list)
        
        for top_matches in top_matches_list:
            for match in top_matches:
                train_dataset = match['dataset']
                similarity = match['similarity']
                
                # Only consider training datasets
                if train_dataset not in train_datasets:
                    continue
                
                # Dataset level
                dataset_similarities[train_dataset].append(similarity)
                
                # Domain level
                train_domain = dataset_to_domain.get(train_dataset)
                if train_domain:
                    domain_similarities[train_domain].append(similarity)
                
                # Language level
                train_language = dataset_to_language.get(train_dataset)
                if train_language:
                    language_similarities[train_language].append(similarity)
        
        # Find best match at dataset level
        if dataset_similarities:
            dataset_avg_sims = {ds: np.mean(sims) for ds, sims in dataset_similarities.items()}
            best_dataset = max(dataset_avg_sims.items(), key=lambda x: x[1])
            mappings['dataset_level'][test_dataset] = {
                'best_match_dataset': best_dataset[0],
                'similarity': float(best_dataset[1]),
                'num_samples': len(dataset_similarities[best_dataset[0]]),
                'all_matches': {ds: float(np.mean(sims)) for ds, sims in dataset_similarities.items()}
            }
        
        # Find best match at domain level
        if domain_similarities:
            domain_avg_sims = {dom: np.mean(sims) for dom, sims in domain_similarities.items()}
            best_domain = max(domain_avg_sims.items(), key=lambda x: x[1])
            mappings['domain_level'][test_dataset] = {
                'best_match_domain': best_domain[0],
                'similarity': float(best_domain[1]),
                'num_samples': len(domain_similarities[best_domain[0]]),
                'test_domain': dataset_to_domain.get(test_dataset, 'unknown'),
                'domain_match': best_domain[0] == dataset_to_domain.get(test_dataset, 'unknown'),
                'all_matches': {dom: float(np.mean(sims)) for dom, sims in domain_similarities.items()}
            }
        
        # Find best match at language level
        if language_similarities:
            language_avg_sims = {lang: np.mean(sims) for lang, sims in language_similarities.items()}
            best_language = max(language_avg_sims.items(), key=lambda x: x[1])
            mappings['language_level'][test_dataset] = {
                'best_match_language': best_language[0],
                'similarity': float(best_language[1]),
                'num_samples': len(language_similarities[best_language[0]]),
                'test_language': dataset_to_language.get(test_dataset, 'unknown'),
                'language_match': best_language[0] == dataset_to_language.get(test_dataset, 'unknown'),
                'all_matches': {lang: float(np.mean(sims)) for lang, sims in language_similarities.items()}
            }
    
    return mappings


def evaluate_classification_consistency(results):
    """
    Evaluate if samples from the same dataset are classified consistently.
    
    Returns:
        metrics: dict with consistency metrics
    """
    # Group results by test dataset
    dataset_results = defaultdict(list)
    for result in results:
        if 'error' in result:
            continue
        dataset = result['dataset']
        best_match = result.get('best_match', {})
        dataset_results[dataset].append({
            'utterance_id': result['utterance_id'],
            'best_match_dataset': best_match.get('dataset'),
            'similarity': best_match.get('similarity', 0.0),
            'is_ood': result.get('is_out_of_domain', False)
        })
    
    metrics = {
        'per_dataset_consistency': {},
        'overall_consistency': {}
    }
    
    # For each test dataset, check consistency
    all_consistencies = []
    for test_dataset, samples in dataset_results.items():
        if len(samples) < 2:
            continue
        
        # Count how many samples have the same best match
        best_match_counts = defaultdict(int)
        for sample in samples:
            match = sample['best_match_dataset']
            if match:
                best_match_counts[match] += 1
        
        # Most common match
        if best_match_counts:
            most_common_match = max(best_match_counts.items(), key=lambda x: x[1])
            consistency = most_common_match[1] / len(samples)
            all_consistencies.append(consistency)
            
            metrics['per_dataset_consistency'][test_dataset] = {
                'consistency': consistency,
                'num_samples': len(samples),
                'most_common_match': most_common_match[0],
                'match_distribution': dict(best_match_counts)
            }
    
    if all_consistencies:
        metrics['overall_consistency'] = {
            'mean_consistency': np.mean(all_consistencies),
            'std_consistency': np.std(all_consistencies),
            'min_consistency': np.min(all_consistencies),
            'max_consistency': np.max(all_consistencies)
        }
    
    return metrics


def evaluate_dataset_separation(results):
    """
    Evaluate if different test datasets can be separated.
    
    Uses clustering metrics and similarity analysis.
    """
    # Group results by test dataset
    dataset_results = defaultdict(list)
    for result in results:
        if 'error' in result:
            continue
        dataset = result['dataset']
        best_match = result.get('best_match', {})
        similarity = best_match.get('similarity', 0.0)
        dataset_results[dataset].append(similarity)
    
    metrics = {
        'intra_dataset_similarity': {},
        'inter_dataset_similarity': {},
        'separation_score': {}
    }
    
    datasets = list(dataset_results.keys())
    
    # Intra-dataset similarity (within same dataset)
    for dataset, similarities in dataset_results.items():
        if len(similarities) > 1:
            metrics['intra_dataset_similarity'][dataset] = {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities)
            }
    
    # Inter-dataset similarity (between different datasets)
    # For each pair of datasets, compute average similarity difference
    inter_similarities = []
    for i, dataset1 in enumerate(datasets):
        for dataset2 in datasets[i+1:]:
            sim1 = np.mean(dataset_results[dataset1])
            sim2 = np.mean(dataset_results[dataset2])
            diff = abs(sim1 - sim2)
            inter_similarities.append(diff)
            metrics['inter_dataset_similarity'][f"{dataset1}_vs_{dataset2}"] = {
                'dataset1_mean': float(sim1),
                'dataset2_mean': float(sim2),
                'difference': float(diff)
            }
    
    # Separation score: ratio of inter-dataset difference to intra-dataset variance
    if inter_similarities and metrics['intra_dataset_similarity']:
        mean_inter_diff = np.mean(inter_similarities)
        mean_intra_std = np.mean([v['std'] for v in metrics['intra_dataset_similarity'].values()])
        separation_score = mean_inter_diff / (mean_intra_std + 1e-8)
        metrics['separation_score'] = {
            'score': float(separation_score),
            'mean_inter_dataset_diff': float(mean_inter_diff),
            'mean_intra_dataset_std': float(mean_intra_std)
        }
    
    return metrics


def evaluate_top_k_accuracy(results, database, k_values=[1, 3, 5]):
    """
    Evaluate top-k agreement: Do samples from the same test dataset agree on their top-k matches?
    
    This measures consistency - samples from the same test dataset should have similar
    top-k matches to training datasets.
    """
    # Group by test dataset
    dataset_results = defaultdict(list)
    for result in results:
        if 'error' in result:
            continue
        dataset = result['dataset']
        top_similar = result.get('top_similar_datasets', [])
        dataset_results[dataset].append(top_similar)
    
    metrics = {}
    
    # For each test dataset, check if samples agree on top matches
    for k in k_values:
        agreement_scores = []
        per_dataset_agreements = {}
        
        for dataset, top_matches_list in dataset_results.items():
            if len(top_matches_list) < 2:
                continue
            
            # Get top-k matches for each sample
            top_k_matches = []
            for top_matches in top_matches_list:
                top_k = [m['dataset'] for m in top_matches[:k]]
                top_k_matches.append(set(top_k))
            
            # Compute pairwise agreement (Jaccard similarity)
            agreements = []
            for i in range(len(top_k_matches)):
                for j in range(i+1, len(top_k_matches)):
                    intersection = len(top_k_matches[i] & top_k_matches[j])
                    union = len(top_k_matches[i] | top_k_matches[j])
                    if union > 0:
                        jaccard = intersection / union
                        agreements.append(jaccard)
            
            if agreements:
                mean_agreement = np.mean(agreements)
                agreement_scores.append(mean_agreement)
                per_dataset_agreements[dataset] = {
                    'mean': float(mean_agreement),
                    'std': float(np.std(agreements)),
                    'num_samples': len(top_matches_list)
                }
        
        metrics[f'top_{k}_agreement'] = {
            'overall_mean': float(np.mean(agreement_scores)) if agreement_scores else 0.0,
            'overall_std': float(np.std(agreement_scores)) if agreement_scores else 0.0,
            'per_dataset': per_dataset_agreements
        }
    
    return metrics


def evaluate_confidence_distribution(results):
    """Evaluate the distribution of confidence scores (similarity scores)."""
    similarities = []
    ood_flags = []
    
    for result in results:
        if 'error' in result:
            continue
        best_match = result.get('best_match', {})
        similarity = best_match.get('similarity', 0.0)
        similarities.append(similarity)
        ood_flags.append(result.get('is_out_of_domain', False))
    
    similarities = np.array(similarities)
    ood_flags = np.array(ood_flags)
    
    metrics = {
        'overall': {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities))
        },
        'in_domain': {
            'mean': float(np.mean(similarities[~ood_flags])) if np.any(~ood_flags) else None,
            'std': float(np.std(similarities[~ood_flags])) if np.any(~ood_flags) else None,
            'count': int(np.sum(~ood_flags))
        },
        'out_of_domain': {
            'mean': float(np.mean(similarities[ood_flags])) if np.any(ood_flags) else None,
            'std': float(np.std(similarities[ood_flags])) if np.any(ood_flags) else None,
            'count': int(np.sum(ood_flags))
        }
    }
    
    return metrics


def create_confusion_matrix(results, output_path=None):
    """
    Create confusion matrix: test dataset vs predicted best match.
    """
    # Build confusion matrix
    test_datasets = set()
    predicted_datasets = set()
    
    for result in results:
        if 'error' in result:
            continue
        test_datasets.add(result['dataset'])
        best_match = result.get('best_match', {})
        pred = best_match.get('dataset', 'OOD')
        predicted_datasets.add(pred)
    
    test_datasets = sorted(list(test_datasets))
    predicted_datasets = sorted(list(predicted_datasets))
    
    # Add OOD if not present
    if 'OOD' not in predicted_datasets:
        predicted_datasets.append('OOD')
    
    confusion = np.zeros((len(test_datasets), len(predicted_datasets)), dtype=int)
    test_to_idx = {d: i for i, d in enumerate(test_datasets)}
    pred_to_idx = {d: i for i, d in enumerate(predicted_datasets)}
    
    for result in results:
        if 'error' in result:
            continue
        test_idx = test_to_idx[result['dataset']]
        best_match = result.get('best_match', {})
        pred = best_match.get('dataset', 'OOD')
        if pred in pred_to_idx:
            pred_idx = pred_to_idx[pred]
            confusion[test_idx, pred_idx] += 1
    
    # Create visualization
    if output_path:
        plt.figure(figsize=(max(12, len(predicted_datasets)), max(10, len(test_datasets))))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                   xticklabels=predicted_datasets, yticklabels=test_datasets,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix: Test Dataset vs Predicted Best Match')
        plt.xlabel('Predicted Dataset')
        plt.ylabel('True Test Dataset')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'matrix': confusion.tolist(),
        'test_datasets': test_datasets,
        'predicted_datasets': predicted_datasets
    }


def evaluate_metric_importance(results, top_k=10):
    """
    Analyze which metrics are most frequently identified as different.
    """
    metric_counts = defaultdict(int)
    metric_z_scores = defaultdict(list)
    
    for result in results:
        if 'error' in result or 'metric_differences' not in result:
            continue
        
        for diff in result['metric_differences']:
            metric = diff['metric']
            z_score = diff['z_score']
            metric_counts[metric] += 1
            metric_z_scores[metric].append(z_score)
    
    # Compute average z-scores
    metric_importance = []
    for metric, z_scores in metric_z_scores.items():
        metric_importance.append({
            'metric': metric,
            'frequency': metric_counts[metric],
            'mean_z_score': float(np.mean(z_scores)),
            'std_z_score': float(np.std(z_scores)),
            'max_z_score': float(np.max(z_scores))
        })
    
    # Sort by frequency and mean z-score
    metric_importance.sort(key=lambda x: (x['frequency'], x['mean_z_score']), reverse=True)
    
    return {
        'top_metrics': metric_importance[:top_k],
        'all_metrics': metric_importance
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection results")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to detection results JSON file")
    parser.add_argument("--database", type=str, required=True,
                       help="Path to database pickle file")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output path for evaluation results")
    parser.add_argument("--confusion_matrix", type=str, default=None,
                       help="Output path for confusion matrix plot (optional)")
    parser.add_argument("--top_k_values", type=int, nargs='+', default=[1, 3, 5],
                       help="K values for top-k accuracy evaluation")
    parser.add_argument("--domains_csv", type=str, 
                       default="/work/nvme/bbjs/shi3/evaluation/espnet/arecho_app/source_data/domains.csv",
                       help="Path to domains.csv file")
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    print(f"Loaded {len(results)} results")
    
    print(f"Loading database from {args.database}...")
    database = load_database(args.database)
    
    print("\nEvaluating classification consistency...")
    consistency_metrics = evaluate_classification_consistency(results)
    
    print("Evaluating dataset separation...")
    separation_metrics = evaluate_dataset_separation(results)
    
    print("Evaluating top-k accuracy...")
    top_k_metrics = evaluate_top_k_accuracy(results, database, k_values=args.top_k_values)
    
    print("Evaluating confidence distribution...")
    confidence_metrics = evaluate_confidence_distribution(results)
    
    print("Creating confusion matrix...")
    confusion_matrix = create_confusion_matrix(results, output_path=args.confusion_matrix)
    
    print("Analyzing metric importance...")
    metric_importance = evaluate_metric_importance(results)
    
    print("Evaluating OOD estimation against domains.csv...")
    ood_evaluation = evaluate_ood_estimation(results, database, args.domains_csv)
    
    print("Finding closest mappings at dataset, domain, and language levels...")
    closest_mappings = find_closest_mappings(results, database, args.domains_csv)
    
    # Compile all metrics
    evaluation_results = {
        'summary': {
            'total_samples': len(results),
            'num_test_datasets': len(set(r['dataset'] for r in results if 'error' not in r)),
            'num_ood_samples': sum(1 for r in results if r.get('is_out_of_domain', False)),
            'num_in_domain_samples': sum(1 for r in results if not r.get('is_out_of_domain', False))
        },
        'classification_consistency': consistency_metrics,
        'dataset_separation': separation_metrics,
        'top_k_accuracy': top_k_metrics,
        'confidence_distribution': confidence_metrics,
        'confusion_matrix': confusion_matrix,
        'metric_importance': metric_importance,
        'ood_evaluation': ood_evaluation,
        'closest_mappings': closest_mappings
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEvaluation results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples: {evaluation_results['summary']['total_samples']}")
    print(f"Test datasets: {evaluation_results['summary']['num_test_datasets']}")
    print(f"OOD samples: {evaluation_results['summary']['num_ood_samples']}")
    print(f"In-domain samples: {evaluation_results['summary']['num_in_domain_samples']}")
    
    if 'overall_consistency' in consistency_metrics:
        print(f"\nClassification Consistency:")
        print(f"  Mean: {consistency_metrics['overall_consistency']['mean_consistency']:.3f}")
        print(f"  Std: {consistency_metrics['overall_consistency']['std_consistency']:.3f}")
    
    if 'separation_score' in separation_metrics:
        print(f"\nDataset Separation Score: {separation_metrics['separation_score']['score']:.3f}")
    
    print(f"\nTop-K Agreement:")
    for k in args.top_k_values:
        key = f'top_{k}_agreement'
        if key in top_k_metrics:
            print(f"  Top-{k}: {top_k_metrics[key]['overall_mean']:.3f} ± {top_k_metrics[key]['overall_std']:.3f}")
    
    print(f"\nConfidence Distribution:")
    print(f"  Overall mean: {confidence_metrics['overall']['mean']:.3f}")
    if confidence_metrics['in_domain']['mean'] is not None:
        print(f"  In-domain mean: {confidence_metrics['in_domain']['mean']:.3f}")
    if confidence_metrics['out_of_domain']['mean'] is not None:
        print(f"  Out-of-domain mean: {confidence_metrics['out_of_domain']['mean']:.3f}")
    
    print(f"\nTop 5 Most Important Metrics:")
    for i, metric in enumerate(metric_importance['top_metrics'][:5], 1):
        print(f"  {i}. {metric['metric']}: freq={metric['frequency']}, mean_z={metric['mean_z_score']:.2f}")
    
    # Print OOD evaluation summary
    if 'overall' in ood_evaluation and ood_evaluation['overall']:
        print(f"\n" + "="*60)
        print("OOD ESTIMATION EVALUATION")
        print("="*60)
        overall = ood_evaluation['overall']
        print(f"Accuracy: {overall['accuracy']:.3f}")
        print(f"Precision: {overall['precision']:.3f}")
        print(f"Recall: {overall['recall']:.3f}")
        print(f"F1 Score: {overall['f1_score']:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (OOD correctly identified): {overall['true_positives']}")
        print(f"  False Positives (In-domain incorrectly identified as OOD): {overall['false_positives']}")
        print(f"  True Negatives (In-domain correctly identified): {overall['true_negatives']}")
        print(f"  False Negatives (OOD incorrectly identified as In-domain): {overall['false_negatives']}")
        
        print(f"\nPer-Dataset OOD Evaluation:")
        for dataset, info in sorted(ood_evaluation['per_dataset'].items()):
            status = "✓" if info['correct'] else "✗"
            print(f"  {status} {dataset}:")
            print(f"    True OOD: {info['true_is_ood']}, Predicted OOD: {info['predicted_is_ood']}")
            print(f"    Domain: {info['domain']}, Language: {info['language']}")
            print(f"    OOD ratio: {info['ood_ratio']:.2f} ({info['num_ood_samples']}/{info['num_samples']})")
    
    # Print closest mappings summary
    print(f"\n" + "="*60)
    print("CLOSEST MAPPINGS")
    print("="*60)
    
    print(f"\nDataset-Level Mappings:")
    for dataset, info in sorted(closest_mappings['dataset_level'].items()):
        print(f"  {dataset} -> {info['best_match_dataset']} (similarity: {info['similarity']:.3f})")
    
    print(f"\nDomain-Level Mappings:")
    for dataset, info in sorted(closest_mappings['domain_level'].items()):
        match_indicator = "✓" if info['domain_match'] else "✗"
        print(f"  {match_indicator} {dataset}:")
        print(f"    Test domain: {info['test_domain']}, Best match: {info['best_match_domain']} (similarity: {info['similarity']:.3f})")
    
    print(f"\nLanguage-Level Mappings:")
    for dataset, info in sorted(closest_mappings['language_level'].items()):
        match_indicator = "✓" if info['language_match'] else "✗"
        print(f"  {match_indicator} {dataset}:")
        print(f"    Test language: {info['test_language']}, Best match: {info['best_match_language']} (similarity: {info['similarity']:.3f})")


if __name__ == "__main__":
    main()

