#!/usr/bin/env python3
"""
Analysis and visualization utilities for audio analysis results.

This script provides tools to:
1. Load and analyze saved results
2. Create visualizations
3. Generate detailed reports
4. Compare different analysis runs
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Visualization disabled.")


def load_results(results_path: str) -> dict:
    """Load analysis results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_results(results: dict) -> dict:
    """Perform detailed analysis on results."""
    results_list = results.get('results', [])
    
    analysis = {
        'summary': {
            'total_files': len(results_list),
            'successful': sum(1 for r in results_list if 'error' not in r),
            'failed': sum(1 for r in results_list if 'error' in r),
            'ood_count': sum(1 for r in results_list if r.get('is_out_of_domain', False)),
            'in_domain_count': sum(1 for r in results_list if not r.get('is_out_of_domain', False))
        },
        'best_matches': defaultdict(int),
        'similarity_distribution': [],
        'top_datasets': defaultdict(list),
        'per_file_details': []
    }
    
    for result in results_list:
        if 'error' in result:
            continue
        
        # Best matches
        best_match = result.get('best_match', {})
        if best_match and best_match.get('dataset'):
            analysis['best_matches'][best_match['dataset']] += 1
        
        # Similarity scores
        if best_match and best_match.get('similarity') is not None:
            analysis['similarity_distribution'].append(best_match['similarity'])
        
        # Top datasets
        top_similar = result.get('top_similar_datasets', [])
        for match in top_similar[:5]:
            analysis['top_datasets'][match['dataset']].append(match['similarity'])
        
        # Per-file details
        analysis['per_file_details'].append({
            'file': result.get('audio_name', 'Unknown'),
            'is_ood': result.get('is_out_of_domain', False),
            'best_match': best_match.get('dataset') if best_match else None,
            'similarity': best_match.get('similarity') if best_match else None,
            'top_3_datasets': [m['dataset'] for m in top_similar[:3]]
        })
    
    # Calculate statistics
    if analysis['similarity_distribution']:
        sim_array = np.array(analysis['similarity_distribution'])
        analysis['similarity_stats'] = {
            'mean': float(np.mean(sim_array)),
            'std': float(np.std(sim_array)),
            'min': float(np.min(sim_array)),
            'max': float(np.max(sim_array)),
            'median': float(np.median(sim_array))
        }
    
    # Average similarities per dataset
    analysis['dataset_avg_similarities'] = {
        dataset: float(np.mean(similarities))
        for dataset, similarities in analysis['top_datasets'].items()
    }
    
    return analysis


def print_analysis_report(analysis: dict):
    """Print a formatted analysis report."""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS REPORT")
    print("="*80)
    
    summary = analysis['summary']
    print(f"\nSummary Statistics:")
    print(f"  Total files: {summary['total_files']}")
    print(f"  Successfully analyzed: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Out-of-domain: {summary['ood_count']} ({summary['ood_count']/summary['successful']*100:.1f}%)" if summary['successful'] > 0 else "N/A")
    print(f"  In-domain: {summary['in_domain_count']} ({summary['in_domain_count']/summary['successful']*100:.1f}%)" if summary['successful'] > 0 else "N/A")
    
    if 'similarity_stats' in analysis:
        stats = analysis['similarity_stats']
        print(f"\nSimilarity Score Statistics:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
    
    if analysis['best_matches']:
        print(f"\nMost Common Best Matches:")
        for dataset, count in sorted(analysis['best_matches'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {dataset:30s}: {count:3d} files ({count/summary['successful']*100:.1f}%)")
    
    if analysis['dataset_avg_similarities']:
        print(f"\nTop Datasets by Average Similarity:")
        for dataset, avg_sim in sorted(analysis['dataset_avg_similarities'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {dataset:30s}: {avg_sim:.4f}")
    
    print(f"\n" + "-"*80)
    print("Per-File Details (first 10):")
    print("-"*80)
    for detail in analysis['per_file_details'][:10]:
        status = "OOD" if detail['is_ood'] else "IN"
        print(f"  {detail['file']:30s} [{status}] -> {detail['best_match']} ({detail['similarity']:.3f})")


def create_visualizations(analysis: dict, output_dir: str = "analysis_plots"):
    """Create visualization plots."""
    if not HAS_PLOTTING:
        print("Visualization libraries not available. Skipping plots.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = analysis['summary']
    
    # 1. Similarity distribution histogram
    if analysis['similarity_distribution']:
        plt.figure(figsize=(10, 6))
        plt.hist(analysis['similarity_distribution'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarity Scores')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_distribution.png', dpi=300)
        plt.close()
        print(f"Saved: {output_dir / 'similarity_distribution.png'}")
    
    # 2. Best matches bar chart
    if analysis['best_matches']:
        top_matches = sorted(analysis['best_matches'].items(), key=lambda x: x[1], reverse=True)[:15]
        datasets = [m[0] for m in top_matches]
        counts = [m[1] for m in top_matches]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(datasets)), counts)
        plt.yticks(range(len(datasets)), datasets)
        plt.xlabel('Number of Files')
        plt.title('Most Common Best Matches')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'best_matches.png', dpi=300)
        plt.close()
        print(f"Saved: {output_dir / 'best_matches.png'}")
    
    # 3. OOD vs In-domain pie chart
    if summary['successful'] > 0:
        plt.figure(figsize=(8, 8))
        labels = ['In-Domain', 'Out-of-Domain']
        sizes = [summary['in_domain_count'], summary['ood_count']]
        colors = ['#66b3ff', '#ff9999']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Out-of-Domain vs In-Domain Distribution')
        plt.tight_layout()
        plt.savefig(output_dir / 'ood_distribution.png', dpi=300)
        plt.close()
        print(f"Saved: {output_dir / 'ood_distribution.png'}")
    
    # 4. Top datasets similarity heatmap
    if analysis['dataset_avg_similarities']:
        top_datasets = sorted(analysis['dataset_avg_similarities'].items(), key=lambda x: x[1], reverse=True)[:20]
        datasets = [d[0] for d in top_datasets]
        similarities = [d[1] for d in top_datasets]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(datasets)), similarities)
        plt.yticks(range(len(datasets)), datasets)
        plt.xlabel('Average Similarity Score')
        plt.title('Top Datasets by Average Similarity')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'top_datasets_similarity.png', dpi=300)
        plt.close()
        print(f"Saved: {output_dir / 'top_datasets_similarity.png'}")


def compare_results(results_files: list, output_path: str = None):
    """Compare multiple result files."""
    print("\n" + "="*80)
    print("COMPARING MULTIPLE RESULTS")
    print("="*80)
    
    all_analyses = {}
    for results_file in results_files:
        print(f"\nLoading: {results_file}")
        results = load_results(results_file)
        analysis = analyze_results(results)
        all_analyses[Path(results_file).stem] = analysis
    
    # Comparison table
    print("\n" + "-"*80)
    print("Comparison Summary:")
    print("-"*80)
    print(f"{'Run':<30s} {'Total':<8s} {'Success':<8s} {'OOD':<8s} {'OOD%':<8s} {'Mean Sim':<10s}")
    print("-"*80)
    
    for name, analysis in all_analyses.items():
        summary = analysis['summary']
        mean_sim = analysis.get('similarity_stats', {}).get('mean', 0.0)
        ood_pct = summary['ood_count'] / summary['successful'] * 100 if summary['successful'] > 0 else 0
        
        print(f"{name:<30s} {summary['total_files']:<8d} {summary['successful']:<8d} "
              f"{summary['ood_count']:<8d} {ood_pct:<7.1f}% {mean_sim:<10.4f}")
    
    if output_path:
        comparison = {
            'runs': {name: analysis for name, analysis in all_analyses.items()}
        }
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize audio analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("results_file", type=str,
                       help="Path to results JSON file")
    parser.add_argument("--output_dir", type=str, default="analysis_plots",
                       help="Output directory for plots (default: analysis_plots)")
    parser.add_argument("--report", type=str, default=None,
                       help="Save detailed report to JSON file")
    parser.add_argument("--compare", type=str, nargs='+',
                       help="Compare multiple result files")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.compare, args.report)
        return
    
    # Load and analyze results
    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)
    
    print("Analyzing results...")
    analysis = analyze_results(results)
    
    # Print report
    print_analysis_report(analysis)
    
    # Create visualizations
    if not args.no_plots:
        print("\nCreating visualizations...")
        create_visualizations(analysis, args.output_dir)
    
    # Save report
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nDetailed report saved to: {args.report}")


if __name__ == "__main__":
    main()


