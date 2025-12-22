#!/usr/bin/env python3
"""
Demonstration script showing how to analyze real detection experiment results.

This script follows the use case exemplified in run_detecting_experiments.sh:
1. Runs detection experiments with different configurations
2. Loads and analyzes real results
3. Demonstrates the analysis workflow with actual data
"""

import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from interactive_audio_analyzer import print_analysis_summary
from analyze_results import analyze_results, print_analysis_report, create_visualizations


def find_existing_results(results_dir="experiment_results"):
    """Find existing detection result files."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    
    result_files = list(results_dir.glob("detection_*.json"))
    return sorted(result_files)


def run_detection_experiment(config):
    """
    Run a single detection experiment following the pattern from run_detecting_experiments.sh.
    
    Args:
        config: dict with experiment configuration
    
    Returns:
        Path to result file or None if failed
    """
    database_path = config.get('database', 'database.pkl')
    feature_dir = config.get('feature_dir', 'feature')
    results_dir = config.get('results_dir', 'experiment_results')
    detection_mode = config['detection_mode']
    
    # Build experiment name
    exp_name_parts = [detection_mode]
    if detection_mode == 'metric':
        exp_name_parts.append(config.get('metric_mode', 'numerical_only'))
    elif detection_mode == 'fewshot':
        exp_name_parts.append(f"k{config.get('k_shot', 5)}")
        exp_name_parts.append(config.get('fewshot_aggregation', 'mean'))
    elif detection_mode == 'raw_features':
        exp_name_parts.append(config.get('raw_features_aggregation', 'mean'))
    
    exp_name_parts.append(config.get('similarity_metric', 'cosine'))
    exp_name_parts.append(f"th{config.get('ood_threshold', 0.5)}")
    
    exp_name = "_".join(exp_name_parts)
    results_file = Path(results_dir) / f"detection_{exp_name}.json"
    
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "functions/detecting.py",
        "--database", database_path,
        "--feature_dir", feature_dir,
        "--output", str(results_file),
        "--detection_mode", detection_mode,
        "--similarity_metric", config.get('similarity_metric', 'cosine'),
        "--ood_threshold", str(config.get('ood_threshold', 0.5)),
    ]
    
    if detection_mode == 'metric':
        cmd.extend(["--metric_mode", config.get('metric_mode', 'numerical_only')])
        cmd.extend(["--top_k_metrics", str(config.get('top_k_metrics', 10))])
    elif detection_mode == 'fewshot':
        cmd.extend(["--k_shot", str(config.get('k_shot', 5))])
        cmd.extend(["--fewshot_aggregation", config.get('fewshot_aggregation', 'mean')])
        cmd.extend(["--raw_features_aggregation", config.get('raw_features_aggregation', 'mean')])
    elif detection_mode == 'raw_features':
        cmd.extend(["--raw_features_aggregation", config.get('raw_features_aggregation', 'mean')])
    
    print(f"Running experiment: {exp_name}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if results_file.exists():
            print(f"  ✓ Results saved to {results_file}")
            return results_file
        else:
            print(f"  ✗ Experiment failed: output file not created")
            return None
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Experiment failed: {e}")
        print(f"  Error output: {e.stderr}")
        return None


def load_and_analyze_result(result_file):
    """Load a detection result file and analyze it."""
    print(f"\nLoading result file: {result_file}")
    
    try:
        with open(result_file, 'r') as f:
            results_data = json.load(f)
        
        # Convert to format expected by analysis functions
        # The detecting.py output is a list of results directly
        if isinstance(results_data, list):
            results_list = results_data
        elif isinstance(results_data, dict) and 'results' in results_data:
            results_list = results_data['results']
        else:
            results_list = [results_data]
        
        return {
            'results': results_list,
            'metadata': {
                'total_files': len(results_list),
                'successful': sum(1 for r in results_list if 'error' not in r),
                'ood_count': sum(1 for r in results_list if r.get('is_out_of_domain', False))
            }
        }
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        return None


def demonstrate_analysis():
    """Demonstrate the analysis workflow using real detection experiments."""
    
    print("="*80)
    print("DEMONSTRATION: Real Detection Experiment Analysis")
    print("="*80)
    print("\nThis demo follows the workflow from run_detecting_experiments.sh")
    print("="*80)
    
    # Configuration (matching run_detecting_experiments.sh)
    database_path = "database.pkl"
    feature_dir = "feature"
    results_dir = "experiment_results"
    
    # Check if database exists
    if not Path(database_path).exists():
        print(f"\nERROR: Database not found: {database_path}")
        print("Please build the database first:")
        print("  python functions/database_construct.py --feature_dir feature --output database.pkl")
        return
    
    # Step 1: Check for existing results
    print("\n" + "="*80)
    print("Step 1: Checking for existing detection results")
    print("="*80)
    
    existing_results = find_existing_results(results_dir)
    
    if existing_results:
        print(f"\nFound {len(existing_results)} existing result files")
        print("Using existing results for demonstration")
        
        # Use a representative set of results
        selected_results = []
        for pattern in [
            "detection_feature_cosine_th0.5.json",
            "detection_metric_numerical_only_cosine_th0.5.json",
            "detection_raw_features_mean_cosine_th0.5.json"
        ]:
            for result_file in existing_results:
                if result_file.name == pattern:
                    selected_results.append(result_file)
                    break
        
        # If patterns not found, use first few
        if not selected_results:
            selected_results = existing_results[:3]
    else:
        print("\nNo existing results found. Running sample experiments...")
        
        # Step 2: Run sample detection experiments
        print("\n" + "="*80)
        print("Step 2: Running sample detection experiments")
        print("="*80)
        print("\nFollowing the pattern from run_detecting_experiments.sh")
        
        # Define sample experiments (matching the script patterns)
        experiments = [
            {
                'database': database_path,
                'feature_dir': feature_dir,
                'results_dir': results_dir,
                'detection_mode': 'feature',
                'similarity_metric': 'cosine',
                'ood_threshold': 0.5
            },
            {
                'database': database_path,
                'feature_dir': feature_dir,
                'results_dir': results_dir,
                'detection_mode': 'metric',
                'metric_mode': 'numerical_only',
                'similarity_metric': 'cosine',
                'ood_threshold': 0.5,
                'top_k_metrics': 10
            },
            {
                'database': database_path,
                'feature_dir': feature_dir,
                'results_dir': results_dir,
                'detection_mode': 'raw_features',
                'raw_features_aggregation': 'mean',
                'similarity_metric': 'cosine',
                'ood_threshold': 0.5
            }
        ]
        
        selected_results = []
        for exp_config in experiments:
            result_file = run_detection_experiment(exp_config)
            if result_file:
                selected_results.append(result_file)
        
        if not selected_results:
            print("\nERROR: No experiments completed successfully")
            print("Please check:")
            print("  1. Database exists: database.pkl")
            print("  2. Feature directory exists: feature/")
            print("  3. Test datasets have features extracted")
            return
    
    # Step 3: Analyze results
    print("\n" + "="*80)
    print("Step 3: Analyzing detection results")
    print("="*80)
    
    all_analyses = {}
    
    for result_file in selected_results:
        print(f"\nAnalyzing: {result_file.name}")
        results_data = load_and_analyze_result(result_file)
        
        if results_data is None:
            continue
        
        # Print summary
        print(f"\n  Summary for {result_file.stem}:")
        print_analysis_summary(results_data['results'])
        
        # Detailed analysis
        analysis = analyze_results(results_data)
        all_analyses[result_file.stem] = {
            'results_data': results_data,
            'analysis': analysis,
            'file': result_file
        }
    
    # Step 4: Compare different detection modes
    print("\n" + "="*80)
    print("Step 4: Comparing Different Detection Modes")
    print("="*80)
    
    if len(all_analyses) > 1:
        print("\nComparison across detection modes:")
        print(f"{'Mode':<40s} {'Total':<8s} {'Success':<8s} {'OOD':<8s} {'OOD%':<8s} {'Mean Sim':<10s}")
        print("-"*80)
        
        for name, data in all_analyses.items():
            analysis = data['analysis']
            summary = analysis['summary']
            mean_sim = analysis.get('similarity_stats', {}).get('mean', 0.0)
            ood_pct = summary['ood_count'] / summary['successful'] * 100 if summary['successful'] > 0 else 0
            
            # Extract mode from filename
            mode_name = name.replace('detection_', '').replace('_th0.5', '')
            
            print(f"{mode_name:<40s} {summary['total_files']:<8d} {summary['successful']:<8d} "
                  f"{summary['ood_count']:<8d} {ood_pct:<7.1f}% {mean_sim:<10.4f}")
    
    # Step 5: Detailed analysis of one result
    print("\n" + "="*80)
    print("Step 5: Detailed Analysis (First Result)")
    print("="*80)
    
    if all_analyses:
        first_name = list(all_analyses.keys())[0]
        first_data = all_analyses[first_name]
        
        print(f"\nDetailed report for: {first_name}")
        print_analysis_report(first_data['analysis'])
    
    # Step 6: Create visualizations
    print("\n" + "="*80)
    print("Step 6: Creating Visualizations")
    print("="*80)
    
    if all_analyses:
        first_data = list(all_analyses.values())[0]
        analysis = first_data['analysis']
        
        try:
            output_dir = "demo_analysis_plots"
            create_visualizations(analysis, output_dir=output_dir)
            print(f"\nVisualizations saved to: {output_dir}/")
        except Exception as e:
            print(f"\nVisualization skipped: {e}")
    
    # Step 7: Save analysis results
    print("\n" + "="*80)
    print("Step 7: Saving Analysis Results")
    print("="*80)
    
    # Save combined analysis
    combined_analysis = {
        'experiments': {
            name: {
                'file': str(data['file']),
                'summary': data['analysis']['summary'],
                'similarity_stats': data['analysis'].get('similarity_stats', {}),
                'best_matches': dict(data['analysis']['best_matches']),
            }
            for name, data in all_analyses.items()
        },
        'comparison': {
            'total_experiments': len(all_analyses),
            'modes_tested': [name.replace('detection_', '').replace('_th0.5', '') 
                           for name in all_analyses.keys()]
        }
    }
    
    output_file = "demo_combined_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(combined_analysis, f, indent=2)
    print(f"\nCombined analysis saved to: {output_file}")
    
    # Step 8: Insights and recommendations
    print("\n" + "="*80)
    print("Step 8: Insights and Recommendations")
    print("="*80)
    
    if all_analyses:
        print("\n1. Detection Mode Comparison:")
        print("   - Different detection modes may yield different OOD rates")
        print("   - Feature-based: Fast, good for general similarity")
        print("   - Metric-based: Detailed, shows which metrics differ")
        print("   - Raw features: Most accurate, slower")
        
        print("\n2. Out-of-Domain Detection:")
        total_ood = sum(data['analysis']['summary']['ood_count'] 
                       for data in all_analyses.values())
        total_samples = sum(data['analysis']['summary']['successful'] 
                           for data in all_analyses.values())
        if total_samples > 0:
            avg_ood_pct = (total_ood / len(all_analyses)) / (total_samples / len(all_analyses)) * 100
            print(f"   - Average OOD rate across modes: {avg_ood_pct:.1f}%")
            print("   - Lower threshold = more OOD, higher threshold = less OOD")
        
        print("\n3. Best Matches:")
        # Aggregate best matches across all experiments
        all_best_matches = defaultdict(int)
        for data in all_analyses.values():
            for dataset, count in data['analysis']['best_matches'].items():
                all_best_matches[dataset] += count
        
        if all_best_matches:
            print("   Most common best matches across all experiments:")
            for dataset, count in sorted(all_best_matches.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"     - {dataset}: {count} files")
        
        print("\n4. Recommendations:")
        print("   - Use multiple detection modes for robust analysis")
        print("   - Compare results across different similarity metrics")
        print("   - Adjust OOD threshold based on your use case")
        print("   - Check metric differences for metric-based mode")
        print("   - Consider domain adaptation for high OOD rates")
    
    print("\n" + "="*80)
    print("To analyze your own results:")
    print("  python analyze_results.py experiment_results/detection_*.json")
    print("="*80)


if __name__ == "__main__":
    demonstrate_analysis()
