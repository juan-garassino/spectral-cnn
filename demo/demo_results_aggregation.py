"""
Demo script for ResultsParser and ResultsAggregator

Shows how to:
1. Parse experiment results from text file
2. Aggregate multiple experiments
3. Generate comparison tables
4. Perform ablation analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spectral_gpt.monitoring import ResultsParser, ResultsAggregator
import json


def main():
    """Demonstrate results aggregation."""
    print("=" * 80)
    print("Results Aggregation Demo")
    print("=" * 80)
    print()
    
    # Parse results from text file
    results_file = 'spectral_gpt/experiment_results/results.txt'
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return
    
    print(f"📥 Parsing results from: {results_file}")
    parser = ResultsParser(results_file)
    experiments = parser.parse()
    
    print(f"✅ Parsed {len(experiments)} experiments")
    print()
    
    # Show experiment names
    print("📋 Experiments found:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
    print()
    
    # Save parsed results to JSON
    json_output = 'spectral_gpt/experiment_results/parsed_results.json'
    parser.save_to_json(json_output)
    print(f"💾 Saved parsed results to: {json_output}")
    print()
    
    # Create aggregator
    print("=" * 80)
    print("Results Aggregation")
    print("=" * 80)
    print()
    
    aggregator = ResultsAggregator()
    aggregator.load_from_parser(parser)
    
    # Generate comparison table
    print("📊 Comparison Table:")
    print()
    print(aggregator.generate_comparison_table())
    print()
    
    # Generate summary statistics
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print()
    
    stats = aggregator.generate_summary_statistics()
    print(json.dumps(stats, indent=2))
    print()
    
    # Generate ablation analysis
    print("=" * 80)
    print("Ablation Analysis")
    print("=" * 80)
    print()
    
    ablation = aggregator.generate_ablation_analysis()
    print(json.dumps(ablation, indent=2))
    print()
    
    # Compare specific experiments
    if len(experiments) >= 2:
        print("=" * 80)
        print("Experiment Comparison")
        print("=" * 80)
        print()
        
        # Compare first two experiments
        exp1_name = experiments[0]['name'].split()[0]
        exp2_name = experiments[1]['name'].split()[0]
        
        comparison = aggregator.compare_experiments(exp1_name, exp2_name)
        print(f"Comparing: {comparison.get('experiment_1')} vs {comparison.get('experiment_2')}")
        print()
        print(json.dumps(comparison, indent=2))
        print()
    
    # Save comprehensive summary
    summary_output = 'spectral_gpt/experiment_results/aggregated_summary.json'
    aggregator.save_summary(summary_output)
    print(f"💾 Saved aggregated summary to: {summary_output}")
    print()
    
    print("✅ Demo complete!")


if __name__ == '__main__':
    main()
