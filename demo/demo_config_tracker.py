"""
Demo script for ConfigTracker functionality.

Shows how to use ConfigTracker to save experiment configuration and results.
"""

import sys
import os
import tempfile
import json

# Add spectral_gpt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spectral_gpt'))

import torch.nn as nn
from monitoring import ConfigTracker


def demo_config_tracker():
    """Demonstrate ConfigTracker functionality"""
    print("=" * 60)
    print("ConfigTracker Demo")
    print("=" * 60)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        experiment_dir = os.path.join(tmpdir, "demo_experiment")
        
        # Initialize ConfigTracker
        tracker = ConfigTracker(experiment_dir=experiment_dir)
        print(f"\n✓ Created ConfigTracker for: {experiment_dir}")
        
        # 1. Save configuration
        print("\n1. Saving experiment configuration...")
        
        config = {
            'model': {
                'd_model': 256,
                'num_layers': 6,
                'num_heads': 8,
                'num_waves': 4,
                'num_harmonics': 3,
                'vocab_size': 5000,
                'block_size': 128,
                'dropout': 0.1
            },
            'training': {
                'optimizer': 'RGD',
                'lr': 0.001,
                'weight_decay': 0.01,
                'batch_size': 32,
                'steps': 10000,
                'warmup_steps': 1000,
                'use_rgd': True,
                'use_qfe': True,
                'rgd_strength': 0.5,
                'qfe_lambda': 0.1
            },
            'dataset': {
                'name': 'TinyShakespeare',
                'num_tokens': 1000000,
                'train_split': 0.9,
                'val_split': 0.1
            }
        }
        
        # Create a simple model for demo
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        dataset_info = {
            'name': 'TinyShakespeare',
            'num_tokens': 1000000,
            'vocab_size': 5000
        }
        
        tracker.save_config(config, model=model, dataset_info=dataset_info)
        
        # 2. Load and display configuration
        print("\n2. Loading configuration...")
        loaded_config = tracker.load_config()
        
        print(f"\n   Experiment ID: {loaded_config['experiment_id']}")
        print(f"   Timestamp: {loaded_config['timestamp']}")
        print(f"   Git Hash: {loaded_config['git_hash']}")
        print("\n   Hardware Info:")
        for key, value in loaded_config['hardware'].items():
            print(f"     - {key}: {value}")
        
        print("\n   Model Info:")
        for key, value in loaded_config['model'].items():
            print(f"     - {key}: {value}")
        
        print("\n   Training Config:")
        for key, value in loaded_config['config']['training'].items():
            print(f"     - {key}: {value}")
        
        # 3. Save results
        print("\n3. Saving experiment results...")
        
        final_metrics = {
            'val_loss': 0.523,
            'perplexity': 1.687,
            'best_val_loss': 0.498,
            'best_step': 8500,
            'total_time_seconds': 3600,
            'tokens_per_second': 2500
        }
        
        generation_samples = [
            "To be or not to be, that is the question",
            "All the world's a stage, and all the men and women merely players",
            "What's in a name? That which we call a rose by any other name would smell as sweet"
        ]
        
        tracker.save_results(
            final_metrics=final_metrics,
            best_checkpoint='checkpoint_step_8500.pt',
            generation_samples=generation_samples
        )
        
        # 4. Display results file
        print("\n4. Results file contents:")
        with open(tracker.results_file, 'r') as f:
            results = json.load(f)
        
        print("\n   Final Metrics:")
        for key, value in results['final_metrics'].items():
            print(f"     - {key}: {value}")
        
        print(f"\n   Best Checkpoint: {results['best_checkpoint']}")
        print("\n   Generation Samples:")
        for i, sample in enumerate(results['generation_samples'], 1):
            print(f"     {i}. {sample}")
        
        # 5. Show file structure
        print("\n5. Experiment directory structure:")
        for root, dirs, files in os.walk(experiment_dir):
            level = root.replace(experiment_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)


if __name__ == "__main__":
    demo_config_tracker()
