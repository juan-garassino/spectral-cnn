"""
Demonstration of experiment monitoring infrastructure.

Shows how to use CheckpointManager in a training loop.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from monitoring import (
    generate_experiment_id,
    create_experiment_directory,
    CheckpointManager,
    MetricsLogger
)


def demo_checkpoint_manager():
    """Demonstrate CheckpointManager usage"""
    print("=" * 60)
    print("CheckpointManager Demo")
    print("=" * 60)
    
    # 1. Generate experiment ID
    exp_id = generate_experiment_id("demo_experiment")
    print(f"\n1. Generated experiment ID: {exp_id}")
    
    # 2. Create directory structure
    base_dir = "experiments"
    dirs = create_experiment_directory(base_dir, exp_id)
    print(f"\n2. Created directory structure:")
    for key, path in dirs.items():
        print(f"   {key}: {path}")
    
    # 3. Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n3. Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 4. Initialize CheckpointManager
    manager = CheckpointManager(
        experiment_dir=dirs['root'],
        save_interval=100,
        keep_last_n=3
    )
    print(f"\n4. Initialized CheckpointManager:")
    print(f"   Save interval: {manager.save_interval}")
    print(f"   Keep last N: {manager.keep_last_n}")
    
    # 5. Simulate training with checkpointing
    print(f"\n5. Simulating training loop...")
    loss_history = []
    config = {
        'model': 'demo_model',
        'lr': 0.001,
        'batch_size': 32
    }
    
    for step in range(1, 501):
        # Dummy training step
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Checkpoint at intervals
        if manager.should_checkpoint(step):
            checkpoint_path = manager.save_checkpoint(
                step=step,
                model=model,
                optimizer=optimizer,
                loss_history=loss_history,
                config=config
            )
            print(f"   Step {step}: Saved checkpoint to {os.path.basename(checkpoint_path)}")
    
    # 6. List all checkpoints
    print(f"\n6. Available checkpoints:")
    checkpoints = manager.list_checkpoints()
    for cp in checkpoints:
        print(f"   Step {cp['step']}: {cp['size_mb']:.2f} MB (modified: {cp['modified']})")
    
    # 7. Load latest checkpoint
    print(f"\n7. Loading latest checkpoint...")
    checkpoint = manager.load_latest_checkpoint()
    if checkpoint:
        print(f"   Loaded checkpoint from step {checkpoint['step']}")
        print(f"   Loss history length: {len(checkpoint['loss_history'])}")
        print(f"   Config: {checkpoint['config']}")
        print(f"   Timestamp: {checkpoint['timestamp']}")
        print(f"   Git hash: {checkpoint['git_hash']}")
    
    # 8. Demonstrate resumption
    print(f"\n8. Demonstrating resumption from checkpoint...")
    new_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
    
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1
    
    print(f"   Resumed training from step {start_step}")
    print(f"   Model state restored successfully")
    
    # 9. Cleanup
    print(f"\n9. Cleaning up demo files...")
    import shutil
    shutil.rmtree(dirs['root'])
    print(f"   Removed {dirs['root']}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def demo_interruption_recovery():
    """Demonstrate recovery from interruption"""
    print("\n" + "=" * 60)
    print("Interruption Recovery Demo")
    print("=" * 60)
    
    exp_id = generate_experiment_id("interruption_demo")
    base_dir = "experiments"
    dirs = create_experiment_directory(base_dir, exp_id)
    
    model = nn.Linear(10, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    manager = CheckpointManager(
        experiment_dir=dirs['root'],
        save_interval=50,
        keep_last_n=2
    )
    
    print(f"\n1. Starting training (will 'interrupt' at step 125)...")
    
    # First training session (interrupted)
    loss_history = []
    config = {'lr': 0.001}
    
    for step in range(1, 126):
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if manager.should_checkpoint(step):
            manager.save_checkpoint(step, model, optimizer, loss_history, config)
            print(f"   Checkpoint saved at step {step}")
    
    print(f"   Training 'interrupted' at step 125")
    
    # Simulate restart
    print(f"\n2. Restarting and loading checkpoint...")
    
    new_model = nn.Linear(10, 5)
    new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
    
    checkpoint = manager.load_latest_checkpoint()
    if checkpoint:
        new_model.load_state_dict(checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_history = checkpoint['loss_history']
        start_step = checkpoint['step'] + 1
        
        print(f"   Resumed from step {checkpoint['step']}")
        print(f"   Continuing from step {start_step}")
    
    # Continue training
    print(f"\n3. Continuing training to step 200...")
    for step in range(start_step, 201):
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        
        new_optimizer.zero_grad()
        loss = nn.functional.mse_loss(new_model(x), y)
        loss.backward()
        new_optimizer.step()
        
        loss_history.append(loss.item())
        
        if manager.should_checkpoint(step):
            manager.save_checkpoint(step, new_model, new_optimizer, loss_history, config)
            print(f"   Checkpoint saved at step {step}")
    
    print(f"\n4. Training completed successfully!")
    print(f"   Total steps: {len(loss_history)}")
    print(f"   Final loss: {loss_history[-1]:.4f}")
    
    # Cleanup
    import shutil
    shutil.rmtree(dirs['root'])
    print(f"\n5. Cleaned up demo files")
    
    print("\n" + "=" * 60)
    print("Interruption recovery demo complete!")
    print("=" * 60)


def demo_metrics_logger():
    """Demonstrate MetricsLogger usage"""
    print("\n" + "=" * 60)
    print("MetricsLogger Demo")
    print("=" * 60)
    
    # 1. Generate experiment ID and create directories
    exp_id = generate_experiment_id("metrics_demo")
    base_dir = "experiments"
    dirs = create_experiment_directory(base_dir, exp_id)
    print(f"\n1. Created experiment: {exp_id}")
    
    # 2. Initialize MetricsLogger
    logger = MetricsLogger(
        log_dir=dirs['logs'],
        log_interval=10
    )
    print(f"\n2. Initialized MetricsLogger:")
    print(f"   Log directory: {logger.log_dir}")
    print(f"   Log interval: {logger.log_interval}")
    print(f"   Metrics file: {logger.metrics_file.name}")
    print(f"   Training log: {logger.training_log.name}")
    
    # 3. Simulate training with metrics logging
    print(f"\n3. Simulating training with metrics logging...")
    
    for step in range(1, 101):
        # Simulate metrics that improve over time
        metrics = {
            'loss': 2.0 - step * 0.015,
            'learning_rate': 0.001 * (0.99 ** (step // 10)),
            'perplexity': 7.5 - step * 0.05,
            'tokens_per_sec': 1000 + step * 5,
            'gpu_memory_mb': 2048 + (step % 10) * 10
        }
        
        # Log at intervals
        if logger.should_log(step):
            logger.log_metrics(step, metrics)
            print(f"   Step {step}: loss={metrics['loss']:.4f}, perplexity={metrics['perplexity']:.4f}")
    
    # 4. Load and analyze metrics
    print(f"\n4. Loading and analyzing metrics...")
    all_metrics = logger.load_metrics()
    print(f"   Total logged entries: {len(all_metrics)}")
    print(f"   First entry: step {all_metrics[0]['step']}")
    print(f"   Last entry: step {all_metrics[-1]['step']}")
    
    # 5. Get latest step
    latest_step = logger.get_latest_step()
    print(f"\n5. Latest logged step: {latest_step}")
    
    # 6. Get metrics summary
    print(f"\n6. Metrics summary:")
    summary = logger.get_metrics_summary()
    print(f"   Total steps logged: {summary['total_steps']}")
    print(f"   Loss: {summary['loss_max']:.4f} → {summary['loss_final']:.4f} (min: {summary['loss_min']:.4f})")
    print(f"   Perplexity: {summary['perplexity_max']:.4f} → {summary['perplexity_final']:.4f}")
    print(f"   Tokens/sec: {summary['tokens_per_sec_mean']:.1f} (avg)")
    
    # 7. Show JSONL format
    print(f"\n7. Sample JSONL entries:")
    with open(logger.metrics_file, 'r') as f:
        lines = f.readlines()
        print(f"   First entry: {lines[0].strip()[:80]}...")
        print(f"   Last entry: {lines[-1].strip()[:80]}...")
    
    # 8. Show human-readable log
    print(f"\n8. Sample human-readable log entries:")
    with open(logger.training_log, 'r') as f:
        lines = f.readlines()
        print(f"   {lines[0].strip()}")
        print(f"   {lines[-1].strip()}")
    
    # 9. Demonstrate persistence across logger instances
    print(f"\n9. Testing persistence across logger instances...")
    logger2 = MetricsLogger(log_dir=dirs['logs'], log_interval=10)
    loaded_metrics = logger2.load_metrics()
    print(f"   New logger loaded {len(loaded_metrics)} entries")
    print(f"   Latest step from new logger: {logger2.get_latest_step()}")
    
    # 10. Cleanup
    print(f"\n10. Cleaning up demo files...")
    import shutil
    shutil.rmtree(dirs['root'])
    print(f"   Removed {dirs['root']}")
    
    print("\n" + "=" * 60)
    print("MetricsLogger demo complete!")
    print("=" * 60)


def demo_integrated_monitoring():
    """Demonstrate CheckpointManager and MetricsLogger working together"""
    print("\n" + "=" * 60)
    print("Integrated Monitoring Demo")
    print("=" * 60)
    
    # Setup
    exp_id = generate_experiment_id("integrated_demo")
    base_dir = "experiments"
    dirs = create_experiment_directory(base_dir, exp_id)
    print(f"\n1. Created experiment: {exp_id}")
    
    # Initialize monitoring components
    checkpoint_manager = CheckpointManager(
        experiment_dir=dirs['root'],
        save_interval=50,
        keep_last_n=2
    )
    
    metrics_logger = MetricsLogger(
        log_dir=dirs['logs'],
        log_interval=10
    )
    
    print(f"\n2. Initialized monitoring components:")
    print(f"   CheckpointManager: save every {checkpoint_manager.save_interval} steps")
    print(f"   MetricsLogger: log every {metrics_logger.log_interval} steps")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with integrated monitoring
    print(f"\n3. Training with integrated monitoring...")
    loss_history = []
    config = {'model': 'demo', 'lr': 0.001}
    
    for step in range(1, 151):
        # Training step
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Metrics logging
        if metrics_logger.should_log(step):
            metrics = {
                'loss': loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'grad_norm': sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            }
            metrics_logger.log_metrics(step, metrics)
            print(f"   Step {step}: loss={loss.item():.4f}")
        
        # Checkpointing
        if checkpoint_manager.should_checkpoint(step):
            checkpoint_manager.save_checkpoint(step, model, optimizer, loss_history, config)
            print(f"   Step {step}: Checkpoint saved")
    
    # Summary
    print(f"\n4. Training complete! Summary:")
    
    # Metrics summary
    summary = metrics_logger.get_metrics_summary()
    print(f"   Metrics logged: {summary['total_steps']} entries")
    print(f"   Loss: {summary['loss_max']:.4f} → {summary['loss_final']:.4f}")
    
    # Checkpoint summary
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"   Checkpoints saved: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"     - Step {cp['step']}: {cp['size_mb']:.2f} MB")
    
    # Cleanup
    print(f"\n5. Cleaning up...")
    import shutil
    shutil.rmtree(dirs['root'])
    
    print("\n" + "=" * 60)
    print("Integrated monitoring demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_checkpoint_manager()
    demo_interruption_recovery()
    demo_metrics_logger()
    demo_integrated_monitoring()



def demo_visualization_manager():
    """Demonstrate VisualizationManager usage"""
    print("\n" + "=" * 60)
    print("VisualizationManager Demo")
    print("=" * 60)
    
    from monitoring import VisualizationManager
    
    # Setup
    exp_id = generate_experiment_id("visualization_demo")
    base_dir = "experiments"
    dirs = create_experiment_directory(base_dir, exp_id)
    print(f"\n1. Created experiment: {exp_id}")
    
    # Initialize VisualizationManager
    viz_manager = VisualizationManager(
        viz_dir=dirs['visualizations'],
        viz_interval=50
    )
    
    print(f"\n2. Initialized VisualizationManager:")
    print(f"   Visualization directory: {viz_manager.viz_dir}")
    print(f"   Visualization interval: {viz_manager.viz_interval}")
    
    # Simulate training and generate visualizations
    print(f"\n3. Simulating training with visualizations...")
    
    loss_history = []
    metrics = {
        'learning_rate': [],
        'perplexity': [],
        'wave_ratio': []
    }
    
    for step in range(1, 151):
        # Simulate improving metrics
        loss = 2.0 - step * 0.01
        lr = 0.001 * (0.99 ** (step // 10))
        ppl = 7.5 - step * 0.04
        wave_ratio = 0.5 + step * 0.003
        
        loss_history.append(loss)
        metrics['learning_rate'].append(lr)
        metrics['perplexity'].append(ppl)
        metrics['wave_ratio'].append(wave_ratio)
        
        # Generate visualizations at intervals
        if viz_manager.should_visualize(step):
            viz_manager.generate_training_plots(step, loss_history, metrics)
            print(f"   Step {step}: Generated training plots")
    
    # List generated visualizations
    print(f"\n4. Generated visualizations:")
    import os
    viz_files = sorted(os.listdir(dirs['visualizations']))
    for viz_file in viz_files:
        file_path = os.path.join(dirs['visualizations'], viz_file)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"   - {viz_file} ({size_kb:.1f} KB)")
    
    # Demonstrate comparison plots
    print(f"\n5. Generating comparison plots...")
    experiments = [
        {
            'name': 'Experiment A',
            'loss_history': [2.0 - i * 0.01 for i in range(100)],
            'metrics': {
                'learning_rate': [0.001] * 100,
                'perplexity': [7.5 - i * 0.04 for i in range(100)]
            }
        },
        {
            'name': 'Experiment B',
            'loss_history': [2.5 - i * 0.015 for i in range(100)],
            'metrics': {
                'learning_rate': [0.0005] * 100,
                'perplexity': [8.0 - i * 0.05 for i in range(100)]
            }
        }
    ]
    
    viz_manager.generate_comparison_plots(experiments)
    print(f"   Generated experiment comparison plot")
    
    # Cleanup
    print(f"\n6. Cleaning up...")
    import shutil
    shutil.rmtree(dirs['root'])
    
    print("\n" + "=" * 60)
    print("VisualizationManager demo complete!")
    print("=" * 60)


def demo_config_tracker():
    """Demonstrate ConfigTracker usage"""
    print("\n" + "=" * 60)
    print("ConfigTracker Demo")
    print("=" * 60)
    
    from monitoring import ConfigTracker
    
    # Setup
    exp_id = generate_experiment_id("config_demo")
    base_dir = "experiments"
    dirs = create_experiment_directory(base_dir, exp_id)
    print(f"\n1. Created experiment: {exp_id}")
    
    # Initialize ConfigTracker
    tracker = ConfigTracker(experiment_dir=dirs['root'])
    print(f"\n2. Initialized ConfigTracker:")
    print(f"   Experiment directory: {tracker.experiment_dir}")
    print(f"   Config file: {tracker.config_file.name}")
    print(f"   Results file: {tracker.results_file.name}")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Save configuration
    print(f"\n3. Saving experiment configuration...")
    config = {
        'model': {
            'type': 'sequential',
            'layers': [10, 20, 5]
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 10,
            'optimizer': 'Adam'
        },
        'dataset': {
            'name': 'demo_dataset',
            'num_samples': 1000
        }
    }
    
    dataset_info = {
        'name': 'demo_dataset',
        'num_samples': 1000,
        'input_dim': 10,
        'output_dim': 5
    }
    
    tracker.save_config(config, model=model, dataset_info=dataset_info)
    print(f"   Configuration saved")
    
    # Load and display configuration
    print(f"\n4. Loading configuration...")
    loaded_config = tracker.load_config()
    print(f"   Experiment ID: {loaded_config['experiment_id']}")
    print(f"   Timestamp: {loaded_config['timestamp']}")
    print(f"   Git hash: {loaded_config['git_hash']}")
    print(f"   Model type: {loaded_config['model']['type']}")
    print(f"   Model parameters: {loaded_config['model']['num_parameters']:,}")
    print(f"   Hardware:")
    for key, value in loaded_config['hardware'].items():
        print(f"     - {key}: {value}")
    
    # Simulate training and save results
    print(f"\n5. Simulating training...")
    import time
    start_time = time.time()
    
    # Dummy training
    for _ in range(10):
        time.sleep(0.1)
    
    training_time = time.time() - start_time
    
    # Save results
    print(f"\n6. Saving experiment results...")
    final_metrics = {
        'val_loss': 0.45,
        'perplexity': 1.57,
        'best_val_loss': 0.42,
        'best_step': 850,
        'total_time_seconds': training_time,
        'tokens_per_second': 1250.5
    }
    
    generation_samples = [
        "This is a sample generated text.",
        "Another example of model output.",
        "The model learned to generate coherent text."
    ]
    
    tracker.save_results(
        final_metrics=final_metrics,
        best_checkpoint='checkpoint_step_850.pt',
        generation_samples=generation_samples
    )
    print(f"   Results saved")
    
    # Load and display results
    print(f"\n7. Loading results...")
    import json
    with open(tracker.results_file, 'r') as f:
        results = json.load(f)
    
    print(f"   Experiment ID: {results['experiment_id']}")
    print(f"   Final metrics:")
    for key, value in results['final_metrics'].items():
        if isinstance(value, float):
            print(f"     - {key}: {value:.4f}")
        else:
            print(f"     - {key}: {value}")
    print(f"   Best checkpoint: {results['best_checkpoint']}")
    print(f"   Generation samples: {len(results['generation_samples'])} samples")
    
    # Cleanup
    print(f"\n8. Cleaning up...")
    import shutil
    shutil.rmtree(dirs['root'])
    
    print("\n" + "=" * 60)
    print("ConfigTracker demo complete!")
    print("=" * 60)


def demo_full_monitoring_system():
    """Demonstrate all monitoring components working together"""
    print("\n" + "=" * 60)
    print("Full Monitoring System Demo")
    print("=" * 60)
    
    from monitoring import VisualizationManager, ConfigTracker
    
    # Setup
    exp_id = generate_experiment_id("full_monitoring_demo")
    base_dir = "experiments"
    dirs = create_experiment_directory(base_dir, exp_id)
    print(f"\n1. Created experiment: {exp_id}")
    print(f"   Directory structure:")
    for key, path in dirs.items():
        print(f"     - {key}: {path}")
    
    # Initialize all monitoring components
    checkpoint_manager = CheckpointManager(
        experiment_dir=dirs['root'],
        save_interval=50,
        keep_last_n=2
    )
    
    metrics_logger = MetricsLogger(
        log_dir=dirs['logs'],
        log_interval=10
    )
    
    viz_manager = VisualizationManager(
        viz_dir=dirs['visualizations'],
        viz_interval=50
    )
    
    config_tracker = ConfigTracker(
        experiment_dir=dirs['root']
    )
    
    print(f"\n2. Initialized all monitoring components:")
    print(f"   ✓ CheckpointManager (save every {checkpoint_manager.save_interval} steps)")
    print(f"   ✓ MetricsLogger (log every {metrics_logger.log_interval} steps)")
    print(f"   ✓ VisualizationManager (visualize every {viz_manager.viz_interval} steps)")
    print(f"   ✓ ConfigTracker")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Save initial configuration
    config = {
        'model': {'type': 'sequential', 'layers': [10, 20, 5]},
        'training': {'lr': 0.001, 'batch_size': 32, 'optimizer': 'Adam'},
        'dataset': {'name': 'demo', 'num_samples': 1000}
    }
    
    config_tracker.save_config(config, model=model)
    print(f"\n3. Saved initial configuration")
    
    # Training loop with full monitoring
    print(f"\n4. Training with full monitoring...")
    loss_history = []
    metrics_history = {
        'learning_rate': [],
        'perplexity': [],
        'wave_ratio': []
    }
    
    import time
    start_time = time.time()
    
    for step in range(1, 151):
        # Training step
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Simulate additional metrics
        lr = optimizer.param_groups[0]['lr']
        ppl = 7.5 - step * 0.04
        wave_ratio = 0.5 + step * 0.003
        
        metrics_history['learning_rate'].append(lr)
        metrics_history['perplexity'].append(ppl)
        metrics_history['wave_ratio'].append(wave_ratio)
        
        # Metrics logging
        if metrics_logger.should_log(step):
            metrics = {
                'loss': loss.item(),
                'learning_rate': lr,
                'perplexity': ppl,
                'wave_ratio': wave_ratio
            }
            metrics_logger.log_metrics(step, metrics)
        
        # Checkpointing
        if checkpoint_manager.should_checkpoint(step):
            checkpoint_manager.save_checkpoint(step, model, optimizer, loss_history, config)
            print(f"   Step {step}: Checkpoint saved")
        
        # Visualization
        if viz_manager.should_visualize(step):
            viz_manager.generate_training_plots(step, loss_history, metrics_history)
            print(f"   Step {step}: Visualizations generated")
    
    training_time = time.time() - start_time
    
    # Save final results
    print(f"\n5. Saving final results...")
    final_metrics = {
        'val_loss': loss_history[-1],
        'best_val_loss': min(loss_history),
        'best_step': loss_history.index(min(loss_history)) + 1,
        'total_time_seconds': training_time,
        'final_perplexity': metrics_history['perplexity'][-1]
    }
    
    config_tracker.save_results(final_metrics, best_checkpoint='checkpoint_step_100.pt')
    
    # Summary
    print(f"\n6. Experiment complete! Summary:")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Total steps: {len(loss_history)}")
    print(f"   Loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f}")
    
    # Metrics summary
    summary = metrics_logger.get_metrics_summary()
    print(f"   Metrics logged: {summary['total_steps']} entries")
    
    # Checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"   Checkpoints saved: {len(checkpoints)}")
    
    # Visualizations
    import os
    viz_files = os.listdir(dirs['visualizations'])
    print(f"   Visualizations generated: {len(viz_files)} files")
    
    # Files created
    print(f"\n7. Files created:")
    print(f"   ✓ {config_tracker.config_file.name}")
    print(f"   ✓ {config_tracker.results_file.name}")
    print(f"   ✓ {metrics_logger.metrics_file.name}")
    print(f"   ✓ {metrics_logger.training_log.name}")
    print(f"   ✓ {len(checkpoints)} checkpoint files")
    print(f"   ✓ {len(viz_files)} visualization files")
    
    # Cleanup
    print(f"\n8. Cleaning up...")
    import shutil
    shutil.rmtree(dirs['root'])
    
    print("\n" + "=" * 60)
    print("Full monitoring system demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_checkpoint_manager()
    demo_interruption_recovery()
    demo_metrics_logger()
    demo_integrated_monitoring()
    demo_visualization_manager()
    demo_config_tracker()
    demo_full_monitoring_system()
