"""
Demo script for paper generation infrastructure

Shows how to generate intuitive guides and technical papers from experiment results.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectral_gpt.paper_generator import SpectralGPTPaperGenerator


def demo_paper_generation():
    """Demonstrate paper generation capabilities"""
    
    print("=" * 80)
    print("Paper Generation Demo")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path("experiments/paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    print("Initializing paper generator...")
    generator = SpectralGPTPaperGenerator(
        output_dir=str(output_dir),
        experiments_base_dir="experiments"
    )
    print(f"✓ Output directory: {output_dir}")
    print()
    
    # Generate intuitive guide
    print("-" * 80)
    print("Generating Intuitive Guide...")
    print("-" * 80)
    
    guide_path = generator.generate_intuitive_guide(experiments=[])
    
    print(f"\n✓ Generated intuitive guide: {guide_path}")
    print(f"  Size: {os.path.getsize(guide_path):,} bytes")
    
    # Show preview
    with open(guide_path, 'r') as f:
        lines = f.readlines()
        print("\n  Preview (first 20 lines):")
        for i, line in enumerate(lines[:20], 1):
            print(f"    {i:2d}: {line.rstrip()}")
    
    print()
    
    # Generate technical paper
    print("-" * 80)
    print("Generating Technical Paper...")
    print("-" * 80)
    
    paper_path = generator.generate_technical_paper(experiments=[])
    
    print(f"\n✓ Generated technical paper: {paper_path}")
    print(f"  Size: {os.path.getsize(paper_path):,} bytes")
    
    # Show preview
    with open(paper_path, 'r') as f:
        lines = f.readlines()
        print("\n  Preview (first 20 lines):")
        for i, line in enumerate(lines[:20], 1):
            print(f"    {i:2d}: {line.rstrip()}")
    
    print()
    
    # Generate layer comparison
    print("-" * 80)
    print("Generating Layer Comparison...")
    print("-" * 80)
    
    wave_info = {
        'num_parameters': 67_473_706,
        'num_waves': 8,
        'num_harmonics': 3
    }
    
    standard_info = {
        'num_parameters': 52_892_160
    }
    
    comparison = generator.generate_layer_comparison(wave_info, standard_info)
    
    print("\n✓ Generated layer comparison")
    print(f"  Length: {len(comparison):,} characters")
    print("\n  Preview (first 500 chars):")
    print(f"    {comparison[:500]}...")
    
    print()
    
    # Generate fitting analysis
    print("-" * 80)
    print("Generating Fitting Analysis...")
    print("-" * 80)
    
    exp_data = [
        {
            'name': 'Standard Transformer',
            'loss_history': [8.3, 6.5, 5.7, 4.9, 4.44],
            'final_loss': 4.44
        },
        {
            'name': 'Spectral GPT',
            'loss_history': [7.9, 6.4, 5.6, 4.8, 4.48],
            'final_loss': 4.48
        }
    ]
    
    fitting = generator.generate_fitting_analysis(exp_data)
    
    print("\n✓ Generated fitting analysis")
    print(f"  Length: {len(fitting):,} characters")
    print("\n  Preview (first 500 chars):")
    print(f"    {fitting[:500]}...")
    
    print()
    
    # Check pandoc availability
    print("-" * 80)
    print("Checking PDF Generation Capability...")
    print("-" * 80)
    
    if generator._check_pandoc_available():
        print("\n✓ Pandoc is available - PDF generation supported")
        print("  You can convert markdown to PDF using:")
        print(f"    generator.render_to_pdf('{guide_path}')")
    else:
        print("\n⚠ Pandoc not available - PDF generation not supported")
        print("  Install pandoc to enable PDF rendering:")
        print("    https://pandoc.org/installing.html")
    
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("Generated documents:")
    print(f"  1. Intuitive Guide: {guide_path}")
    print(f"  2. Technical Paper: {paper_path}")
    print()
    print("The paper generator provides:")
    print("  ✓ Intuitive explanations with visual diagrams")
    print("  ✓ Technical papers with mathematical formulations")
    print("  ✓ Layer-by-layer architecture comparisons")
    print("  ✓ Fitting analysis explaining convergence")
    print("  ✓ Automatic figure referencing")
    print("  ✓ PDF rendering (if pandoc available)")
    print()
    print("Next steps:")
    print("  - Review generated markdown files")
    print("  - Customize sections as needed")
    print("  - Add experiment data for richer analysis")
    print("  - Generate PDFs for publication")
    print()


if __name__ == "__main__":
    demo_paper_generation()
