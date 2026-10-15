"""
Demo: Code Extraction for Documentation

This script demonstrates the code extraction functionality for generating
academic documentation with real code examples.
"""

import sys
from pathlib import Path

# Add spectral_gpt to path
sys.path.insert(0, str(Path(__file__).parent.parent / "spectral_gpt"))

from code_extractor import CodeExtractor
from code_examples import SpectralGPTCodeExamples


def demo_basic_extraction():
    """Demonstrate basic code extraction."""
    print("=" * 70)
    print("DEMO 1: Basic Code Extraction")
    print("=" * 70)
    
    extractor = CodeExtractor()
    
    # Extract WavePacketEmbedding class
    print("\n1. Extracting WavePacketEmbedding class...")
    snippet = extractor.extract_class(
        "spectral_gpt/wave_gpt.py",
        "WavePacketEmbedding",
        include_methods=["forward"]
    )
    
    if snippet:
        print(f"   ✓ Extracted: {snippet.name}")
        print(f"   - Type: {snippet.snippet_type}")
        print(f"   - Lines: {snippet.start_line}-{snippet.end_line}")
        print(f"   - Docstring: {snippet.docstring[:80]}..." if snippet.docstring else "   - No docstring")
        print(f"   - Code length: {len(snippet.code)} characters")
    
    # Format for markdown
    print("\n2. Formatting for markdown...")
    markdown = extractor.format_for_markdown(snippet, max_lines=20)
    print(f"   ✓ Generated markdown ({len(markdown)} characters)")
    print("\n   Preview:")
    print("   " + "\n   ".join(markdown.split('\n')[:10]))
    print("   ...")


def demo_spectral_gpt_examples():
    """Demonstrate Spectral GPT specific examples."""
    print("\n" + "=" * 70)
    print("DEMO 2: Spectral GPT Code Examples")
    print("=" * 70)
    
    examples = SpectralGPTCodeExamples()
    
    # Extract key components
    components = [
        ("Wave Packet Embedding", lambda: examples.get_wave_packet_embedding()),
        ("RGD Optimizer", lambda: examples.get_rgd_optimizer()),
        ("QFE Loss", lambda: examples.get_qfe_loss()),
    ]
    
    for name, extractor_func in components:
        print(f"\n{name}:")
        snippet = extractor_func()
        if snippet:
            print(f"   ✓ Extracted {snippet.name}")
            print(f"   - Lines: {snippet.end_line - snippet.start_line}")
            print(f"   - Has docstring: {'Yes' if snippet.docstring else 'No'}")
        else:
            print("   ✗ Not found")


def demo_api_examples():
    """Demonstrate API usage examples."""
    print("\n" + "=" * 70)
    print("DEMO 3: API Usage Examples")
    print("=" * 70)
    
    examples = SpectralGPTCodeExamples()
    
    # Get API usage example
    print("\n1. High-Level API Usage:")
    api_example = examples.get_api_usage_example()
    lines = api_example.split('\n')
    print(f"   ✓ Generated {len(lines)} lines of example code")
    print("\n   Preview:")
    for line in lines[:15]:
        print(f"   {line}")
    print("   ...")
    
    # Get training example
    print("\n2. Complete Training Example:")
    training_example = examples.get_training_example()
    lines = training_example.split('\n')
    print(f"   ✓ Generated {len(lines)} lines of training code")


def demo_code_appendix():
    """Demonstrate generating a complete code appendix."""
    print("\n" + "=" * 70)
    print("DEMO 4: Generate Code Appendix")
    print("=" * 70)
    
    examples = SpectralGPTCodeExamples()
    
    print("\nGenerating complete code appendix...")
    appendix = examples.generate_code_appendix(
        output_file="experiments/paper/demo_code_appendix.md"
    )
    
    print("✓ Generated appendix:")
    print(f"  - Total length: {len(appendix)} characters")
    print("  - Sections: Wave Packet Embedding, Interference Attention, RGD, QFE, API")
    print("  - Saved to: experiments/paper/demo_code_appendix.md")
    
    # Show structure
    print("\n  Appendix structure:")
    for line in appendix.split('\n'):
        if line.startswith('##'):
            print(f"    {line}")


def demo_integration_with_paper_generator():
    """Demonstrate integration with paper generator."""
    print("\n" + "=" * 70)
    print("DEMO 5: Integration with Paper Generator")
    print("=" * 70)
    
    from paper_generator import SpectralGPTPaperGenerator
    
    print("\n1. Creating paper generator with code extraction...")
    generator = SpectralGPTPaperGenerator(
        output_dir="experiments/paper",
        experiments_base_dir="experiments"
    )
    
    if generator.code_examples:
        print("   ✓ Code extraction enabled")
        
        print("\n2. Generating code appendix via paper generator...")
        appendix = generator.generate_code_appendix(
            output_file="experiments/paper/integrated_code_appendix.md"
        )
        print(f"   ✓ Generated appendix ({len(appendix)} characters)")
        print("   ✓ Saved to: experiments/paper/integrated_code_appendix.md")
    else:
        print("   ✗ Code extraction not available")


def main():
    """Run all demos."""
    print("\n" + "🔬" * 35)
    print("CODE EXTRACTION DEMO FOR SPECTRAL GPT DOCUMENTATION")
    print("🔬" * 35)
    
    try:
        demo_basic_extraction()
        demo_spectral_gpt_examples()
        demo_api_examples()
        demo_code_appendix()
        demo_integration_with_paper_generator()
        
        print("\n" + "=" * 70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - experiments/paper/demo_code_appendix.md")
        print("  - experiments/paper/integrated_code_appendix.md")
        print("\nThese files contain real extracted code from the implementation")
        print("and can be included in academic papers for reproducibility.")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
