#!/usr/bin/env python3
"""
llama.cpp Model Performance Tester
Test multiple GGUF models and compare performance
"""

import time
from pathlib import Path
from llama_cpp import Llama

# ============================================
# CONFIGURATION - Edit these!
# ============================================

# Models directory
MODELS_DIR = Path.home() / "Sentinel" / "modls"

# List your GGUF models here
MODELS_TO_TEST = [
    #"granite-3.0-1b-a400m-instruct.Q4_K_M.gguf",
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    # "gemma-2-2b-it-Q4_K_M.gguf",
]

# Or test just one model
SINGLE_MODEL = None  # Set to filename to test just one

# LLM Configuration
N_THREADS = 4  # Change to 2 if camera lags
N_CTX = 4096

# ============================================

# Test prompts (same as Ollama tester)
TEST_PROMPTS = [
    {
        "name": "Initial Greeting",
        "system": "You are an in-car voice assistant designed to keep a driver awake. Speak in short, calm sentences. Keep responses under 3 sentences.",
        "user": "I am feeling a little drowsy"
    },
    {
        "name": "Follow-up Question",
        "system": "You are an in-car voice assistant designed to keep a driver awake. Speak in short, calm sentences. Keep responses under 3 sentences.",
        "user": "I had a long day at work and I'm tired"
    },
    {
        "name": "Engagement",
        "system": "You are an in-car voice assistant designed to keep a driver awake. Speak in short, calm sentences. Keep responses under 3 sentences.",
        "user": "Yes, I would like to chat to stay awake"
    },
    {
        "name": "Simple Math",
        "system": "You are a helpful assistant.",
        "user": "What is 15 + 27?"
    },
    {
        "name": "Quick Fact",
        "system": "You are a helpful assistant.",
        "user": "Tell me one interesting fact about coffee"
    }
]

def load_model(model_path, n_threads=N_THREADS):
    """Load a GGUF model."""
    print(f"\nLoading model: {model_path.name}")
    print(f"  Threads: {n_threads}")
    print(f"  Context: {N_CTX}")
    
    try:
        start = time.time()
        
        llm = Llama(
            model_path=str(model_path),
            n_ctx=N_CTX,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU only on Pi
            verbose=False
        )
        
        load_time = time.time() - start
        
        # Get model info
        size_mb = model_path.stat().st_size / (1024 * 1024)
        
        print(f"  ✓ Loaded in {load_time:.2f}s")
        print(f"  Size: {size_mb:.1f} MB")
        
        return llm
        
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return None

def test_model(llm, model_name, prompt_data):
    """Test a single prompt with the model."""
    
    messages = [
        {"role": "system", "content": prompt_data["system"]},
        {"role": "user", "content": prompt_data["user"]}
    ]
    
    print(f"\n{'='*60}")
    print(f"Test: {prompt_data['name']}")
    print(f"{'='*60}")
    print(f"User: {prompt_data['user']}")
    print(f"\nGenerating response...")
    
    try:
        # Time the response
        start_time = time.time()
        
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        end_time = time.time()
        
        # Extract response
        assistant_msg = response['choices'][0]['message']['content']
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        words = assistant_msg.split()
        word_count = len(words)
        words_per_sec = word_count / elapsed_time if elapsed_time > 0 else 0
        
        # Display results
        print(f"\n{'─'*60}")
        print(f"Assistant: {assistant_msg}")
        print(f"{'─'*60}")
        print(f"⏱️  Time taken: {elapsed_time:.2f} seconds")
        print(f"📝 Words generated: ~{word_count}")
        print(f"🚀 Speed: ~{words_per_sec:.1f} words/sec")
        print(f"{'='*60}")
        
        return {
            'success': True,
            'time': elapsed_time,
            'words': word_count,
            'words_per_sec': words_per_sec,
            'response': assistant_msg
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def test_single_model(model_path):
    """Test all prompts for a single model."""
    
    print("\n" + "="*60)
    print(f"TESTING MODEL: {model_path.name}")
    print("="*60)
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"\n  Available models in {MODELS_DIR}:")
        if MODELS_DIR.exists():
            for f in MODELS_DIR.glob("*.gguf"):
                print(f"    - {f.name}")
        return None
    
    # Load model
    llm = load_model(model_path)
    
    if llm is None:
        return None
    
    # Warm-up inference (first run is always slower)
    print("\nWarm-up inference...")
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("✓ Warm-up complete")
    
    # Run tests
    results = []
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[Test {i}/{len(TEST_PROMPTS)}]")
        result = test_model(llm, model_path.name, prompt)
        results.append(result)
        time.sleep(0.5)
    
    # Calculate summary
    successful_tests = [r for r in results if r.get('success')]
    
    if successful_tests:
        avg_time = sum(r['time'] for r in successful_tests) / len(successful_tests)
        avg_speed = sum(r['words_per_sec'] for r in successful_tests) / len(successful_tests)
        
        return {
            'model': model_path.name,
            'model_path': model_path,
            'success_count': len(successful_tests),
            'total_tests': len(results),
            'avg_time': avg_time,
            'avg_speed': avg_speed,
            'all_results': results
        }
    else:
        return {
            'model': model_path.name,
            'success_count': 0,
            'total_tests': len(results),
            'error': 'All tests failed'
        }

def test_all_models(model_list):
    """Test all models consecutively."""
    
    print("\n" + "#"*60)
    print("# TESTING MULTIPLE MODELS CONSECUTIVELY")
    print("#"*60)
    print(f"\nModels to test: {len(model_list)}")
    
    for model in model_list:
        model_path = MODELS_DIR / model
        status = "✓" if model_path.exists() else "❌ NOT FOUND"
        size = f"({model_path.stat().st_size / (1024**2):.0f} MB)" if model_path.exists() else ""
        print(f"  - {model:<40} {status} {size}")
    
    # Test each model
    all_results = []
    
    for i, model in enumerate(model_list, 1):
        print(f"\n\n{'#'*60}")
        print(f"# MODEL {i}/{len(model_list)}")
        print(f"{'#'*60}")
        
        model_path = MODELS_DIR / model
        result = test_single_model(model_path)
        
        if result:
            all_results.append(result)
        
        # Pause between models
        if i < len(model_list):
            print(f"\n⏸️  Pausing 2 seconds before next model...\n")
            time.sleep(2)
    
    return all_results

def print_comparison_summary(all_results):
    """Print comparison table of all tested models."""
    
    print("\n\n" + "="*70)
    print("COMPARISON SUMMARY - ALL MODELS")
    print("="*70)
    
    if not all_results:
        print("❌ No results to compare!")
        return
    
    # Header
    print(f"\n{'Model':<35} {'Threads':>8} {'Avg Time':>12} {'Avg Speed':>12} {'Rating':>8}")
    print("─"*70)
    
    # Sort by speed (fastest first)
    sorted_results = sorted(all_results, key=lambda x: x.get('avg_speed', 0), reverse=True)
    
    for result in sorted_results:
        model = result['model']
        
        if result.get('error'):
            print(f"{model:<35} {N_THREADS:>8} {'ERROR':>12} {'ERROR':>12} {'-':>8}")
            continue
        
        avg_time = result['avg_time']
        avg_speed = result['avg_speed']
        
        # Speed rating
        if avg_speed >= 10:
            rating = "⭐⭐⭐⭐⭐"
        elif avg_speed >= 8:
            rating = "⭐⭐⭐⭐"
        elif avg_speed >= 6:
            rating = "⭐⭐⭐"
        elif avg_speed >= 4:
            rating = "⭐⭐"
        else:
            rating = "⭐"
        
        print(f"{model:<35} {N_THREADS:>8} {avg_time:>10.2f}s {avg_speed:>10.1f}/s {rating:>8}")
    
    print("\n" + "─"*70)
    
    # Winner
    if sorted_results:
        winner = sorted_results[0]
        print(f"\n🏆 FASTEST MODEL: {winner['model']}")
        print(f"   Speed: {winner['avg_speed']:.1f} words/sec")
        print(f"   Time: {winner['avg_time']:.2f}s average")
        print(f"   Threads: {N_THREADS}")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        if winner['avg_speed'] >= 8:
            print(f"   ✓ {winner['model']} is excellent for real-time!")
        elif winner['avg_speed'] >= 6:
            print(f"   ✓ {winner['model']} is good for real-time.")
        else:
            print(f"   ⚠️  {winner['avg_speed']:.1f} words/sec is borderline.")
            print(f"   Consider testing smaller models or more threads.")
    
    print("="*70)

def quick_test(model_path):
    """Quick single-prompt test."""
    
    print("\n" + "="*60)
    print(f"QUICK TEST: {model_path.name}")
    print("="*60)
    
    llm = load_model(model_path)
    
    if llm is None:
        return
    
    # Warm-up
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5
    )
    
    # Test first prompt only
    result = test_model(llm, model_path.name, TEST_PROMPTS[0])
    
    if result.get('success'):
        print(f"\n✓ Quick Results:")
        print(f"  Speed: {result['words_per_sec']:.1f} words/sec")
        print(f"  Time: {result['time']:.2f}s")

def list_available_models():
    """List all GGUF models in models directory."""
    
    print("\n" + "="*60)
    print("AVAILABLE GGUF MODELS")
    print("="*60)
    print(f"Directory: {MODELS_DIR}\n")
    
    if not MODELS_DIR.exists():
        print("❌ Models directory doesn't exist!")
        print(f"   Create it: mkdir -p {MODELS_DIR}")
        return
    
    models = list(MODELS_DIR.glob("*.gguf"))
    
    if not models:
        print("❌ No GGUF models found!")
        print("\nDownload models from HuggingFace:")
        print("  - granite-3.0-1b: Q4_K_M quantization")
        print("  - qwen2.5-0.5b: Q4_K_M quantization")
        return
    
    print(f"Found {len(models)} model(s):\n")
    
    for model in sorted(models):
        size_mb = model.stat().st_size / (1024 * 1024)
        
        # Detect quantization level
        quant = "Unknown"
        if "Q4_K_M" in model.name or "q4_k_m" in model.name:
            quant = "Q4_K_M (Recommended)"
        elif "Q3" in model.name:
            quant = "Q3 (Faster, lower quality)"
        elif "Q5" in model.name:
            quant = "Q5 (Slower, higher quality)"
        elif "Q8" in model.name:
            quant = "Q8 (Very slow, best quality)"
        
        print(f"  {model.name}")
        print(f"    Size: {size_mb:.1f} MB")
        print(f"    Quantization: {quant}\n")
    
    print("="*60)

def main():
    """Run tests based on configuration."""
    import sys
    
    # Check for command line args
    mode = "full"
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    if mode == "--list":
        list_available_models()
        return
    
    if mode == "--quick":
        models = [SINGLE_MODEL] if SINGLE_MODEL else MODELS_TO_TEST
        for model in models:
            quick_test(MODELS_DIR / model)
        return
    
    # Full test mode
    if SINGLE_MODEL:
        # Test single model
        result = test_single_model(MODELS_DIR / SINGLE_MODEL)
        
        if result and result['success_count'] > 0:
            print("\n\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Model: {result['model']}")
            print(f"Threads: {N_THREADS}")
            print(f"✓ Successful tests: {result['success_count']}/{result['total_tests']}")
            print(f"⏱️  Average time: {result['avg_time']:.2f} seconds")
            print(f"🚀 Average speed: {result['avg_speed']:.1f} words/sec")
            print("="*60)
    else:
        # Test all models
        all_results = test_all_models(MODELS_TO_TEST)
        print_comparison_summary(all_results)

if __name__ == "__main__":
    # ========================================
    # USAGE:
    # python llamacpp_tester.py           - Test all models (full)
    # python llamacpp_tester.py --quick   - Quick test (first prompt only)
    # python llamacpp_tester.py --list    - List available models
    # 
    # Or set SINGLE_MODEL at top to test just one
    # ========================================
    
    main()
