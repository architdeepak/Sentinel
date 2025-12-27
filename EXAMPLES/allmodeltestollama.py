#!/usr/bin/env python3
"""
Ollama Model Performance Tester
Test multiple models consecutively and compare results
"""

import ollama
import time

# ============================================
# LIST YOUR MODELS HERE - Add/remove as needed!
# ============================================
MODELS_TO_TEST = [
    "granite3-moe:1b",
    "qwen2.5:0.5b",
    "qwen:4b",
    "phi4-mini",
    "gemma2:2b",
    # "tinyllama",
    # "llama3.2:1b",
]

# Or test just one model (uncomment and set):
# SINGLE_MODEL = "granite3-moe:1b"
SINGLE_MODEL = None  # Set to None to test all models

# ============================================

# Test prompts for drowsiness assistant
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

def test_model(model_name, prompt_data):
    """Test a single prompt with the model and return timing info."""
    
    messages = [
        {"role": "system", "content": prompt_data["system"]},
        {"role": "user", "content": prompt_data["user"]}
    ]
    
    print(f"\n{'='*60}")
    print(f"Test: {prompt_data['name']}")
    print(f"{'='*60}")
    print(f"User: {prompt_data['user']}")
    print(f"\nGenerating response...")
    
    # Time the response
    start_time = time.time()
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={
                "temperature": 0.7,
                "num_predict": 150  # Limit for faster testing
            }
        )
        
        end_time = time.time()
        
        # Extract response
        assistant_msg = response.get('message', {}).get('content', 'No response')
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        
        # Count tokens (approximate - by words)
        words = assistant_msg.split()
        approx_tokens = len(words)
        tokens_per_sec = approx_tokens / elapsed_time if elapsed_time > 0 else 0
        
        # Display results
        print(f"\n{'─'*60}")
        print(f"Assistant: {assistant_msg}")
        print(f"{'─'*60}")
        print(f"⏱️  Time taken: {elapsed_time:.2f} seconds")
        print(f"📝 Words generated: ~{approx_tokens}")
        print(f"🚀 Speed: ~{tokens_per_sec:.1f} words/sec")
        print(f"{'='*60}")
        
        return {
            'success': True,
            'time': elapsed_time,
            'tokens': approx_tokens,
            'tokens_per_sec': tokens_per_sec,
            'response': assistant_msg
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def check_model_exists(model_name):
    """Check if model is available."""
    try:
        ollama.show(model_name)
        return True
    except:
        return False

def test_single_model(model_name):
    """Run all test prompts for a single model."""
    
    print("\n" + "="*60)
    print(f"TESTING MODEL: {model_name}")
    print("="*60)
    
    # Check if model exists
    if not check_model_exists(model_name):
        print(f"❌ Model '{model_name}' not found!")
        print(f"   Run: ollama pull {model_name}")
        return None
    
    print(f"✓ Model '{model_name}' found")
    
    results = []
    
    # Run each test
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[Test {i}/{len(TEST_PROMPTS)}]")
        result = test_model(model_name, prompt)
        results.append(result)
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Calculate summary stats
    successful_tests = [r for r in results if r.get('success')]
    
    if successful_tests:
        avg_time = sum(r['time'] for r in successful_tests) / len(successful_tests)
        avg_speed = sum(r['tokens_per_sec'] for r in successful_tests) / len(successful_tests)
        
        return {
            'model': model_name,
            'success_count': len(successful_tests),
            'total_tests': len(results),
            'avg_time': avg_time,
            'avg_speed': avg_speed,
            'all_results': results
        }
    else:
        return {
            'model': model_name,
            'success_count': 0,
            'total_tests': len(results),
            'error': 'All tests failed'
        }

def test_all_models(models_list):
    """Test all models consecutively and compare results."""
    
    print("\n" + "#"*60)
    print("# TESTING MULTIPLE MODELS CONSECUTIVELY")
    print("#"*60)
    print(f"\nModels to test: {len(models_list)}")
    for m in models_list:
        status = "✓" if check_model_exists(m) else "❌ NOT FOUND"
        print(f"  - {m:<30} {status}")
    
    # Test each model
    all_model_results = []
    
    for i, model in enumerate(models_list, 1):
        print(f"\n\n{'#'*60}")
        print(f"# MODEL {i}/{len(models_list)}")
        print(f"{'#'*60}")
        
        result = test_single_model(model)
        if result:
            all_model_results.append(result)
        
        # Pause between models
        if i < len(models_list):
            print(f"\n⏸️  Pausing 2 seconds before next model...\n")
            time.sleep(2)
    
    return all_model_results

def print_comparison_summary(all_results):
    """Print a comparison table of all tested models."""
    
    print("\n\n" + "="*70)
    print("COMPARISON SUMMARY - ALL MODELS")
    print("="*70)
    
    if not all_results:
        print("❌ No results to compare!")
        return
    
    # Header
    print(f"\n{'Model':<25} {'Avg Time':>12} {'Avg Speed':>12} {'Tests':>10} {'Rating':>8}")
    print("─"*70)
    
    # Sort by speed (fastest first)
    sorted_results = sorted(all_results, key=lambda x: x.get('avg_speed', 0), reverse=True)
    
    for result in sorted_results:
        model = result['model']
        
        if result.get('error'):
            print(f"{model:<25} {'ERROR':>12} {'ERROR':>12} {'-':>10} {'-':>8}")
            continue
        
        avg_time = result['avg_time']
        avg_speed = result['avg_speed']
        success = result['success_count']
        total = result['total_tests']
        
        # Speed rating
        if avg_speed >= 12:
            rating = "⭐⭐⭐⭐⭐"
        elif avg_speed >= 9:
            rating = "⭐⭐⭐⭐"
        elif avg_speed >= 6:
            rating = "⭐⭐⭐"
        elif avg_speed >= 3:
            rating = "⭐⭐"
        else:
            rating = "⭐"
        
        print(f"{model:<25} {avg_time:>10.2f}s {avg_speed:>10.1f}/s {success}/{total:>7} {rating:>8}")
    
    print("\n" + "─"*70)
    
    # Winner
    if sorted_results:
        winner = sorted_results[0]
        print(f"\n🏆 FASTEST MODEL: {winner['model']}")
        print(f"   Speed: {winner['avg_speed']:.1f} words/sec")
        print(f"   Time: {winner['avg_time']:.2f}s average")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        if winner['avg_speed'] >= 10:
            print(f"   ✓ {winner['model']} is excellent for real-time conversation!")
            print(f"   Use this model for your drowsiness detector.")
        elif winner['avg_speed'] >= 7:
            print(f"   ✓ {winner['model']} is good enough for real-time use.")
            print(f"   Should work well for your drowsiness detector.")
        else:
            print(f"   ⚠️  {winner['avg_speed']:.1f} words/sec is a bit slow.")
            print(f"   Consider testing smaller/faster models.")
    
    print("="*70)

def quick_comparison(models_list):
    """Quick test - only first prompt for each model."""
    
    print("\n" + "#"*60)
    print("# QUICK COMPARISON - FIRST PROMPT ONLY")
    print("#"*60)
    
    results = []
    
    for i, model in enumerate(models_list, 1):
        print(f"\n[Model {i}/{len(models_list)}: {model}]")
        
        if not check_model_exists(model):
            print(f"❌ Model not found: {model}")
            continue
        
        result = test_model(model, TEST_PROMPTS[0])
        
        if result.get('success'):
            results.append({
                'model': model,
                'time': result['time'],
                'speed': result['tokens_per_sec']
            })
        
        time.sleep(0.5)
    
    # Summary
    print("\n\n" + "="*60)
    print("QUICK COMPARISON RESULTS")
    print("="*60)
    print(f"\n{'Model':<25} {'Time':>10} {'Speed':>12}")
    print("─"*60)
    
    sorted_results = sorted(results, key=lambda x: x['speed'], reverse=True)
    
    for r in sorted_results:
        print(f"{r['model']:<25} {r['time']:>8.2f}s {r['speed']:>10.1f}/s")
    
    print("="*60)

def list_available_models():
    """List all available Ollama models."""
    try:
        models = ollama.list()
        print("\n" + "="*60)
        print("AVAILABLE MODELS")
        print("="*60)
        
        if models and 'models' in models:
            for model in models['models']:
                name = model.get('name', 'Unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f"  {name:<30} ({size:.1f} GB)")
        else:
            print("  No models found!")
        
        print("="*60)
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    mode = "full"  # default mode
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    # ========================================
    # CHOOSE TEST MODE
    # ========================================
    
    if mode == "--list":
        # List available models
        list_available_models()
    
    elif mode == "--quick":
        # Quick test (first prompt only)
        models = [SINGLE_MODEL] if SINGLE_MODEL else MODELS_TO_TEST
        quick_comparison(models)
    
    elif SINGLE_MODEL:
        # Test single model (full test)
        result = test_single_model(SINGLE_MODEL)
        
        if result and result['success_count'] > 0:
            print("\n\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"✓ Successful tests: {result['success_count']}/{result['total_tests']}")
            print(f"⏱️  Average time: {result['avg_time']:.2f} seconds")
            print(f"🚀 Average speed: {result['avg_speed']:.1f} words/sec")
            
            print(f"\n📊 MANUAL QUALITY RATING:")
            print(f"   1 = Poor (nonsense, too long, irrelevant)")
            print(f"   2 = Fair (some issues, usable)")
            print(f"   3 = Good (works well, appropriate)")
            print(f"   4 = Excellent (perfect responses)")
            print(f"\n   Rate this model: ___/4")
            print("="*60)
    
    else:
        # Test all models (full test)
        all_results = test_all_models(MODELS_TO_TEST)
        print_comparison_summary(all_results)
    
    # ========================================
    # USAGE:
    # python model_tester.py           - Test all models (full)
    # python model_tester.py --quick   - Quick test (first prompt only)
    # python model_tester.py --list    - List available models
    # 
    # Or set SINGLE_MODEL = "model-name" at top to test just one
    # ========================================