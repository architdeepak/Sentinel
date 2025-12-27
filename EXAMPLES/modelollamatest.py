#!/usr/bin/env python3
"""
Ollama Model Performance Tester
Quick script to test different models for drowsiness assistant
"""

import ollama
import time

# ============================================
# CHANGE MODEL HERE - Just edit this line!
# ============================================
MODEL_NAME = "granite3-moe:1b"

# Available models you can try:
# - granite3-moe:1b
# - qwen2.5:0.5b
# - qwen:4b
# - phi4-mini
# - gemma2:2b
# - tinyllama
# - llama3.2:1b
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
    print(f"\n{'Generating response...'}")
    
    # Time the response
    start_time = time.time()
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={
                "temperature": 0.7,
                "max_tokens": 150  # Limit for faster testing
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

def run_all_tests(model_name):
    """Run all test prompts and show summary."""
    
    print("\n" + "="*60)
    print(f"TESTING MODEL: {model_name}")
    print("="*60)
    
    # Check if model exists
    try:
        ollama.show(model_name)
        print(f"✓ Model '{model_name}' found")
    except Exception as e:
        print(f"❌ Model '{model_name}' not found!")
        print(f"   Run: ollama pull {model_name}")
        return
    
    results = []
    
    # Run each test
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n\n[Test {i}/{len(TEST_PROMPTS)}]")
        result = test_model(model_name, prompt)
        results.append(result)
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful_tests = [r for r in results if r.get('success')]
    
    if successful_tests:
        avg_time = sum(r['time'] for r in successful_tests) / len(successful_tests)
        avg_speed = sum(r['tokens_per_sec'] for r in successful_tests) / len(successful_tests)
        
        print(f"✓ Successful tests: {len(successful_tests)}/{len(results)}")
        print(f"⏱️  Average time: {avg_time:.2f} seconds")
        print(f"🚀 Average speed: {avg_speed:.1f} words/sec")
        
        # Quality rating (manual)
        print(f"\n📊 MANUAL QUALITY RATING:")
        print(f"   1 = Poor (nonsense, too long, irrelevant)")
        print(f"   2 = Fair (some issues, usable)")
        print(f"   3 = Good (works well, appropriate)")
        print(f"   4 = Excellent (perfect responses)")
        print(f"\n   Rate this model: ___/4")
    else:
        print("❌ All tests failed!")
    
    print("="*60)

def quick_test(model_name):
    """Run just one quick test for rapid model comparison."""
    
    print("\n" + "="*60)
    print(f"QUICK TEST: {model_name}")
    print("="*60)
    
    # Just test the first prompt
    result = test_model(model_name, TEST_PROMPTS[0])
    
    if result.get('success'):
        print(f"\n✓ Quick Rating:")
        print(f"  Speed: {result['tokens_per_sec']:.1f} words/sec")
        print(f"  Time: {result['time']:.2f}s")
        print(f"  Quality: Check the response above!")

if __name__ == "__main__":
    # ========================================
    # CHOOSE TEST MODE
    # ========================================
    
    # Option 1: Full test suite (recommended first time)
    run_all_tests(MODEL_NAME)
    
    # Option 2: Quick single test (uncomment to use instead)
    # quick_test(MODEL_NAME)
    
    # ========================================
    # TO TEST ANOTHER MODEL:
    # 1. Change MODEL_NAME at the top
    # 2. Run script again
    # ========================================