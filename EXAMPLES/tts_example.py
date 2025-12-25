import pyttsx3
import time

def test_single_engine_issue():
    """Demonstrates the issue where engine stops working after first use."""
    print("=" * 50)
    print("Test 1: Single Engine (Demonstrates the Bug)")
    print("=" * 50)
    
    try:
        engine = pyttsx3.init()
        
        print("\nFirst speech (should work):")
        engine.say("This is the first message.")
        engine.runAndWait()
        print("✓ First speech completed")
        
        time.sleep(1)
        
        print("\nSecond speech (may not work):")
        engine.say("This is the second message.")
        engine.runAndWait()
        print("✓ Second speech completed")
        
        time.sleep(1)
        
        print("\nThird speech (may not work):")
        engine.say("This is the third message.")
        engine.runAndWait()
        print("✓ Third speech completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_reinit_solution():
    """Solution 1: Reinitialize engine each time."""
    print("\n" + "=" * 50)
    print("Test 2: Reinitialize Engine Each Time (SOLUTION 1)")
    print("=" * 50)
    
    def speak(text):
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\nFirst speech:")
    speak("This is the first message with reinitialization.")
    print("✓ First speech completed")
    
    time.sleep(1)
    
    print("\nSecond speech:")
    speak("This is the second message with reinitialization.")
    print("✓ Second speech completed")
    
    time.sleep(1)
    
    print("\nThird speech:")
    speak("This is the third message with reinitialization.")
    print("✓ Third speech completed")

def test_global_engine_with_stop():
    """Solution 2: Use global engine but call stop() after each use."""
    print("\n" + "=" * 50)
    print("Test 3: Global Engine with stop() (SOLUTION 2)")
    print("=" * 50)
    
    engine = pyttsx3.init()
    
    def speak(text):
        try:
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\nFirst speech:")
    speak("This is the first message with stop.")
    print("✓ First speech completed")
    
    time.sleep(1)
    
    print("\nSecond speech:")
    speak("This is the second message with stop.")
    print("✓ Second speech completed")
    
    time.sleep(1)
    
    print("\nThird speech:")
    speak("This is the third message with stop.")
    print("✓ Third speech completed")

def test_queue_solution():
    """Solution 3: Queue all messages before running."""
    print("\n" + "=" * 50)
    print("Test 4: Queue Multiple Messages (SOLUTION 3)")
    print("=" * 50)
    
    try:
        engine = pyttsx3.init()
        
        messages = [
            "First queued message.",
            "Second queued message.",
            "Third queued message."
        ]
        
        print("\nQueueing all messages...")
        for i, msg in enumerate(messages, 1):
            print(f"  Queueing message {i}")
            engine.say(msg)
        
        print("\nPlaying all queued messages...")
        engine.runAndWait()
        print("✓ All messages completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_conversation_simulation():
    """Simulate a conversation like your drowsiness system."""
    print("\n" + "=" * 50)
    print("Test 5: Conversation Simulation")
    print("=" * 50)
    
    def speak_safe(text):
        """Safe speak function that reinitializes each time."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine  # Force cleanup
        except Exception as e:
            print(f"❌ Speech error: {e}")
    
    print("\nSimulating assistant conversation:")
    
    messages = [
        "I notice you're feeling drowsy.",
        "Let me help you stay alert.",
        "Would you like to chat for a bit?",
        "Okay, drive safely!",
        "Resuming monitoring."
    ]
    
    for i, msg in enumerate(messages, 1):
        print(f"\n  Message {i}: {msg}")
        speak_safe(msg)
        time.sleep(0.5)
    
    print("\n✓ Conversation simulation complete")

if __name__ == "__main__":
    print("🔊 PYTTSX3 DEBUGGING TESTS\n")
    
    # Test the problem
    test_single_engine_issue()
    
    time.sleep(2)
    
    # Test solutions
    test_reinit_solution()
    
    time.sleep(2)
    
    test_global_engine_with_stop()
    
    time.sleep(2)
    
    test_queue_solution()
    
    time.sleep(2)
    
    test_conversation_simulation()
    
    print("\n" + "=" * 50)
    print("🎉 All tests complete!")
    print("=" * 50)
    print("\nRecommendation: Use Solution 1 or 5 (reinitialize each time)")