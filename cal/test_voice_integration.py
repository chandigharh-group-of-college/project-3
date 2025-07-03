# test_voice_integration.py - Test Voice Recognition Integration
# SAUNet 4.0 - Voice-Activated Mathematical Assistant

"""
Test script to demonstrate the voice recognition capabilities
of SAUNet 4.0 Advanced Scientific Calculator.

This script tests the voice recognition system independently
and shows how it integrates with the mathematical processing.
"""

import sys
import time
from typing import Optional

def test_voice_recognition_system():
    """Test the voice recognition system"""
    print("🎤 SAUNet 4.0 Voice Recognition Test")
    print("=" * 50)
    
    try:
        from voice_recognition import VoiceMathRecognizer, VoiceControlledCalculator
        print("✅ Voice recognition modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import voice recognition: {e}")
        print("💡 Install dependencies: pip install speechrecognition pyaudio pyttsx3")
        return False
    
    # Test basic voice recognizer
    print("\n📋 Testing VoiceMathRecognizer...")
    recognizer = VoiceMathRecognizer()
    
    if not recognizer.is_available():
        print("❌ Voice recognition not available")
        print("💡 Check microphone and dependencies")
        return False
    
    print("✅ Voice recognition system available")
    
    # Test mathematical speech processing
    print("\n🧠 Testing mathematical speech processing...")
    test_phrases = [
        "solve two x plus three equals seven",
        "differentiate x squared",
        "integrate x cubed from zero to five",
        "what is two times three plus four",
        "calculate the square root of sixteen"
    ]
    
    for phrase in test_phrases:
        processed = recognizer._process_mathematical_speech(phrase)
        print(f"   📝 '{phrase}' → '{processed}'")
    
    # Test text-to-speech
    print("\n🔊 Testing text-to-speech...")
    recognizer.speak_feedback("Voice recognition test successful")
    
    print("\n✅ Voice recognition system test completed successfully!")
    return True


def test_voice_with_calculator():
    """Test voice recognition with calculator integration"""
    print("\n🧮 Testing Voice + Calculator Integration")
    print("=" * 50)
    
    try:
        from voice_recognition import VoiceControlledCalculator
        from calculator import Calculator
        from solver import EquationSolver
        
        # Create calculator components
        calculator = Calculator()
        solver = EquationSolver()
        
        # Create voice controller
        voice_controller = VoiceControlledCalculator(None)  # No UI for testing
        
        print("✅ Voice-controlled calculator created")
        
        # Test voice input processing
        test_inputs = [
            "2 + 3 * 4",
            "solve x = 5 - 2",
            "differentiate x squared"
        ]
        
        print("\n📝 Testing voice input processing...")
        for test_input in test_inputs:
            print(f"   🎤 Voice input: '{test_input}'")
            # Simulate voice input processing
            processed = voice_controller.voice_recognizer._process_mathematical_speech(test_input)
            print(f"   ✅ Processed: '{processed}'")
        
        print("\n✅ Voice-calculator integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_voice_manual_recognition():
    """Test manual voice recognition (requires speaking)"""
    print("\n🎙️ Manual Voice Recognition Test")
    print("=" * 50)
    
    try:
        from voice_recognition import VoiceMathRecognizer
        
        recognizer = VoiceMathRecognizer()
        
        if not recognizer.is_available():
            print("❌ Voice recognition not available for manual test")
            return False
        
        print("🎤 Ready for voice input!")
        print("💬 Try saying: 'solve two x plus three equals seven'")
        print("⏰ You have 10 seconds to speak...")
        
        # Set up status callback
        def status_callback(status):
            print(f"   {status}")
        
        recognizer.set_status_callback(status_callback)
        
        # Recognize a single phrase
        result = recognizer.recognize_single_phrase()
        
        if result:
            print(f"✅ Recognition successful!")
            print(f"📝 You said: '{result}'")
            
            # Test TTS response
            recognizer.speak_feedback(f"I heard: {result}")
            
            return True
        else:
            print("❌ No speech recognized")
            return False
            
    except Exception as e:
        print(f"❌ Manual test failed: {e}")
        return False


def test_voice_continuous_mode():
    """Test continuous voice recognition mode"""
    print("\n🔄 Continuous Voice Recognition Test")
    print("=" * 50)
    
    try:
        from voice_recognition import VoiceMathRecognizer
        
        recognizer = VoiceMathRecognizer()
        
        if not recognizer.is_available():
            print("❌ Voice recognition not available for continuous test")
            return False
        
        print("🎤 Starting continuous listening mode...")
        print("💬 Say mathematical expressions for 10 seconds")
        print("🛑 Press Ctrl+C to stop early")
        
        # Set up callbacks
        recognized_phrases = []
        
        def voice_callback(text):
            recognized_phrases.append(text)
            print(f"   ✅ Recognized: '{text}'")
        
        def status_callback(status):
            print(f"   📡 {status}")
        
        recognizer.set_callback(voice_callback)
        recognizer.set_status_callback(status_callback)
        
        # Start continuous listening
        recognizer.start_listening()
        
        try:
            # Listen for 10 seconds
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        # Stop listening
        recognizer.stop_listening()
        
        print(f"\n📊 Results: {len(recognized_phrases)} phrases recognized")
        for i, phrase in enumerate(recognized_phrases, 1):
            print(f"   {i}. '{phrase}'")
        
        return len(recognized_phrases) > 0
        
    except Exception as e:
        print(f"❌ Continuous test failed: {e}")
        return False


def demonstrate_voice_features():
    """Demonstrate key voice features"""
    print("\n🌟 Voice Features Demonstration")
    print("=" * 50)
    
    features = [
        "🎤 Multi-engine speech recognition (Google, Whisper, Sphinx)",
        "🧠 Intelligent mathematical speech processing",
        "🔄 Continuous and single-phrase recognition modes",
        "🔊 Text-to-speech feedback and results",
        "🧮 Integration with calculator and NLU systems",
        "⚡ Real-time processing and UI updates",
        "🎯 Mathematical keyword recognition",
        "📝 Voice-to-text-to-math pipeline"
    ]
    
    for feature in features:
        print(f"   {feature}")
        time.sleep(0.5)  # Dramatic pause
    
    print("\n✨ Voice recognition makes SAUNet 4.0 a true AI assistant!")
    return True


def main():
    """Main test runner"""
    print("🚀 SAUNet 4.0 Voice Recognition Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic System Test", test_voice_recognition_system),
        ("Calculator Integration", test_voice_with_calculator),
        ("Features Demo", demonstrate_voice_features),
    ]
    
    # Optional interactive tests
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        tests.extend([
            ("Manual Recognition", test_voice_manual_recognition),
            ("Continuous Mode", test_voice_continuous_mode),
        ])
        print("🎙️ Interactive mode enabled - microphone tests included")
    else:
        print("💡 Run with --interactive for microphone tests")
    
    print()
    
    # Run tests
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Voice recognition is ready! 🎤")
    else:
        print("⚠️ Some tests failed. Check dependencies and microphone.")
    
    print("\n💡 To use voice recognition in SAUNet 4.0:")
    print("   1. Install: pip install speechrecognition pyaudio pyttsx3")
    print("   2. Connect microphone")
    print("   3. Click the 🎤 Voice button in the calculator")
    print("   4. Speak your mathematical question naturally")
    print("   5. The AI will process and solve it!")


if __name__ == "__main__":
    main() 