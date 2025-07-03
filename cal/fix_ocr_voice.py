# fix_ocr_voice.py - SAUNet 4.0 OCR & Voice Fix Tool
"""
Comprehensive diagnostic and fix tool for SAUNet 4.0 OCR and Voice issues.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def test_dependencies():
    """Test and diagnose dependency issues"""
    print("🔧 SAUNet 4.0 Dependency Diagnostic Tool")
    print("="*50)
    
    # Test OCR dependencies
    print("\n📷 OCR Dependencies:")
    try:
        import easyocr
        print("✅ EasyOCR: Available")
    except ImportError:
        print("❌ EasyOCR: Missing")
        print("💡 Install: pip install easyocr")
    
    try:
        import cv2
        print("✅ OpenCV: Available")
    except ImportError:
        print("❌ OpenCV: Missing") 
        print("💡 Install: pip install opencv-python")
    
    # Test Voice dependencies
    print("\n🎤 Voice Dependencies:")
    try:
        import speech_recognition as sr
        print("✅ SpeechRecognition: Available")
    except ImportError:
        print("❌ SpeechRecognition: Missing")
        print("💡 Install: pip install speechrecognition")
    
    try:
        import pyaudio
        print("✅ PyAudio: Available")
        # Test microphone access
        try:
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            print(f"   🎤 {device_count} audio devices found")
            p.terminate()
        except Exception as e:
            print(f"   ⚠️ Audio system error: {e}")
    except ImportError:
        print("❌ PyAudio: Missing")
        print("💡 Windows install: pip install pipwin && pipwin install pyaudio")
    
    try:
        import pyttsx3
        print("✅ pyttsx3 (TTS): Available")
    except ImportError:
        print("❌ pyttsx3: Missing")
        print("💡 Install: pip install pyttsx3")

def test_ocr_functionality():
    """Test OCR functionality"""
    print("\n📸 Testing OCR Functionality:")
    try:
        from solve_from_image import IntelligentMathAnalyzer
        analyzer = IntelligentMathAnalyzer()
        
        # Test text processing
        test_text = "solve x = 2 + 3"
        analysis = analyzer.analyze_mathematical_content(test_text)
        result = analyzer.solve_intelligently(analysis)
        
        if result['success']:
            print(f"✅ OCR processing works: '{test_text}' → {result['solutions']}")
        else:
            print(f"⚠️ OCR processing issue: {result}")
            
    except Exception as e:
        print(f"❌ OCR test failed: {e}")

def test_voice_functionality():
    """Test voice functionality"""
    print("\n🎙️ Testing Voice Recognition:")
    try:
        from voice_recognition import VoiceMathRecognizer
        recognizer = VoiceMathRecognizer()
        
        if recognizer.is_available():
            print("✅ Voice recognition is available")
            
            # Test speech processing
            test_phrase = "solve two plus three"
            processed = recognizer._process_mathematical_speech(test_phrase)
            print(f"✅ Speech processing: '{test_phrase}' → '{processed}'")
            
        else:
            print("❌ Voice recognition not available")
            print("💡 Check microphone and dependencies")
            
    except Exception as e:
        print(f"❌ Voice test failed: {e}")

def create_setup_instructions():
    """Create comprehensive setup instructions"""
    instructions = """
SAUNet 4.0 Setup Instructions

OCR SETUP:
1. pip install easyocr opencv-python pillow
2. For Tesseract: Download from https://github.com/tesseract-ocr/tesseract
3. pip install pytesseract

VOICE SETUP (Windows):
1. pip install speechrecognition pyttsx3
2. For PyAudio:
   Method 1: pip install pipwin && pipwin install pyaudio
   Method 2: pip install pyaudio
3. Check Windows microphone permissions

TROUBLESHOOTING:
- PyAudio issues: Try different installation methods
- Microphone not working: Check Windows Device Manager
- OCR fails: Ensure proper file permissions
- Import errors: Reinstall packages in clean environment

After installation, run this script again to verify fixes.
"""
    
    with open("SETUP_INSTRUCTIONS.txt", "w", encoding='utf-8') as f:
        f.write(instructions)
    print("Setup instructions saved to SETUP_INSTRUCTIONS.txt")

if __name__ == "__main__":
    test_dependencies()
    test_ocr_functionality() 
    test_voice_functionality()
    create_setup_instructions()
    
    print("\n🎯 Run 'python fix_ocr_voice.py' after installing dependencies to verify fixes") 