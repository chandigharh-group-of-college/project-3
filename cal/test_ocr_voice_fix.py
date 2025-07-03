# test_ocr_voice_fix.py - Comprehensive OCR & Voice Test Script
# SAUNet 4.0 - Diagnostic and Setup Validation

"""
Comprehensive test script to diagnose and fix OCR and Voice Recording issues
in SAUNet 4.0 Advanced Scientific Calculator.

This script will:
1. Test all dependencies
2. Validate OCR functionality 
3. Test voice recognition with detailed error reporting
4. Provide specific setup instructions for Windows
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

def print_header(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print('='*60)

def test_basic_dependencies():
    """Test basic Python dependencies"""
    print_header("BASIC DEPENDENCIES TEST")
    
    dependencies = {
        'PyQt5': ['PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui'],
        'NumPy': ['numpy'],
        'OpenCV': ['cv2'],
        'SymPy': ['sympy'],
        'Pillow': ['PIL']
    }
    
    results = {}
    
    for lib_name, modules in dependencies.items():
        try:
            for module in modules:
                __import__(module)
            print(f"✅ {lib_name}: Available")
            results[lib_name] = True
        except ImportError as e:
            print(f"❌ {lib_name}: Missing - {e}")
            results[lib_name] = False
    
    return results

def test_ocr_dependencies():
    """Test OCR-specific dependencies"""
    print_header("OCR DEPENDENCIES TEST")
    
    ocr_results = {}
    
    # Test EasyOCR
    try:
        import easyocr
        print("✅ EasyOCR: Available")
        print(f"   Version: {easyocr.__version__ if hasattr(easyocr, '__version__') else 'Unknown'}")
        ocr_results['easyocr'] = True
    except ImportError as e:
        print(f"❌ EasyOCR: Missing - {e}")
        print("💡 Install: pip install easyocr")
        ocr_results['easyocr'] = False
    
    # Test PyTesseract
    try:
        import pytesseract
        print("✅ PyTesseract: Available")
        # Try to get Tesseract version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"   Tesseract version: {version}")
        except Exception as e:
            print(f"⚠️ Tesseract executable issue: {e}")
            print("💡 Install Tesseract: https://github.com/tesseract-ocr/tesseract")
        ocr_results['pytesseract'] = True
    except ImportError as e:
        print(f"❌ PyTesseract: Missing - {e}")
        print("💡 Install: pip install pytesseract")
        ocr_results['pytesseract'] = False
    
    # Test advanced preprocessing dependencies
    advanced_deps = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision', 
        'skimage': 'scikit-image',
        'rembg': 'Background Removal'
    }
    
    for module, name in advanced_deps.items():
        try:
            __import__(module)
            print(f"✅ {name}: Available")
            ocr_results[module] = True
        except ImportError:
            print(f"⚠️ {name}: Missing (optional)")
            ocr_results[module] = False
    
    return ocr_results

def test_voice_dependencies():
    """Test voice recognition dependencies"""
    print_header("VOICE RECOGNITION DEPENDENCIES TEST")
    
    voice_results = {}
    
    # Test SpeechRecognition
    try:
        import speech_recognition as sr
        print("✅ SpeechRecognition: Available")
        print(f"   Version: {sr.__version__ if hasattr(sr, '__version__') else 'Unknown'}")
        voice_results['speechrecognition'] = True
    except ImportError as e:
        print(f"❌ SpeechRecognition: Missing - {e}")
        print("💡 Install: pip install speechrecognition")
        voice_results['speechrecognition'] = False
    
    # Test PyAudio
    try:
        import pyaudio
        print("✅ PyAudio: Available")
        
        # Test PyAudio initialization
        try:
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            print(f"   Audio devices found: {device_count}")
            
            # List microphones
            microphones = []
            for i in range(device_count):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    microphones.append(device_info['name'])
            
            print(f"   Microphones available: {len(microphones)}")
            for i, mic in enumerate(microphones[:3]):  # Show first 3
                print(f"     {i+1}. {mic}")
            
            p.terminate()
            voice_results['pyaudio_functional'] = True
            
        except Exception as e:
            print(f"⚠️ PyAudio initialization failed: {e}")
            voice_results['pyaudio_functional'] = False
            
        voice_results['pyaudio'] = True
        
    except ImportError as e:
        print(f"❌ PyAudio: Missing - {e}")
        print("💡 Windows install: pip install pipwin && pipwin install pyaudio")
        print("💡 Alternative: pip install pyaudio")
        voice_results['pyaudio'] = False
        voice_results['pyaudio_functional'] = False
    
    # Test Text-to-Speech
    try:
        import pyttsx3
        print("✅ pyttsx3 (TTS): Available")
        
        # Test TTS initialization
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            print(f"   TTS voices available: {len(voices) if voices else 0}")
            engine.stop()
            voice_results['tts_functional'] = True
        except Exception as e:
            print(f"⚠️ TTS initialization failed: {e}")
            voice_results['tts_functional'] = False
            
        voice_results['pyttsx3'] = True
        
    except ImportError as e:
        print(f"❌ pyttsx3: Missing - {e}")
        print("💡 Install: pip install pyttsx3")
        voice_results['pyttsx3'] = False
        voice_results['tts_functional'] = False
    
    return voice_results

def test_ocr_functionality():
    """Test actual OCR functionality"""
    print_header("OCR FUNCTIONALITY TEST")
    
    try:
        # Test solve_from_image import
        from solve_from_image import solve_from_image, IntelligentMathAnalyzer
        print("✅ OCR modules imported successfully")
        
        # Test analyzer initialization
        analyzer = IntelligentMathAnalyzer()
        print("✅ IntelligentMathAnalyzer initialized")
        
        # Test mathematical text processing
        test_texts = [
            "solve x = 2 + 3",
            "what is 15 + 23",
            "calculate 7 * 8"
        ]
        
        print("\n📝 Testing mathematical text processing:")
        for text in test_texts:
            try:
                analysis = analyzer.analyze_mathematical_content(text)
                result = analyzer.solve_intelligently(analysis)
                print(f"   ✅ '{text}' → {result.get('solutions', 'No solution')}")
            except Exception as e:
                print(f"   ❌ '{text}' failed: {e}")
        
        # Test image creation and processing
        print("\n🖼️ Testing image processing:")
        try:
            import cv2
            import numpy as np
            
            # Create a simple test image with text
            img = np.ones((100, 300, 3), dtype=np.uint8) * 255
            cv2.putText(img, "2 + 3 = ?", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Save test image
            test_image_path = "test_math_ocr.png"
            cv2.imwrite(test_image_path, img)
            print(f"   ✅ Test image created: {test_image_path}")
            
            # Test OCR on the image
            result = solve_from_image(test_image_path, 'medium')
            if result[0]:  # OCR text found
                print(f"   ✅ OCR successful: '{result[0]}'")
            else:
                print("   ⚠️ OCR returned no text")
            
            # Clean up
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                
            return True
            
        except Exception as e:
            print(f"   ❌ Image processing failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ OCR functionality test failed: {e}")
        return False

def test_voice_functionality():
    """Test actual voice recognition functionality"""
    print_header("VOICE RECOGNITION FUNCTIONALITY TEST")
    
    try:
        from voice_recognition import VoiceMathRecognizer
        print("✅ Voice recognition modules imported")
        
        # Initialize recognizer
        recognizer = VoiceMathRecognizer()
        print("✅ VoiceMathRecognizer initialized")
        
        # Check if available
        if not recognizer.is_available():
            print("❌ Voice recognition not available")
            print("🔍 Possible issues:")
            print("   • Microphone not connected or not working")
            print("   • PyAudio not properly installed")
            print("   • Missing system audio drivers")
            print("   • Python doesn't have microphone permissions")
            return False
        
        print("✅ Voice recognition is available")
        
        # Test mathematical speech processing
        print("\n🧠 Testing mathematical speech processing:")
        test_phrases = [
            "solve two x plus three equals seven",
            "what is fifteen plus twenty three", 
            "calculate seven times eight"
        ]
        
        for phrase in test_phrases:
            try:
                processed = recognizer._process_mathematical_speech(phrase)
                print(f"   ✅ '{phrase}' → '{processed}'")
            except Exception as e:
                print(f"   ❌ '{phrase}' failed: {e}")
        
        # Test TTS if available
        if recognizer.tts_engine:
            print("\n🔊 Testing text-to-speech:")
            try:
                recognizer.speak_feedback("Voice recognition test successful")
                print("   ✅ TTS working")
            except Exception as e:
                print(f"   ⚠️ TTS failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Voice functionality test failed: {e}")
        return False

def provide_setup_instructions():
    """Provide detailed setup instructions"""
    print_header("SETUP INSTRUCTIONS")
    
    print("🛠️ To fix OCR and Voice issues, follow these steps:\n")
    
    print("📷 FOR OCR ISSUES:")
    print("1. Install basic OCR dependencies:")
    print("   pip install easyocr opencv-python pillow")
    print("\n2. Install Tesseract OCR (optional but recommended):")
    print("   • Windows: Download from https://github.com/tesseract-ocr/tesseract")
    print("   • Add Tesseract to your system PATH")
    print("   • Install Python wrapper: pip install pytesseract")
    print("\n3. Install advanced ML dependencies (optional):")
    print("   pip install torch torchvision scikit-image rembg")
    
    print("\n🎤 FOR VOICE RECOGNITION ISSUES:")
    print("1. Install voice dependencies:")
    print("   pip install speechrecognition pyttsx3")
    print("\n2. Install PyAudio (Windows-specific solution):")
    print("   Method 1: pip install pipwin && pipwin install pyaudio")
    print("   Method 2: pip install pyaudio")
    print("   Method 3: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
    print("\n3. Check microphone permissions:")
    print("   • Windows Settings → Privacy → Microphone")
    print("   • Allow desktop apps to access microphone")
    print("   • Test microphone in Windows Sound settings")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("• If PyAudio fails: Try different installation methods above")
    print("• If microphone not detected: Check Windows Device Manager")
    print("• If OCR fails: Ensure image files are accessible and readable")
    print("• If imports fail: Try uninstalling and reinstalling packages")
    
    print("\n✅ VALIDATION:")
    print("Run this script again after installing dependencies to verify fixes")

def main():
    """Main test runner"""
    print("🚀 SAUNet 4.0 OCR & Voice Recognition Diagnostic Tool")
    print("This script will identify and help fix OCR and voice recording issues")
    
    # Test all dependencies
    basic_results = test_basic_dependencies()
    ocr_results = test_ocr_dependencies()
    voice_results = test_voice_dependencies()
    
    # Test functionality if dependencies are available
    ocr_functional = False
    voice_functional = False
    
    if ocr_results.get('easyocr', False):
        ocr_functional = test_ocr_functionality()
    
    if voice_results.get('speechrecognition', False) and voice_results.get('pyaudio', False):
        voice_functional = test_voice_functionality()
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    print("📊 DEPENDENCY STATUS:")
    print(f"   Basic dependencies: {'✅ Good' if all(basic_results.values()) else '⚠️ Issues found'}")
    print(f"   OCR dependencies: {'✅ Good' if ocr_results.get('easyocr') else '❌ Missing critical'}")
    print(f"   Voice dependencies: {'✅ Good' if voice_results.get('pyaudio') and voice_results.get('speechrecognition') else '❌ Missing critical'}")
    
    print("\n🔧 FUNCTIONALITY STATUS:")
    print(f"   OCR functionality: {'✅ Working' if ocr_functional else '❌ Not working'}")
    print(f"   Voice functionality: {'✅ Working' if voice_functional else '❌ Not working'}")
    
    # Provide instructions if issues found
    if not ocr_functional or not voice_functional:
        provide_setup_instructions()
    else:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("Both OCR and Voice Recognition should work properly in SAUNet 4.0")

if __name__ == "__main__":
    main() 