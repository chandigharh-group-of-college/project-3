# voice_recognition.py - Voice Recognition for Mathematical Queries
# SAUNet 4.0 - AI-Powered Voice-Activated Mathematical Assistant

import threading
import time
import wave
import io
from typing import Optional, Callable, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try importing speech recognition libraries with fallbacks
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    print("✅ SpeechRecognition library loaded successfully")
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️ SpeechRecognition not available. Install with: pip install speechrecognition pyaudio")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    print("✅ PyAudio library loaded successfully")
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️ PyAudio not available. Install with: pip install pyaudio")

# Optional text-to-speech for feedback
try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("✅ Text-to-Speech available")
except ImportError:
    TTS_AVAILABLE = False
    print("ℹ️ Text-to-Speech not available (optional)")


class VoiceMathRecognizer:
    """
    Advanced Voice Recognition system for mathematical queries
    Supports multiple speech recognition engines and intelligent processing
    """
    
    def __init__(self):
        self.is_listening = False
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.callback = None
        self.status_callback = None
        
        # Recognition settings - optimized for Windows compatibility
        self.recognition_timeout = 3.0   # Longer timeout for Windows microphone access
        self.phrase_timeout = 1.5       # More time for microphone detection
        self.energy_threshold = 300     # Lower threshold for better detection
        self.dynamic_energy_threshold = True
        self.pause_threshold = 1.0      # More time to detect speech completion
        self.phrase_threshold = 0.3     # minimum length of phrase
        
        # Mathematical keywords for improved recognition
        self.math_keywords = [
            'differentiate', 'derivative', 'integrate', 'integral', 'solve', 'equation',
            'calculate', 'compute', 'simplify', 'expand', 'factor', 'limit',
            'sine', 'cosine', 'tangent', 'logarithm', 'exponential', 'square root',
            'plus', 'minus', 'times', 'divided by', 'equals', 'x squared', 'x cubed'
        ]
        
        # Initialize components
        self._initialize_recognition()
        self._initialize_tts()
    
    def _initialize_recognition(self):
        """Initialize speech recognition components with Windows compatibility"""
        if not SPEECH_RECOGNITION_AVAILABLE or not PYAUDIO_AVAILABLE:
            print("⚠️ Voice recognition not available - missing dependencies")
            print("💡 Install with: pip install speechrecognition pyaudio")
            print("💡 For Windows PyAudio: pip install pipwin && pipwin install pyaudio")
            return
        
        try:
            self.recognizer = sr.Recognizer()
            
            # Try to initialize microphone with better error handling
            print("🎤 Initializing microphone...")
            try:
                # List available microphones
                mic_list = sr.Microphone.list_microphone_names()
                print(f"📋 Available microphones: {len(mic_list)} devices found")
                
                # Use default microphone
                self.microphone = sr.Microphone()
                print("✅ Default microphone initialized")
                
            except Exception as mic_error:
                print(f"⚠️ Microphone initialization failed: {mic_error}")
                print("💡 Check microphone permissions and connections")
                self.microphone = None
                return
            
            # Calibrate microphone with extended error handling
            print("🎤 Calibrating microphone for ambient noise...")
            try:
                with self.microphone as source:
                    # Longer calibration for Windows microphone detection
                    print("🔧 Adjusting for ambient noise (this may take 3-5 seconds)...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=3)
                    
                    # Get the detected energy threshold
                    detected_energy = self.recognizer.energy_threshold
                    print(f"🔊 Detected ambient energy: {detected_energy}")
                    
                    # Set threshold appropriately for detected environment
                    if detected_energy < 100:
                        self.energy_threshold = 300  # Quiet environment
                    elif detected_energy < 500:
                        self.energy_threshold = detected_energy + 50  # Normal environment
                    else:
                        self.energy_threshold = detected_energy + 100  # Noisy environment
                        
            except Exception as calibration_error:
                print(f"⚠️ Microphone calibration failed: {calibration_error}")
                print("🔧 Using default energy threshold")
                self.energy_threshold = 300  # Safe default
            
            # Set recognition parameters optimized for Windows
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.dynamic_energy_threshold = self.dynamic_energy_threshold
            self.recognizer.pause_threshold = self.pause_threshold
            self.recognizer.phrase_threshold = self.phrase_threshold
            
            print(f"🎯 Energy threshold set to: {self.energy_threshold}")
            print(f"⚡ Pause threshold: {self.pause_threshold}s")
            print(f"📏 Phrase threshold: {self.phrase_threshold}s")
            
            print("✅ Voice recognition initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing voice recognition: {e}")
            print("🔍 Troubleshooting tips:")
            print("   • Check if microphone is connected and working")
            print("   • Ensure Python has microphone permissions")
            print("   • Try: pip uninstall pyaudio && pip install pyaudio")
            print("   • On Windows: pip install pipwin && pipwin install pyaudio")
            self.recognizer = None
            self.microphone = None
    
    def _initialize_tts(self):
        """Initialize text-to-speech for feedback"""
        if not TTS_AVAILABLE:
            return
        
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice for mathematical assistant
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.tts_engine.setProperty('rate', 180)  # Speaking rate
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            
            print("✅ Text-to-Speech initialized successfully")
            
        except Exception as e:
            print(f"⚠️ TTS initialization error: {e}")
            self.tts_engine = None
    
    def is_available(self) -> bool:
        """Check if voice recognition is available"""
        return (SPEECH_RECOGNITION_AVAILABLE and 
                PYAUDIO_AVAILABLE and 
                self.recognizer is not None and 
                self.microphone is not None)
    
    def set_callback(self, callback: Callable[[str], None]):
        """Set callback function for recognized text"""
        self.callback = callback
    
    def set_status_callback(self, status_callback: Callable[[str], None]):
        """Set callback function for status updates"""
        self.status_callback = status_callback
    
    def _update_status(self, status: str):
        """Update status via callback"""
        if self.status_callback:
            self.status_callback(status)
        print(f"🎤 [VOICE] {status}")
    
    def speak_feedback(self, text: str):
        """Provide voice feedback to the user"""
        if not self.tts_engine:
            return
        
        try:
            # Run TTS in a separate thread to avoid blocking
            def tts_worker():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            tts_thread.start()
            
        except Exception as e:
            print(f"⚠️ TTS error: {e}")
    
    def start_listening(self):
        """Start continuous listening for voice input"""
        if not self.is_available():
            self._update_status("Voice recognition not available")
            return False
        
        if self.is_listening:
            self._update_status("Already listening...")
            return True
        
        self.is_listening = True
        
        # Start listening in a separate thread
        listen_thread = threading.Thread(target=self._listen_worker, daemon=True)
        listen_thread.start()
        
        self._update_status("🎤 Listening... Speak your mathematical question")
        self.speak_feedback("Ready for your mathematical question")
        
        return True
    
    def stop_listening(self):
        """Stop listening for voice input"""
        if self.is_listening:
            self.is_listening = False
            self._update_status("Stopped listening")
            self.speak_feedback("Voice recognition stopped")
    
    def _listen_worker(self):
        """Worker thread for continuous listening - ChatGPT-like responsiveness"""
        consecutive_timeouts = 0
        max_timeouts = 20  # Stop after too many timeouts
        
        self._update_status("🎤 Ready to listen - speak naturally!")
        
        while self.is_listening and consecutive_timeouts < max_timeouts:
            try:
                # Continuous listening with voice activity detection
                with self.microphone as source:
                    # Listen for speech with shorter timeout for responsiveness
                    try:
                        audio = self.recognizer.listen(
                            source, 
                            timeout=self.recognition_timeout,
                            phrase_time_limit=15  # Max 15 seconds per phrase
                        )
                        consecutive_timeouts = 0  # Reset timeout counter
                        
                    except sr.WaitTimeoutError:
                        consecutive_timeouts += 1
                        if consecutive_timeouts % 5 == 0:  # Every 5 timeouts
                            self._update_status("🎧 Still listening... speak when ready")
                        continue
                
                if not self.is_listening:
                    break
                
                self._update_status("🎯 Speech detected! Processing...")
                
                # Try multiple recognition engines
                recognized_text = self._recognize_audio(audio)
                
                if recognized_text and len(recognized_text.strip()) > 2:
                    # Filter out very short or meaningless responses
                    if not self._is_meaningless_speech(recognized_text):
                        self._update_status(f"✅ Recognized: {recognized_text}")
                        
                        # Process the recognized text
                        processed_text = self._process_mathematical_speech(recognized_text)
                        
                        # Send to callback
                        if self.callback:
                            self.callback(processed_text)
                        
                        # Provide feedback
                        self.speak_feedback(f"Processing: {processed_text}")
                        
                        # Stop listening after successful recognition
                        self.is_listening = False
                        self._update_status("✅ Speech processed successfully")
                        return
                    else:
                        self._update_status("🤔 Detected noise or unclear speech, continuing to listen...")
                else:
                    self._update_status("🎧 No clear speech detected, listening again...")
                    time.sleep(0.2)  # Brief pause before trying again
            
            except sr.RequestError as e:
                self._update_status(f"❌ Recognition service error: {e}")
                time.sleep(2)  # Wait before retrying
                continue
            
            except Exception as e:
                self._update_status(f"❌ Unexpected error: {e}")
                time.sleep(1)
                continue
        
        if consecutive_timeouts >= max_timeouts:
            self._update_status("⏰ Listening session ended due to inactivity")
        
        self.is_listening = False
    
    def _is_meaningless_speech(self, text: str) -> bool:
        """Check if the recognized speech is meaningless noise or system sounds"""
        if not text or len(text.strip()) < 3:
            return True
        
        # Convert to lowercase for checking
        text_lower = text.lower().strip()
        
        # Common noise patterns or system responses
        meaningless_patterns = [
            "ready for your mathematical question",
            "i heard",
            "processing",
            "the result is",
            "uh", "um", "ah", "er", "hmm",
            "noise", "sound", "background",
            "system", "error", "test",
            # Single letters or very short words
        ]
        
        # Check if it's just noise
        if any(pattern in text_lower for pattern in meaningless_patterns):
            return True
        
        # Check if it's just single characters or very short
        if len(text_lower) < 4 and not any(char.isdigit() for char in text_lower):
            return True
        
        # Check if it contains actual mathematical content
        math_indicators = [
            'solve', 'calculate', 'compute', 'find', 'what', 'how',
            'plus', 'minus', 'times', 'divide', 'equals', 'equal',
            'x', 'y', 'derivative', 'integral', 'square', 'root',
            'sin', 'cos', 'tan', 'log', 'exp', 'pi', 'e',
            '+', '-', '*', '/', '=', '²', '³', '^'
        ]
        
        # If it contains mathematical indicators, it's probably meaningful
        if any(indicator in text_lower for indicator in math_indicators):
            return False
        
        # If it contains numbers, it might be mathematical
        if any(char.isdigit() for char in text):
            return False
        
        # Default: if it's longer than 8 characters and doesn't match noise patterns, consider it meaningful
        return len(text_lower) < 8
    
    def _recognize_audio(self, audio) -> Optional[str]:
        """Try multiple speech recognition engines"""
        recognition_methods = [
            ("Google", self._recognize_google),
            ("Google Cloud", self._recognize_google_cloud),
            ("Whisper", self._recognize_whisper),
            ("Sphinx", self._recognize_sphinx)
        ]
        
        for method_name, method in recognition_methods:
            try:
                self._update_status(f"🔍 Trying {method_name} recognition...")
                result = method(audio)
                if result:
                    self._update_status(f"✅ {method_name} recognition successful")
                    return result
            except Exception as e:
                print(f"⚠️ {method_name} recognition failed: {e}")
                continue
        
        return None
    
    def _recognize_google(self, audio) -> Optional[str]:
        """Google Speech Recognition (free)"""
        try:
            result = self.recognizer.recognize_google(audio, language='en-US')
            return result.lower() if result else None
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None
    
    def _recognize_google_cloud(self, audio) -> Optional[str]:
        """Google Cloud Speech Recognition (requires API key)"""
        try:
            # This requires Google Cloud credentials
            result = self.recognizer.recognize_google_cloud(audio, language='en-US')
            return result.lower() if result else None
        except:
            return None
    
    def _recognize_whisper(self, audio) -> Optional[str]:
        """OpenAI Whisper recognition (if available)"""
        try:
            result = self.recognizer.recognize_whisper(audio, model="base", language="english")
            return result.lower() if result else None
        except:
            return None
    
    def _recognize_sphinx(self, audio) -> Optional[str]:
        """CMU Sphinx recognition (offline)"""
        try:
            result = self.recognizer.recognize_sphinx(audio)
            return result.lower() if result else None
        except:
            return None
    
    def _process_mathematical_speech(self, text: str) -> str:
        """Process and clean mathematical speech for better understanding"""
        if not text:
            return ""
        
        # Convert common spoken mathematical phrases to written form
        math_conversions = {
            # Numbers and operations
            'plus': '+',
            'add': '+',
            'added to': '+',
            'minus': '-',
            'subtract': '-',
            'take away': '-',
            'times': '*',
            'multiply': '*',
            'multiplied by': '*',
            'divided by': '/',
            'over': '/',
            'equals': '=',
            'is equal to': '=',
            
            # Powers and roots
            'squared': '²',
            'cubed': '³',
            'to the power of': '^',
            'raised to': '^',
            'square root of': 'sqrt(',
            'square root': 'sqrt(',
            
            # Trigonometric functions
            'sine': 'sin',
            'sin of': 'sin(',
            'cosine': 'cos',
            'cos of': 'cos(',
            'tangent': 'tan',
            'tan of': 'tan(',
            
            # Calculus operations
            'differentiate': 'differentiate',
            'find the derivative of': 'differentiate',
            'derive': 'differentiate',
            'integrate': 'integrate',
            'find the integral of': 'integrate',
            
            # Equation solving
            'solve for': 'solve',
            'find x when': 'solve',
            'what is x if': 'solve',
            
            # Common mathematical expressions
            'x squared': 'x²',
            'x cubed': 'x³',
            'two x': '2x',
            'three x': '3x',
            'pi': 'π',
            'e': 'e',
            
            # Parentheses (spoken)
            'open parenthesis': '(',
            'close parenthesis': ')',
            'left bracket': '(',
            'right bracket': ')',
        }
        
        # Apply conversions
        processed_text = text
        for spoken, written in math_conversions.items():
            processed_text = processed_text.replace(spoken, written)
        
        # Handle number words
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20'
        }
        
        for word, number in number_words.items():
            processed_text = processed_text.replace(f' {word} ', f' {number} ')
            processed_text = processed_text.replace(f'{word} ', f'{number} ')
            processed_text = processed_text.replace(f' {word}', f' {number}')
        
        # Clean up spacing
        processed_text = ' '.join(processed_text.split())
        
        return processed_text.strip()
    
    def recognize_single_phrase(self) -> Optional[str]:
        """Recognize a single phrase (blocking call) - ChatGPT-like responsiveness"""
        if not self.is_available():
            return None
        
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                attempt += 1
                self._update_status(f"🎤 Listening for your question... (attempt {attempt}/{max_attempts})")
                
                with self.microphone as source:
                    # Brief recalibration for current conditions
                    if attempt == 1:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    
                    # Listen for a single phrase with optimized settings
                    audio = self.recognizer.listen(
                        source, 
                        timeout=8,      # Longer timeout for single phrase
                        phrase_time_limit=20  # Allow longer phrases
                    )
                
                self._update_status("🎯 Speech detected! Processing...")
                
                # Recognize the audio
                recognized_text = self._recognize_audio(audio)
                
                if recognized_text and len(recognized_text.strip()) > 2:
                    if not self._is_meaningless_speech(recognized_text):
                        processed_text = self._process_mathematical_speech(recognized_text)
                        self._update_status(f"✅ Recognized: {processed_text}")
                        return processed_text
                    else:
                        self._update_status("🤔 Detected noise, please try again...")
                else:
                    self._update_status("❓ Could not understand speech clearly...")
                
                if attempt < max_attempts:
                    self._update_status(f"🔄 Trying again... ({max_attempts - attempt} attempts left)")
                    time.sleep(1)
            
            except sr.WaitTimeoutError:
                if attempt < max_attempts:
                    self._update_status(f"⏰ No speech detected, trying again... ({max_attempts - attempt} attempts left)")
                    time.sleep(0.5)
                else:
                    self._update_status("⏰ No speech detected within timeout")
            
            except Exception as e:
                self._update_status(f"❌ Recognition error: {e}")
                if attempt < max_attempts:
                    time.sleep(1)
        
        self._update_status("❌ Could not recognize speech after multiple attempts")
        return None


class VoiceControlledCalculator:
    """
    Voice-controlled interface for the mathematical calculator
    Integrates voice recognition with mathematical NLU
    """
    
    def __init__(self, calculator_ui=None):
        self.calculator_ui = calculator_ui
        self.voice_recognizer = VoiceMathRecognizer()
        self.is_voice_mode = False
        
        # Try to initialize enhanced voice recognition
        try:
            from enhanced_voice import EnhancedVoiceRecognizer
            self.enhanced_voice = EnhancedVoiceRecognizer()
            self.use_enhanced = True
            print("🎤 Enhanced multi-language voice recognition initialized")
        except ImportError:
            self.enhanced_voice = None
            self.use_enhanced = False
            print("🎤 Basic voice recognition initialized")
        
        # Set up callbacks
        self.voice_recognizer.set_callback(self._handle_voice_input)
        if calculator_ui:
            self.voice_recognizer.set_status_callback(self._update_ui_status)
    
    def _handle_voice_input(self, text: str):
        """Handle recognized voice input - ChatGPT-like processing"""
        if not self.calculator_ui or not text or len(text.strip()) < 2:
            return
        
        # Clean the input text
        text = text.strip()
        
        # Determine the best processing method based on content
        if self._is_natural_language_query(text):
            # Use natural language processing for complex queries
            if hasattr(self.calculator_ui, 'nl_input') and hasattr(self.calculator_ui, 'process_nl_query'):
                self.calculator_ui.nl_input.setPlainText(text)
                # Switch to NL tab if not already there
                if hasattr(self.calculator_ui, 'tabs'):
                    self.calculator_ui.tabs.setCurrentIndex(2)  # Natural Language tab
                # Process the query
                self.calculator_ui.process_nl_query()
            else:
                # Fallback to main input
                self.calculator_ui.input_field.setText(text)
                self.calculator_ui.evaluate_expression()
        else:
            # Use direct calculation for simple expressions
            self.calculator_ui.input_field.setText(text)
            self.calculator_ui.evaluate_expression()
    
    def _is_natural_language_query(self, text: str) -> bool:
        """Determine if the input should be processed as natural language"""
        text_lower = text.lower()
        
        # Natural language indicators
        nl_indicators = [
            'solve', 'find', 'calculate', 'compute', 'what is', 'how much',
            'differentiate', 'derivative', 'integrate', 'integral', 'limit',
            'simplify', 'expand', 'factor', 'evaluate', 'determine'
        ]
        
        # Check for natural language patterns
        return any(indicator in text_lower for indicator in nl_indicators)
    
    def _handle_enhanced_voice_input(self, text: str):
        """Handle enhanced voice input with multi-language support"""
        if not self.calculator_ui or not text or len(text.strip()) < 2:
            return
        
        # Clean the input text
        text = text.strip()
        
        # Enhanced voice processing includes automatic language detection
        # and mathematical expression conversion
        if self.enhanced_voice:
            # Process with enhanced voice recognition
            math_expression = self.enhanced_voice.process_mathematical_command(text)
            if math_expression:
                text = math_expression
        
        # Use the same routing logic as basic voice input
        self._handle_voice_input(text)
    
    def _update_ui_status(self, status: str):
        """Update the UI with voice recognition status"""
        if self.calculator_ui and hasattr(self.calculator_ui, 'statusBar'):
            status_bar = self.calculator_ui.statusBar()
            if status_bar:
                status_bar.showMessage(f"🎤 {status}")
    
    def start_voice_recognition(self):
        """Start voice recognition with enhanced multi-language support"""
        if self.use_enhanced and self.enhanced_voice:
            try:
                # Use enhanced voice recognition with multi-language support
                self.enhanced_voice.start_listening(self._handle_enhanced_voice_input)
                self.is_voice_mode = True
                return True
            except Exception as e:
                print(f"Enhanced voice recognition failed: {e}, falling back to basic")
                self.use_enhanced = False
        
        # Fall back to basic voice recognition
        if not self.voice_recognizer.is_available():
            return False
        
        success = self.voice_recognizer.start_listening()
        if success:
            self.is_voice_mode = True
        return success
    
    def stop_voice_recognition(self):
        """Stop voice recognition"""
        if self.use_enhanced and self.enhanced_voice:
            self.enhanced_voice.stop_listening()
        else:
            self.voice_recognizer.stop_listening()
        self.is_voice_mode = False
    
    def recognize_single_command(self):
        """Recognize a single voice command"""
        return self.voice_recognizer.recognize_single_phrase()
    
    def speak_result(self, text: str):
        """Speak a calculation result"""
        self.voice_recognizer.speak_feedback(text)


# Test and demonstration
def test_voice_recognition():
    """Test the voice recognition system"""
    print("🧪 Testing Voice Recognition System")
    print("=" * 50)
    
    recognizer = VoiceMathRecognizer()
    
    if not recognizer.is_available():
        print("❌ Voice recognition not available")
        print("💡 Install dependencies: pip install speechrecognition pyaudio")
        return
    
    print("✅ Voice recognition system ready")
    print("🎤 Say a mathematical expression (e.g., 'solve two x plus three equals seven')")
    
    result = recognizer.recognize_single_phrase()
    
    if result:
        print(f"📝 Recognized text: {result}")
        print("✅ Voice recognition test successful!")
    else:
        print("❌ No speech recognized")


if __name__ == "__main__":
    test_voice_recognition() 