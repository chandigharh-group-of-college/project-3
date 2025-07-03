# enhanced_voice.py - Google Assistant-like Voice Recognition System
# Support for Hindi, Hinglish, English and multiple recognition engines

import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import os
import json
from typing import Dict, List, Optional, Callable
import re

# Advanced speech recognition libraries
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ OpenAI Whisper available")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ OpenAI Whisper not available. Install with: pip install openai-whisper")

try:
    from google.cloud import speech
    GOOGLE_CLOUD_SPEECH_AVAILABLE = True
    print("✅ Google Cloud Speech available")
except ImportError:
    GOOGLE_CLOUD_SPEECH_AVAILABLE = False
    print("⚠️ Google Cloud Speech not available")

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SPEECH_AVAILABLE = True
    print("✅ Azure Speech available")
except ImportError:
    AZURE_SPEECH_AVAILABLE = False
    print("⚠️ Azure Speech not available")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️ PyAudio not available for streaming")


class EnhancedVoiceRecognizer:
    """Google Assistant-like voice recognition with multi-language support"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts_engine = None
        self.listening = False
        self.callback = None
        
        # Language configuration
        self.supported_languages = {
            'english': {
                'code': 'en-US',
                'google_code': 'en-US',
                'azure_code': 'en-US',
                'whisper_code': 'en'
            },
            'hindi': {
                'code': 'hi-IN',
                'google_code': 'hi-IN',
                'azure_code': 'hi-IN',
                'whisper_code': 'hi'
            },
            'hinglish': {
                'code': 'en-IN',  # English India for Hinglish
                'google_code': 'en-IN',
                'azure_code': 'en-IN',
                'whisper_code': 'en'
            }
        }
        
        self.current_language = 'english'
        self.auto_detect_language = True
        
        # Recognition engines
        self.engines = {}
        self.initialize_engines()
        
        # Voice patterns for mathematical commands
        self.math_patterns = self._load_math_patterns()
        
        print("🎤 Enhanced Voice Recognition System initialized")
    
    def initialize_engines(self):
        """Initialize all available speech recognition engines"""
        print("🚀 Initializing voice recognition engines...")
        
        # Initialize microphone
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                print("🎧 Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("  ✅ Microphone calibrated")
        except Exception as e:
            print(f"  ❌ Microphone error: {e}")
        
        # Initialize TTS
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Slower for better understanding
            print("  ✅ Text-to-Speech initialized")
        except Exception as e:
            print(f"  ❌ TTS error: {e}")
        
        # Initialize Whisper
        if WHISPER_AVAILABLE:
            try:
                self.engines['whisper'] = whisper.load_model("base")
                print("  ✅ Whisper model loaded")
            except Exception as e:
                print(f"  ❌ Whisper error: {e}")
        
        # Initialize Google Cloud Speech
        if GOOGLE_CLOUD_SPEECH_AVAILABLE and os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            try:
                self.engines['google_cloud'] = speech.SpeechClient()
                print("  ✅ Google Cloud Speech initialized")
            except Exception as e:
                print(f"  ❌ Google Cloud Speech error: {e}")
        
        # Initialize Azure Speech
        if AZURE_SPEECH_AVAILABLE and os.getenv('AZURE_SPEECH_KEY'):
            try:
                speech_key = os.getenv('AZURE_SPEECH_KEY')
                service_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
                speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
                self.engines['azure'] = speech_config
                print("  ✅ Azure Speech initialized")
            except Exception as e:
                print(f"  ❌ Azure Speech error: {e}")
        
        print(f"🎯 Voice engines ready: {list(self.engines.keys())}")
    
    def _load_math_patterns(self) -> Dict:
        """Load mathematical voice command patterns for multi-language support"""
        return {
            'english': {
                # Basic operations
                r'\b(what is|calculate|compute|solve)\s+(.+)': 'calculation',
                r'\b(\d+)\s+(plus|add|added to)\s+(\d+)': 'addition',
                r'\b(\d+)\s+(minus|subtract|take away)\s+(\d+)': 'subtraction',
                r'\b(\d+)\s+(times|multiply|multiplied by)\s+(\d+)': 'multiplication',
                r'\b(\d+)\s+(divided by|divide)\s+(\d+)': 'division',
                
                # Advanced operations
                r'\b(square root of|sqrt of)\s+(\d+)': 'sqrt',
                r'\b(\d+)\s+(squared|to the power of 2)': 'square',
                r'\b(\d+)\s+(cubed|to the power of 3)': 'cube',
                r'\b(sin|sine|cos|cosine|tan|tangent)\s+(of\s+)?(\d+)': 'trigonometry',
                
                # Equations
                r'\b(solve|find x|what is x)\s+(.+)': 'equation',
                r'\b(derivative|differentiate)\s+(.+)': 'derivative',
                r'\b(integrate|integral)\s+(.+)': 'integral'
            },
            'hindi': {
                # Hindi mathematical terms
                r'\b(क्या है|गणना करें|हल करें)\s+(.+)': 'calculation',
                r'\b(\d+)\s+(जमा|प्लस|और)\s+(\d+)': 'addition',
                r'\b(\d+)\s+(घटा|माइनस|कम)\s+(\d+)': 'subtraction',
                r'\b(\d+)\s+(गुणा|टाइम्स)\s+(\d+)': 'multiplication',
                r'\b(\d+)\s+(भाग|डिवाइड)\s+(\d+)': 'division',
                r'\b(वर्गमूल|स्क्वेयर रूट)\s+(\d+)': 'sqrt'
            },
            'hinglish': {
                # Mix of Hindi and English
                r'\b(kya hai|calculate karo|solve karo)\s+(.+)': 'calculation',
                r'\b(\d+)\s+(plus|joda|add karo)\s+(\d+)': 'addition',
                r'\b(\d+)\s+(minus|ghata|subtract karo)\s+(\d+)': 'subtraction',
                r'\b(\d+)\s+(times|guna|multiply karo)\s+(\d+)': 'multiplication',
                r'\b(\d+)\s+(divide|bhag|divide karo)\s+(\d+)': 'division'
            }
        }
    
    def set_language(self, language: str):
        """Set the primary language for recognition"""
        if language in self.supported_languages:
            self.current_language = language
            print(f"🌐 Language set to: {language}")
        else:
            print(f"❌ Unsupported language: {language}")
    
    def recognize_with_multiple_engines(self, audio_data) -> Dict:
        """Recognize speech using multiple engines and return best result"""
        results = {}
        
        # Try Google's built-in recognizer (free)
        try:
            for lang_name, lang_info in self.supported_languages.items():
                try:
                    text = self.recognizer.recognize_google(
                        audio_data, 
                        language=lang_info['google_code']
                    )
                    if text.strip():
                        results[f'google_{lang_name}'] = {
                            'text': text.lower().strip(),
                            'confidence': 0.8,
                            'language': lang_name,
                            'engine': 'google'
                        }
                        break
                except sr.UnknownValueError:
                    continue
                except Exception as e:
                    print(f"Google ({lang_name}) error: {e}")
        except Exception as e:
            print(f"Google recognition error: {e}")
        
        # Try Whisper (if available)
        if 'whisper' in self.engines:
            try:
                # Save audio to temporary file for Whisper
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_data.get_wav_data())
                    tmp_path = tmp_file.name
                
                # Recognize with Whisper
                result = self.engines['whisper'].transcribe(
                    tmp_path, 
                    language=None,  # Auto-detect
                    fp16=False
                )
                
                text = result['text'].strip()
                detected_lang = result.get('language', 'en')
                
                if text:
                    results['whisper'] = {
                        'text': text.lower(),
                        'confidence': 0.9,  # Whisper is usually very good
                        'language': detected_lang,
                        'engine': 'whisper'
                    }
                
                # Clean up
                os.unlink(tmp_path)
                
            except Exception as e:
                print(f"Whisper error: {e}")
        
        # Try Google Cloud Speech (if available)
        if 'google_cloud' in self.engines:
            try:
                client = self.engines['google_cloud']
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=self.supported_languages[self.current_language]['google_code'],
                    alternative_language_codes=[
                        self.supported_languages['hindi']['google_code'],
                        self.supported_languages['hinglish']['google_code']
                    ]
                )
                
                audio = speech.RecognitionAudio(content=audio_data.get_wav_data())
                response = client.recognize(config=config, audio=audio)
                
                if response.results:
                    result = response.results[0]
                    text = result.alternatives[0].transcript
                    confidence = result.alternatives[0].confidence
                    
                    results['google_cloud'] = {
                        'text': text.lower().strip(),
                        'confidence': confidence,
                        'language': 'auto',
                        'engine': 'google_cloud'
                    }
                    
            except Exception as e:
                print(f"Google Cloud Speech error: {e}")
        
        return results
    
    def get_best_recognition_result(self, results: Dict) -> Optional[Dict]:
        """Select the best recognition result based on confidence and content"""
        if not results:
            return None
        
        # Score each result
        scored_results = []
        for engine, result in results.items():
            text = result['text']
            confidence = result['confidence']
            
            # Base score from confidence
            score = confidence
            
            # Bonus for mathematical content
            if self._contains_mathematical_content(text):
                score += 0.1
            
            # Bonus for longer text (more complete)
            if len(text.split()) > 2:
                score += 0.05
            
            # Engine-specific bonuses
            if 'whisper' in engine:
                score += 0.1  # Whisper is excellent for multi-language
            elif 'google_cloud' in engine:
                score += 0.08  # Google Cloud is very good
            
            scored_results.append((score, engine, result))
        
        # Return the best result
        if scored_results:
            scored_results.sort(reverse=True)
            return scored_results[0][2]
        
        return None
    
    def _contains_mathematical_content(self, text: str) -> bool:
        """Check if text contains mathematical terms"""
        math_keywords = [
            # English
            'calculate', 'solve', 'plus', 'minus', 'times', 'divide', 'square', 'root',
            'sin', 'cos', 'tan', 'derivative', 'integral', 'equation',
            # Hindi
            'गणना', 'हल', 'जमा', 'घटा', 'गुणा', 'भाग', 'वर्गमूल',
            # Hinglish
            'kya hai', 'calculate karo', 'solve karo', 'joda', 'ghata', 'guna'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in math_keywords) or \
               bool(re.search(r'\d+', text))  # Contains numbers
    
    def process_mathematical_command(self, text: str) -> Optional[str]:
        """Process and convert voice command to mathematical expression"""
        text = text.lower().strip()
        
        # Try patterns for current language and auto-detect
        for lang_name, patterns in self.math_patterns.items():
            for pattern, command_type in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return self._convert_to_math_expression(text, command_type, match)
        
        # If no specific pattern matches, try general conversion
        return self._general_math_conversion(text)
    
    def _convert_to_math_expression(self, text: str, command_type: str, match) -> str:
        """Convert matched pattern to mathematical expression"""
        if command_type == 'addition':
            groups = match.groups()
            if len(groups) >= 3:
                return f"{groups[0]} + {groups[2]}"
        
        elif command_type == 'subtraction':
            groups = match.groups()
            if len(groups) >= 3:
                return f"{groups[0]} - {groups[2]}"
        
        elif command_type == 'multiplication':
            groups = match.groups()
            if len(groups) >= 3:
                return f"{groups[0]} * {groups[2]}"
        
        elif command_type == 'division':
            groups = match.groups()
            if len(groups) >= 3:
                return f"{groups[0]} / {groups[2]}"
        
        elif command_type == 'sqrt':
            groups = match.groups()
            if len(groups) >= 2:
                return f"sqrt({groups[1]})"
        
        elif command_type == 'square':
            groups = match.groups()
            if len(groups) >= 1:
                return f"{groups[0]}**2"
        
        elif command_type == 'cube':
            groups = match.groups()
            if len(groups) >= 1:
                return f"{groups[0]}**3"
        
        elif command_type in ['calculation', 'equation']:
            # Extract the mathematical part
            groups = match.groups()
            if len(groups) >= 2:
                return groups[1]
        
        return text  # Fallback to original text
    
    def _general_math_conversion(self, text: str) -> str:
        """General mathematical text conversion"""
        # Replace common spoken math with symbols
        conversions = {
            # English
            'plus': '+', 'add': '+', 'added to': '+',
            'minus': '-', 'subtract': '-', 'take away': '-',
            'times': '*', 'multiply': '*', 'multiplied by': '*',
            'divided by': '/', 'divide': '/',
            'equals': '=', 'is equal to': '=',
            'squared': '**2', 'cubed': '**3',
            'to the power of': '**',
            
            # Hindi transliterations
            'joda': '+', 'ghata': '-', 'guna': '*', 'bhag': '/',
            
            # Numbers in words (basic)
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10'
        }
        
        result = text
        for word, symbol in conversions.items():
            result = re.sub(r'\b' + re.escape(word) + r'\b', symbol, result, flags=re.IGNORECASE)
        
        return result.strip()
    
    def speak(self, text: str):
        """Text-to-speech with language support"""
        if self.tts_engine:
            try:
                # Set voice based on language
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Try to find appropriate voice for language
                    for voice in voices:
                        if self.current_language == 'hindi' and 'hindi' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                        elif self.current_language == 'english' and 'english' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                
                # Speak in a separate thread to avoid blocking
                def speak_text():
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                
                thread = threading.Thread(target=speak_text)
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                print(f"TTS error: {e}")
    
    def start_listening(self, callback: Callable[[str], None]):
        """Start continuous listening with improved responsiveness"""
        self.callback = callback
        self.listening = True
        
        def listen_continuously():
            print("🎤 Starting enhanced listening...")
            self.speak("Voice recognition ready. You can speak in English, Hindi, or Hinglish.")
            
            while self.listening:
                try:
                    with self.microphone as source:
                        # Quick adjustment for ambient noise
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        
                        # More responsive listening settings
                        audio = self.recognizer.listen(
                            source,
                            timeout=1,      # Wait 1 second for speech to start
                            phrase_time_limit=10  # Allow up to 10 seconds of speech
                        )
                    
                    print("🎯 Processing speech...")
                    
                    # Use multiple engines for recognition
                    results = self.recognize_with_multiple_engines(audio)
                    best_result = self.get_best_recognition_result(results)
                    
                    if best_result:
                        recognized_text = best_result['text']
                        confidence = best_result['confidence']
                        language = best_result['language']
                        engine = best_result['engine']
                        
                        print(f"✅ Recognized ({engine}, {language}, {confidence:.2f}): {recognized_text}")
                        
                        # Process mathematical command
                        math_expression = self.process_mathematical_command(recognized_text)
                        
                        if math_expression and math_expression != recognized_text:
                            print(f"🧮 Converted to: {math_expression}")
                            if self.callback:
                                self.callback(math_expression)
                        else:
                            if self.callback:
                                self.callback(recognized_text)
                    else:
                        print("❓ Could not understand speech clearly")
                
                except sr.WaitTimeoutError:
                    # No speech detected, continue listening
                    continue
                except sr.UnknownValueError:
                    print("❓ Could not understand speech")
                except Exception as e:
                    print(f"❌ Listening error: {e}")
                    time.sleep(1)  # Brief pause before retrying
        
        # Start listening in background thread
        listen_thread = threading.Thread(target=listen_continuously)
        listen_thread.daemon = True
        listen_thread.start()
    
    def stop_listening(self):
        """Stop continuous listening"""
        self.listening = False
        print("🔇 Voice recognition stopped")
        self.speak("Voice recognition stopped")


# Test function
if __name__ == "__main__":
    def test_callback(text):
        print(f"📝 Received: {text}")
    
    voice = EnhancedVoiceRecognizer()
    
    print("🎤 Enhanced Voice Recognition Test")
    print("Supported commands:")
    print("- English: 'What is two plus three', 'Calculate five times seven'")
    print("- Hindi: 'दो जमा तीन क्या है', 'पांच गुणा सात'")
    print("- Hinglish: 'Do plus teen kya hai', 'Calculate karo five times seven'")
    
    voice.start_listening(test_callback)
    
    try:
        input("Press Enter to stop...")
    except KeyboardInterrupt:
        pass
    
    voice.stop_listening() 