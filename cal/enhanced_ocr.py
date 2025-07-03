# enhanced_ocr.py - ChatGPT-like Powerful OCR System
# Support for multiple OCR engines and mathematical text recognition

import cv2
import numpy as np
import base64
import requests
import os
from typing import List, Dict, Tuple, Optional
import re

# Core OCR libraries
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

# Advanced OCR APIs
try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
    from msrest.authentication import CognitiveServicesCredentials
    AZURE_VISION_AVAILABLE = True
except ImportError:
    AZURE_VISION_AVAILABLE = False

# Transformer-based OCR
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

print("🔍 OCR Engine Availability:")
print(f"  ✅ EasyOCR: {EASYOCR_AVAILABLE}")
print(f"  ✅ Tesseract: {TESSERACT_AVAILABLE}")
print(f"  ✅ PaddleOCR: {PADDLEOCR_AVAILABLE}")
print(f"  ✅ Google Cloud Vision: {GOOGLE_VISION_AVAILABLE}")
print(f"  ✅ Azure Vision: {AZURE_VISION_AVAILABLE}")
print(f"  ✅ TrOCR (Transformer): {TROCR_AVAILABLE}")


class PowerfulOCREngine:
    """ChatGPT-like powerful OCR engine with multiple backends"""
    
    def __init__(self):
        self.engines = {}
        self.initialize_engines()
        
    def initialize_engines(self):
        """Initialize all available OCR engines"""
        print("🚀 Initializing Enhanced OCR Engines...")
        
        # Initialize EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.engines['easyocr'] = easyocr.Reader(['en', 'hi'], gpu=False)
                print("  ✅ EasyOCR (English + Hindi) initialized")
            except Exception as e:
                print(f"  ❌ EasyOCR failed: {e}")
        
        # Initialize PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                self.engines['paddleocr'] = paddleocr.PaddleOCR(
                    use_angle_cls=True, 
                    lang='en',  # Can be changed to 'hi' for Hindi
                    use_gpu=False,
                    show_log=False
                )
                print("  ✅ PaddleOCR initialized")
            except Exception as e:
                print(f"  ❌ PaddleOCR failed: {e}")
        
        # Initialize Google Cloud Vision
        if GOOGLE_VISION_AVAILABLE:
            try:
                # Check if credentials are available
                if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                    self.engines['google_vision'] = vision.ImageAnnotatorClient()
                    print("  ✅ Google Cloud Vision initialized")
                else:
                    print("  ⚠️ Google Cloud Vision: No credentials found")
            except Exception as e:
                print(f"  ❌ Google Cloud Vision failed: {e}")
        
        # Initialize TrOCR (Transformer-based)
        if TROCR_AVAILABLE:
            try:
                self.engines['trocr_processor'] = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                self.engines['trocr_model'] = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
                print("  ✅ TrOCR (Transformer) initialized")
            except Exception as e:
                print(f"  ❌ TrOCR failed: {e}")
        
        # Initialize Tesseract
        if TESSERACT_AVAILABLE:
            try:
                # Test if tesseract is available
                pytesseract.get_tesseract_version()
                self.engines['tesseract'] = True
                print("  ✅ Tesseract initialized")
            except Exception as e:
                print(f"  ❌ Tesseract failed: {e}")
        
        print(f"🎯 Total engines initialized: {len(self.engines)}")
    
    def extract_text_multi_engine(self, image_path: str, languages: List[str] = ['en', 'hi']) -> Dict:
        """Extract text using multiple OCR engines and combine results"""
        results = {}
        image = cv2.imread(image_path)
        
        if image is None:
            return {"error": "Could not load image"}
        
        print(f"🔍 Running multi-engine OCR analysis...")
        
        # EasyOCR
        if 'easyocr' in self.engines:
            try:
                result = self.engines['easyocr'].readtext(image, detail=0)
                text = ' '.join(result)
                results['easyocr'] = {
                    'text': text,
                    'confidence': 0.8,  # EasyOCR doesn't provide confidence
                    'language': 'multi'
                }
                print(f"  ✅ EasyOCR: {text[:50]}...")
            except Exception as e:
                results['easyocr'] = {"error": str(e)}
        
        # PaddleOCR
        if 'paddleocr' in self.engines:
            try:
                result = self.engines['paddleocr'].ocr(image, cls=True)
                if result and result[0]:
                    text = ' '.join([item[1][0] for item in result[0] if item[1][1] > 0.5])
                    avg_confidence = sum([item[1][1] for item in result[0]]) / len(result[0])
                    results['paddleocr'] = {
                        'text': text,
                        'confidence': avg_confidence,
                        'language': 'en'
                    }
                    print(f"  ✅ PaddleOCR: {text[:50]}...")
            except Exception as e:
                results['paddleocr'] = {"error": str(e)}
        
        # Google Cloud Vision
        if 'google_vision' in self.engines:
            try:
                text, confidence = self._google_vision_ocr(image_path)
                results['google_vision'] = {
                    'text': text,
                    'confidence': confidence,
                    'language': 'auto'
                }
                print(f"  ✅ Google Vision: {text[:50]}...")
            except Exception as e:
                results['google_vision'] = {"error": str(e)}
        
        # TrOCR (Transformer)
        if 'trocr_processor' in self.engines and 'trocr_model' in self.engines:
            try:
                text = self._trocr_extract(image_path)
                results['trocr'] = {
                    'text': text,
                    'confidence': 0.9,  # TrOCR is generally high quality
                    'language': 'en'
                }
                print(f"  ✅ TrOCR: {text[:50]}...")
            except Exception as e:
                results['trocr'] = {"error": str(e)}
        
        # Tesseract with multiple languages
        if 'tesseract' in self.engines:
            try:
                # Try different language combinations
                for lang_combo in ['eng', 'hin', 'eng+hin']:
                    try:
                        text = pytesseract.image_to_string(image, lang=lang_combo)
                        if text.strip():
                            results[f'tesseract_{lang_combo}'] = {
                                'text': text.strip(),
                                'confidence': 0.7,
                                'language': lang_combo
                            }
                            print(f"  ✅ Tesseract ({lang_combo}): {text[:50]}...")
                            break
                    except:
                        continue
            except Exception as e:
                results['tesseract'] = {"error": str(e)}
        
        # Mathpix API (if API key available)
        mathpix_result = self._mathpix_ocr(image_path)
        if mathpix_result:
            results['mathpix'] = mathpix_result
            print(f"  ✅ Mathpix: {mathpix_result.get('text', '')[:50]}...")
        
        return results
    
    def _google_vision_ocr(self, image_path: str) -> Tuple[str, float]:
        """Extract text using Google Cloud Vision API"""
        client = self.engines['google_vision']
        
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        
        if texts:
            return texts[0].description, 0.95  # Google Vision is usually very accurate
        
        return "", 0.0
    
    def _trocr_extract(self, image_path: str) -> str:
        """Extract text using TrOCR (Transformer-based OCR)"""
        processor = self.engines['trocr_processor']
        model = self.engines['trocr_model']
        
        image = Image.open(image_path).convert('RGB')
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text
    
    def _mathpix_ocr(self, image_path: str) -> Optional[Dict]:
        """Extract mathematical text using Mathpix API"""
        api_key = os.getenv('MATHPIX_APP_KEY')
        app_id = os.getenv('MATHPIX_APP_ID')
        
        if not api_key or not app_id:
            return None
        
        try:
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            response = requests.post(
                'https://api.mathpix.com/v3/text',
                headers={
                    'app_id': app_id,
                    'app_key': api_key,
                    'Content-Type': 'application/json'
                },
                json={
                    'src': f'data:image/png;base64,{image_data}',
                    'formats': ['text', 'latex_styled'],
                    'data_options': {
                        'include_asciimath': True,
                        'include_latex': True
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'text': result.get('text', ''),
                    'latex': result.get('latex_styled', ''),
                    'confidence': result.get('confidence', 0.0),
                    'language': 'math'
                }
        except Exception as e:
            print(f"Mathpix API error: {e}")
        
        return None
    
    def get_best_result(self, results: Dict) -> Dict:
        """Select the best OCR result based on confidence and content quality"""
        if not results:
            return {"text": "", "confidence": 0.0, "engine": "none"}
        
        # Filter out error results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {"text": "", "confidence": 0.0, "engine": "none", "error": "All engines failed"}
        
        # Scoring system
        scores = {}
        for engine, result in valid_results.items():
            text = result.get('text', '').strip()
            confidence = result.get('confidence', 0.0)
            
            if not text:
                scores[engine] = 0.0
                continue
            
            # Base score from confidence
            score = confidence
            
            # Bonus for mathematical content
            if re.search(r'[=+\-*/^()0-9]', text):
                score += 0.1
            
            # Bonus for longer text (more complete)
            if len(text) > 10:
                score += 0.05
            
            # Engine-specific bonuses
            if engine == 'google_vision':
                score += 0.1  # Google Vision is usually very good
            elif engine == 'mathpix':
                score += 0.15  # Mathpix is excellent for math
            elif engine == 'trocr':
                score += 0.05  # TrOCR is good for printed text
            
            scores[engine] = score
        
        # Get the best engine
        best_engine = max(scores.keys(), key=lambda k: scores[k])
        best_result = valid_results[best_engine].copy()
        best_result['engine'] = best_engine
        best_result['all_results'] = valid_results
        
        return best_result
    
    def extract_and_analyze(self, image_path: str, languages: List[str] = ['en', 'hi']) -> Dict:
        """Main method to extract text and return the best analysis"""
        print(f"🎯 Starting powerful OCR analysis for: {image_path}")
        
        # Get results from all engines
        all_results = self.extract_text_multi_engine(image_path, languages)
        
        # Get the best result
        best_result = self.get_best_result(all_results)
        
        # Add analysis
        if best_result.get('text'):
            best_result['analysis'] = self._analyze_mathematical_content(best_result['text'])
        
        print(f"🏆 Best result from: {best_result.get('engine', 'unknown')}")
        print(f"📝 Text: {best_result.get('text', '')[:100]}...")
        
        return best_result
    
    def _analyze_mathematical_content(self, text: str) -> Dict:
        """Analyze the extracted text for mathematical content"""
        analysis = {
            'has_equations': bool(re.search(r'[=]', text)),
            'has_arithmetic': bool(re.search(r'[+\-*/]', text)),
            'has_variables': bool(re.search(r'[a-zA-Z]', text)),
            'has_numbers': bool(re.search(r'\d', text)),
            'complexity': 'simple'
        }
        
        # Determine complexity
        if analysis['has_equations'] and analysis['has_variables']:
            analysis['complexity'] = 'advanced'
        elif analysis['has_arithmetic'] and analysis['has_numbers']:
            analysis['complexity'] = 'intermediate'
        
        return analysis


# Enhanced image preprocessing for better OCR
def enhance_image_for_ocr(image_path: str) -> str:
    """Apply advanced preprocessing to improve OCR accuracy"""
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple enhancement techniques
    
    # 1. Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # 2. Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # 3. Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
    
    # 4. Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # 5. Morphological operations to clean up
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Save enhanced image
    enhanced_path = image_path.replace('.', '_enhanced.')
    cv2.imwrite(enhanced_path, cleaned)
    
    return enhanced_path


# Main function for testing
if __name__ == "__main__":
    ocr = PowerfulOCREngine()
    
    # Test with sample image
    test_image = "sample.jpg"
    if os.path.exists(test_image):
        result = ocr.extract_and_analyze(test_image)
        print("\n" + "="*50)
        print("ENHANCED OCR RESULTS:")
        print("="*50)
        print(f"Best Engine: {result.get('engine', 'unknown')}")
        print(f"Text: {result.get('text', '')}")
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"Analysis: {result.get('analysis', {})}")
    else:
        print("No test image found. Place sample.jpg to test.") 