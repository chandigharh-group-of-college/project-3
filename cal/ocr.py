import cv2
import numpy as np
import os
import re
import sys
import threading
import concurrent.futures
from pathlib import Path

# Try importing OCR libraries, but make them optional
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Pytesseract not available. OCR functionality will be limited.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available. OCR functionality will be limited.")

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. OCR functionality will be limited.")

# Check if GPU is available for EasyOCR
USE_GPU = True  # Default setting
try:
    import torch
    if torch.cuda.is_available():
        print("CUDA GPU detected. Using GPU acceleration for OCR.")
    else:
        print("CUDA GPU not detected. Using CPU for OCR.")
        USE_GPU = False
except ImportError:
    print("PyTorch not available or CUDA not installed. Using CPU for OCR.")
    USE_GPU = False

# Global OCR engine instances
easyocr_engine = None
tesseract_available = False

# Global lock for thread-safe initialization
ocr_lock = threading.Lock()

def initialize_ocr_engines():
    """Initialize OCR engines if not already initialized."""
    global easyocr_engine, tesseract_available
    # Set tesseract_available if TESSERACT_AVAILABLE is True
    if TESSERACT_AVAILABLE:
        tesseract_available = True
    # Initialize EasyOCR with thread safety
    with ocr_lock:
        if easyocr_engine is None and easyocr is not None:
            try:
                print(f"Initializing EasyOCR with GPU={USE_GPU}. This may take a moment...")
                # Languages: English + basic math symbols
                easyocr_engine = easyocr.Reader(['en'], gpu=USE_GPU)
                print("EasyOCR initialized successfully.")
            except Exception as e:
                print(f"Error initializing EasyOCR: {str(e)}")
                easyocr_engine = None

def preprocess_image(image_path, is_path=True):
    """
    Preprocess the image for better OCR results.
    
    Args:
        image_path: Can be either a file path or a numpy image array
        is_path: Boolean indicating if image_path is a file path or an image array
        
    Returns:
        Preprocessed image
    """
    try:
        # Read image
        if is_path:
            img = cv2.imread(image_path)
            if img is None:
                return None, "Failed to read image file"
        else:
            img = image_path
        
        # Check if image is valid
        if img is None or img.size == 0:
            return None, "Invalid image data"
            
        # Make a copy to avoid modifying the original
        processed = img.copy()
        
        # Convert to grayscale if it's a color image
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed
            
        # Resize if the image is too small or too large
        height, width = gray.shape[:2]
        
        # Optimal size for OCR (empirically determined)
        ideal_height = 1200
        
        # Resize while maintaining aspect ratio if needed
        if height < 300 or height > 3000:
            aspect_ratio = width / height
            new_height = ideal_height
            new_width = int(aspect_ratio * new_height)
            gray = cv2.resize(gray, (new_width, new_height))
        
        # Apply adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal - median blur
        denoised = cv2.medianBlur(binary, 3)
        
        # Dilation to connect nearby text components
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        
        return dilated, None
        
    except Exception as e:
        return None, f"Error preprocessing image: {str(e)}"

def extract_text_from_image(image_path, is_path=True):
    """
    Extract text from an image using OCR.
    
    Args:
        image_path: Path to the image file or numpy image array
        is_path: Boolean indicating if image_path is a file path or an image array
        
    Returns:
        Extracted text or error message
    """
    # Initialize OCR engines if needed
    initialize_ocr_engines()
    
    # Preprocess the image
    processed_img, error = preprocess_image(image_path, is_path)
    if error:
        return f"Error: {error}"
    
    # Execute OCR with multiple engines concurrently
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_tasks = []
        
        # Use EasyOCR if available
        if easyocr_engine:
            future_tasks.append(executor.submit(extract_with_easyocr, processed_img))
            
        # Use Tesseract as fallback if available
        if tesseract_available:
            future_tasks.append(executor.submit(extract_with_tesseract, processed_img))
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_tasks):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"OCR thread error: {str(e)}")
    
    if not results:
        return "Error: No text could be extracted from the image"
    
    # Process and combine results, prioritizing mathematical expressions
    return process_ocr_results(results)

def extract_with_easyocr(processed_img):
    """Extract text using EasyOCR."""
    try:
        # Check if easyocr_engine is initialized
        global easyocr_engine
        if easyocr_engine is None:
            return None
            
        results = easyocr_engine.readtext(processed_img)
        # Extract text from results (results is a list of tuples or lists with bounding boxes and text)
        # Access elements safely
        extracted_text = ""
        for detection in results:
            # Handle both tuple and list return types from different EasyOCR versions
            if isinstance(detection, (list, tuple)) and len(detection) > 1:
                extracted_text += " " + str(detection[1])  # Convert to string for safety
        return extracted_text.strip()
    except Exception as e:
        print(f"EasyOCR error: {str(e)}")
        return None

def extract_with_tesseract(processed_img):
    """Extract text using pytesseract with math configuration."""
    try:
        # Convert to PIL Image for pytesseract
        pil_img = Image.fromarray(processed_img)
        
        # Use a custom configuration to improve math symbol recognition
        custom_config = r'--oem 3 --psm 6 -l eng+equ'
        extracted_text = pytesseract.image_to_string(pil_img, config=custom_config)
        return extracted_text
    except Exception as e:
        print(f"Tesseract error: {str(e)}")
        return None

def process_ocr_results(results):
    """Process and clean OCR results to get the best mathematical expression."""
    if not results:
        return None
    
    # Join all results
    combined_text = "\n".join(filter(None, results))
    
    # Clean up the text
    cleaned_text = clean_ocr_text(combined_text)
    
    # Try to detect equations or mathematical expressions
    math_expr = extract_math_expression(cleaned_text)
    
    if math_expr:
        return math_expr
    else:
        return cleaned_text

def clean_ocr_text(text):
    """Clean OCR text by removing unwanted characters and normalizing math symbols."""
    if not text:
        return ""
    
    # Replace common OCR errors in mathematical symbols
    replacements = {
        '×': '*',  # Replace multiplication sign with *
        '÷': '/',  # Normalize division sign
        '−': '-',  # Normalize minus sign
        '=': '=',  # Keep equals sign
        '^': '^',  # Keep caret for exponents
        '{': '(',  # Replace brackets with parentheses
        '}': ')',
        '[': '(',
        ']': ')',
    }
    
    # List of chars to keep intact (dont replace)
    preserving_chars = set('1234567890.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=^')
    
    # Normalize text
    cleaned = []
    for char in text:
        if char in replacements:
            cleaned.append(replacements[char])
        elif char in preserving_chars:
            cleaned.append(char)
        elif char.isspace():
            cleaned.append(' ')
    
    result = ''.join(cleaned).strip()
    
    # Only replace 'x' with '*' when it's between numbers or brackets
    # This preserves 'x' when used as a variable name
    result = re.sub(r'(\d|\))\s*x\s*(\d|\()', r'\1*\2', result)
    
    # Preserve variable 'x' when it's preceded by operators
    result = re.sub(r'([+\-*/=])\s*x', r'\1 x', result)
    
    # Preserve variable 'x' when it's followed by operators
    result = re.sub(r'x\s*([+\-*/=])', r'x \1', result)
    
    # Insert multiplication between number and variable (e.g., 2x -> 2*x)
    # But only when 'x' is clearly a variable, not a multiplication sign
    result = re.sub(r'(\d)([a-wyz])', r'\1*\2', result)  # all letters except 'x'
    result = re.sub(r'(\d)x(?![a-zA-Z0-9])', r'\1*x', result)  # only 'x' when not part of a word
    
    return result

def extract_math_expression(text):
    """Extract mathematical expressions from text."""
    if not text:
        return None
    
    # Define patterns for different types of mathematical expressions
    patterns = [
        # Matrix pattern like [[1,2],[3,4]]
        r'\[\s*\[\s*[-+]?\d+(?:\.\d+)?(?:\s*,\s*[-+]?\d+(?:\.\d+)?)*\s*\](?:\s*,\s*\[\s*[-+]?\d+(?:\.\d+)?(?:\s*,\s*[-+]?\d+(?:\.\d+)?)*\s*\])*\s*\]',
        
        # Basic equation pattern
        r'[-+]?(?:\d+(?:\.\d+)?|[a-zA-Z])\s*(?:[+\-*/=^]\s*[-+]?(?:\d+(?:\.\d+)?|[a-zA-Z]))+',
        
        # Polynomial equation pattern
        r'[-+]?(?:\d+(?:\.\d+)?)?[a-zA-Z](?:\^[-+]?\d+)?\s*(?:[+\-]\s*(?:\d+(?:\.\d+)?)?[a-zA-Z](?:\^[-+]?\d+)?)*\s*=\s*[-+]?(?:\d+(?:\.\d+)?)',
        
        # Function pattern
        r'\w+\([a-zA-Z0-9\s,]+\)',
        
        # Simple calculation
        r'[-+]?\d+(?:\.\d+)?\s*[-+*/]\s*[-+]?\d+(?:\.\d+)?(?:\s*[-+*/]\s*[-+]?\d+(?:\.\d+)?)*',
    ]
    
    # Look for matches using the patterns
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the longest match as it's likely the most complete expression
            return max(matches, key=len)
    
    # If no structured pattern was found, just return the cleaned text
    return text

class OCRThread(threading.Thread):
    def __init__(self, image_path, callback):
        super().__init__()
        self.image_path = image_path
        self.callback = callback
        self.daemon = True  # Thread will exit when main program exits
    
    def run(self):
        try:
            # Extract text from the image
            text = extract_text_from_image(self.image_path)
            
            # Call the callback with the result
            if self.callback:
                self.callback(text)
        except Exception as e:
            # Call the callback with the error message
            if self.callback:
                self.callback(f"Error: {str(e)}")

class MathOCR:
    """Class for extracting mathematical expressions from images using OCR."""
    
    def __init__(self, use_gpu=None):
        self.tesseract_available = TESSERACT_AVAILABLE
        self.easyocr_available = EASYOCR_AVAILABLE
        
        # Use provided GPU setting or default
        self.use_gpu = USE_GPU if use_gpu is None else use_gpu
        
        # Initialize EasyOCR reader if available
        if self.easyocr_available:
            try:
                print(f"Initializing EasyOCR with GPU={self.use_gpu}. This may take a moment...")
                self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
                print("EasyOCR initialized successfully.")
            except Exception as e:
                print(f"Error initializing EasyOCR with GPU: {str(e)}")
                # Try again with CPU if GPU failed
                if self.use_gpu:
                    print("Falling back to CPU...")
                    try:
                        self.use_gpu = False
                        self.reader = easyocr.Reader(['en'], gpu=False)
                        print("EasyOCR initialized with CPU successfully.")
                    except Exception as e2:
                        print(f"Error initializing EasyOCR with CPU: {str(e2)}")
                        self.easyocr_available = False
                else:
                    self.easyocr_available = False
        
        # Set path for pytesseract if available
        if self.tesseract_available:
            if os.name == 'nt':  # Windows
                # Check multiple possible Tesseract installation paths
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    # Add path from environment variable if set
                    os.environ.get('TESSERACT_PATH', '')
                ]
                
                # Try to find tesseract in PATH
                try:
                    from shutil import which
                    path_tesseract = which('tesseract')
                    if path_tesseract:
                        possible_paths.append(path_tesseract)
                except ImportError:
                    pass
                
                # Check each path
                found_tesseract = False
                for path in possible_paths:
                    if path and os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        found_tesseract = True
                        print(f"Tesseract found at: {path}")
                        break
                        
                if not found_tesseract:
                    print("Tesseract not found. Please install Tesseract OCR or set the TESSERACT_PATH environment variable.")
                    self.tesseract_available = False
    
    def preprocess_image(self, image):
        """Preprocess the image to improve OCR accuracy."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle uneven lighting
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Denoise the image
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        return denoised
    
    def extract_text_tesseract(self, image):
        """Extract text from image using Tesseract OCR."""
        if not self.tesseract_available:
            return "Tesseract OCR not available."
        
        try:
            preprocessed = self.preprocess_image(image)
            
            # Configure Tesseract for math equation recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789.+-*/()=xyz^√∫∂Σπ"'
            
            # Perform OCR
            text = pytesseract.image_to_string(preprocessed, config=custom_config)
            
            # Clean the extracted text
            text = self.clean_math_expression(text)
            
            return text
        except Exception as e:
            return f"Tesseract error: {str(e)}"
    
    def extract_text_easyocr(self, image):
        """Extract text from image using EasyOCR."""
        if not self.easyocr_available:
            return "EasyOCR not available."
        
        try:
            # EasyOCR works better with the original image
            result = self.reader.readtext(image)
            
            # Extract and combine text from all detected regions safely
            text_parts = []
            for item in result:
                # Handle both tuple and list return types from different EasyOCR versions
                if isinstance(item, (list, tuple)) and len(item) > 1:
                    text_parts.append(str(item[1]))  # Convert to string for safety
            
            text = ' '.join(text_parts)
            
            # Clean the extracted text
            text = self.clean_math_expression(text)
            
            return text
        except Exception as e:
            return f"EasyOCR error: {str(e)}"
    
    def extract_from_image(self, image_path=None, image_bytes=None, use_tesseract=True):
        """Extract mathematical expression from an image."""
        try:
            # Load the image
            if image_path:
                image = cv2.imread(image_path)
                if image is None:
                    return f"Error: Could not read image file at {image_path}"
            elif image_bytes:
                try:
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is None:
                        return "Error: Could not decode image data"
                except Exception as e:
                    return f"Error processing image data: {str(e)}"
            else:
                return "Error: No image provided"
            
            # Check if the image is valid before further processing
            if image is None or image.size == 0:
                return "Error: Invalid image data or empty image"
            
            # Check image size and resize if necessary
            max_dimension = 1200
            height, width = image.shape[:2]
            if width > max_dimension or height > max_dimension:
                # Calculate new dimensions while preserving aspect ratio
                if width > height:
                    new_width = max_dimension
                    new_height = int((max_dimension / width) * height)
                else:
                    new_height = max_dimension
                    new_width = int((max_dimension / height) * width)
                
                print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
                image = cv2.resize(image, (new_width, new_height))
            
            # Check if any OCR method is available
            if not self.tesseract_available and not self.easyocr_available:
                return "Error: No OCR engine available. Please install pytesseract or easyocr."
            
            # Try OCR methods
            results = []
            
            # Try Tesseract if requested and available
            if use_tesseract and self.tesseract_available:
                try:
                    text1 = self.extract_text_tesseract(image)
                    if text1 and not text1.startswith("Tesseract error:"):
                        results.append(text1)
                except Exception as e:
                    print(f"Tesseract OCR error: {str(e)}")
            
            # Try EasyOCR if available
            if self.easyocr_available:
                try:
                    text2 = self.extract_text_easyocr(image)
                    if text2 and not text2.startswith("EasyOCR error:"):
                        results.append(text2)
                except Exception as e:
                    print(f"EasyOCR error: {str(e)}")
            
            # Return the best result
            if not results:
                return "Error: All OCR methods failed. Try a clearer image or different OCR engine."
            
            # Choose the longer text as it might contain more information
            best_text = max(results, key=len)
            
            # Validate the result - ensure it contains some mathematical elements
            if len(best_text) < 3 or not any(c in best_text for c in "0123456789+-*/()=^√"):
                return "Error: No mathematical expression detected in the image."
                
            return best_text
            
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    def clean_math_expression(self, text):
        """Clean up the OCR-extracted text to make it more suitable for calculation."""
        if not text:
            return ""
        
        # Replace common OCR errors and format for proper calculation
        replacements = {
            '×': '*',  # Replace multiplication sign with *
            '÷': '/',  # Replace division sign with /
            '−': '-',  # Replace unicode minus with standard minus
            '\n': '',  # Remove newlines
            ' ': '',   # Remove spaces
        }
        
        # Apply replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Only replace 'x' with '*' when it's between numbers or brackets
        # This preserves 'x' when used as a variable name
        text = re.sub(r'(\d|\))\s*x\s*(\d|\()', r'\1*\2', text)
        
        # Insert multiplication between number and variable (e.g., 2x -> 2*x)
        # But only for variables other than 'x'
        text = re.sub(r'(\d)([a-wyz])', r'\1*\2', text)  # all letters except 'x'
        
        # For 'x', only add multiplication when it's not part of a word/identifier
        text = re.sub(r'(\d)x(?![a-zA-Z0-9])', r'\1*x', text)
        
        # Handle special mathematical characters
        text = text.replace('sqrt', 'sqrt')
        text = text.replace('pi', 'pi')
        
        return text
    
    def extract(self, image_path):
        """Extract mathematical expression from an image (UI compatibility method)."""
        return self.extract_from_image(image_path=image_path) 