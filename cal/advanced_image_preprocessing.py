# advanced_image_preprocessing.py - Streamlined Image Preprocessing for Math OCR

import cv2
import numpy as np
from PIL import Image, ImageEnhance

class ImagePreprocessor:
    """Streamlined image preprocessing for mathematical OCR"""
    
    def __init__(self):
        print("🔬 Image preprocessor initialized")
    
    def process_math_image(self, image, enhancement_level='medium'):
        """
        Process mathematical images for better OCR
        
        Args:
            image: Input image (numpy array or PIL Image)
            enhancement_level: 'light', 'medium', 'aggressive'
        """
        print(f"🔬 Processing image (level: {enhancement_level})")
        
        # Convert to OpenCV format
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Basic denoising
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Background removal for medium/aggressive levels
        if enhancement_level in ['medium', 'aggressive']:
            enhanced = self._remove_background(enhanced)
        
        # Text optimization
        optimized = self._optimize_for_text(enhanced)
        
        # Final enhancement based on level
        if enhancement_level == 'aggressive':
            optimized = self._aggressive_enhance(optimized)
        elif enhancement_level == 'medium':
            optimized = self._medium_enhance(optimized)
        
        print("✅ Processing completed")
        return optimized
    
    def _remove_background(self, image):
        """Simple background removal"""
        # Gaussian blur to estimate background
        blur = cv2.GaussianBlur(image, (15, 15), 0)
        
        # Create foreground mask
        diff = cv2.absdiff(image, blur)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask and set background to white
        result = image.copy()
        result[mask == 0] = 255
        
        return result
    
    def _optimize_for_text(self, image):
        """Optimize for mathematical text recognition"""
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _medium_enhance(self, image):
        """Medium level enhancement"""
        try:
            pil_img = Image.fromarray(image)
            
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = enhancer.enhance(1.2)
            
            # Sharpness enhancement
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.3)
            
            return np.array(enhanced)
        except:
            return image
    
    def _aggressive_enhance(self, image):
        """Aggressive enhancement"""
        try:
            pil_img = Image.fromarray(image)
            
            # Strong contrast enhancement
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = enhancer.enhance(1.5)
            
            # Strong sharpness enhancement
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(2.0)
            
            return np.array(enhanced)
        except:
            return image


def create_preprocessor():
    """Create image preprocessor instance"""
    return ImagePreprocessor()


def preprocess_math_image(image_path, enhancement_level='medium', output_path=None):
    """
    Convenience function for preprocessing mathematical images
    
    Args:
        image_path: Path to input image
        enhancement_level: 'light', 'medium', 'aggressive'
        output_path: Optional path to save processed image
    """
    print(f"🔬 Processing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Process
    preprocessor = ImagePreprocessor()
    processed = preprocessor.process_math_image(image, enhancement_level)
    
    # Save if requested
    if output_path:
        cv2.imwrite(output_path, processed)
        print(f"💾 Saved to: {output_path}")
    
    return processed


if __name__ == "__main__":
    print("🧪 Testing Image Preprocessing")
    
    # Create test image
    test_image = np.ones((200, 300), dtype=np.uint8) * 255
    cv2.putText(test_image, "2x + 3 = 7", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add noise
    noise = np.random.randint(0, 30, test_image.shape, dtype=np.uint8)
    noisy_image = cv2.add(test_image, noise)
    
    # Test processing
    preprocessor = ImagePreprocessor()
    result = preprocessor.process_math_image(noisy_image, 'medium')
    
    print("✅ Test completed!") 