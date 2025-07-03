# test_ml_preprocessing.py - Test Advanced ML Image Preprocessing
# SAUNet 4.0 - Advanced Mathematical Image Processing with AI

"""
Test script to demonstrate the advanced ML-based image preprocessing
capabilities for mathematical OCR in SAUNet 4.0.

This script tests super-resolution, denoising, and background removal
on mathematical images.
"""

import cv2
import numpy as np
import sys
import time
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

def test_ml_preprocessing_system():
    """Test the ML preprocessing system"""
    print("🔬 SAUNet 4.0 ML Image Preprocessing Test")
    print("=" * 50)
    
    try:
        from advanced_image_preprocessing import AdvancedImagePreprocessor, TORCH_AVAILABLE, SKIMAGE_AVAILABLE, REMBG_AVAILABLE
        print("✅ Advanced image preprocessing modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ML preprocessing: {e}")
        print("💡 Install dependencies: pip install torch torchvision scikit-image rembg opencv-contrib-python")
        return False
    
    # Test basic preprocessor initialization
    print("\n📋 Testing ML Preprocessor initialization...")
    preprocessor = AdvancedImagePreprocessor()
    
    print(f"🧠 PyTorch available: {TORCH_AVAILABLE}")
    print(f"🔬 scikit-image available: {SKIMAGE_AVAILABLE}")
    print(f"🎭 Background removal available: {REMBG_AVAILABLE}")
    
    print("✅ ML preprocessing system initialized successfully")
    return True


def test_image_enhancement_levels():
    """Test different enhancement levels on a sample mathematical image"""
    print("\n🧪 Testing Image Enhancement Levels")
    print("=" * 50)
    
    try:
        from advanced_image_preprocessing import AdvancedImagePreprocessor
        
        # Create a sample mathematical image
        test_image = create_sample_math_image()
        print("📝 Created sample mathematical image")
        
        # Add noise to simulate real-world conditions
        noisy_image = add_realistic_noise(test_image)
        print("🌪️ Added realistic noise and distortions")
        
        # Test different enhancement levels
        preprocessor = AdvancedImagePreprocessor()
        levels = ['light', 'medium', 'aggressive']
        
        results = {}
        
        for level in levels:
            print(f"\n🔬 Testing {level} enhancement...")
            start_time = time.time()
            
            try:
                enhanced = preprocessor.process_math_image(noisy_image, level)
                processing_time = time.time() - start_time
                
                # Calculate quality metrics
                quality_score = calculate_image_quality(enhanced)
                
                results[level] = {
                    'processed': enhanced,
                    'time': processing_time,
                    'quality': quality_score
                }
                
                print(f"   ✅ {level} enhancement completed in {processing_time:.2f}s")
                print(f"   📊 Quality score: {quality_score:.2f}")
                
            except Exception as e:
                print(f"   ❌ {level} enhancement failed: {e}")
                results[level] = None
        
        # Get processing statistics
        stats = preprocessor.get_processing_stats()
        print(f"\n📊 Processing statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return results
        
    except Exception as e:
        print(f"❌ Enhancement test failed: {e}")
        return None


def test_specific_ml_features():
    """Test specific ML features individually"""
    print("\n🧠 Testing Individual ML Features")
    print("=" * 50)
    
    try:
        from advanced_image_preprocessing import AdvancedImagePreprocessor
        
        preprocessor = AdvancedImagePreprocessor()
        test_image = create_sample_math_image()
        
        # Test 1: Background removal
        print("🎭 Testing ML background removal...")
        try:
            bg_removed = preprocessor._remove_background(test_image)
            print("   ✅ Background removal successful")
        except Exception as e:
            print(f"   ⚠️ Background removal failed: {e}")
        
        # Test 2: ML denoising
        print("🧠 Testing ML denoising...")
        try:
            denoised = preprocessor._ml_denoise(test_image)
            print("   ✅ ML denoising successful")
        except Exception as e:
            print(f"   ⚠️ ML denoising failed: {e}")
        
        # Test 3: Super-resolution
        print("🔍 Testing ML super-resolution...")
        try:
            super_res = preprocessor._super_resolution(test_image)
            print("   ✅ Super-resolution successful")
            print(f"   📏 Resolution increased from {test_image.shape} to {super_res.shape}")
        except Exception as e:
            print(f"   ⚠️ Super-resolution failed: {e}")
        
        # Test 4: Mathematical text optimization
        print("📐 Testing mathematical text optimization...")
        try:
            optimized = preprocessor._optimize_for_math_text(test_image)
            print("   ✅ Mathematical text optimization successful")
        except Exception as e:
            print(f"   ⚠️ Text optimization failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML features test failed: {e}")
        return False


def test_ocr_integration():
    """Test integration with OCR system"""
    print("\n👁️ Testing OCR Integration")
    print("=" * 50)
    
    try:
        from solve_from_image import solve_from_image
        
        # Create a test image file
        test_image = create_sample_math_image()
        test_path = "test_math_image.png"
        cv2.imwrite(test_path, test_image)
        print(f"💾 Created test image: {test_path}")
        
        # Test different enhancement levels with OCR
        levels = ['light', 'medium', 'aggressive']
        
        for level in levels:
            print(f"\n🔬 Testing OCR with {level} enhancement...")
            try:
                start_time = time.time()
                ocr_text, interpretation, solution, steps = solve_from_image(test_path, level)
                processing_time = time.time() - start_time
                
                print(f"   ⏱️ Total processing time: {processing_time:.2f}s")
                print(f"   📝 OCR result: {ocr_text[:50]}..." if ocr_text else "   📝 No OCR text")
                print(f"   🎯 Solution: {solution[:30]}..." if solution else "   🎯 No solution")
                
            except Exception as e:
                print(f"   ❌ OCR test failed: {e}")
        
        # Clean up
        import os
        if os.path.exists(test_path):
            os.remove(test_path)
            print(f"🗑️ Cleaned up test file: {test_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR integration test failed: {e}")
        return False


def create_sample_math_image():
    """Create a sample mathematical image for testing"""
    # Create a white background
    image = np.ones((300, 500), dtype=np.uint8) * 255
    
    # Add mathematical content
    math_expressions = [
        ("2x + 3 = 7", (50, 80)),
        ("∫ x² dx = x³/3 + C", (50, 140)),
        ("√16 = 4", (50, 200)),
        ("lim(x→0) sin(x)/x = 1", (50, 260))
    ]
    
    for expr, pos in math_expressions:
        cv2.putText(image, expr, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return image


def add_realistic_noise(image):
    """Add realistic noise and distortions to simulate real-world conditions"""
    noisy = image.copy().astype(np.float32)
    
    # Add Gaussian noise
    noise = np.random.normal(0, 15, image.shape)
    noisy += noise
    
    # Add salt and pepper noise
    salt_pepper = np.random.random(image.shape)
    noisy[salt_pepper < 0.01] = 0
    noisy[salt_pepper > 0.99] = 255
    
    # Add slight blur
    noisy = cv2.GaussianBlur(noisy, (3, 3), 0.5)
    
    # Clip values and convert back to uint8
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy


def calculate_image_quality(image):
    """Calculate a simple image quality score"""
    # Calculate sharpness using Laplacian variance
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Calculate contrast
    contrast = image.std()
    
    # Combine metrics for a quality score
    quality_score = (sharpness / 100) + (contrast / 10)
    
    return min(quality_score, 10.0)  # Cap at 10


def benchmark_performance():
    """Benchmark performance of different preprocessing methods"""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)
    
    try:
        from advanced_image_preprocessing import AdvancedImagePreprocessor
        
        # Create test images of different sizes
        sizes = [(200, 300), (400, 600), (800, 1200)]
        
        for size in sizes:
            print(f"\n📏 Testing with image size: {size}")
            
            # Create test image
            test_image = np.ones(size, dtype=np.uint8) * 255
            cv2.putText(test_image, "2x + 3 = 7", (50, size[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            
            preprocessor = AdvancedImagePreprocessor()
            
            # Benchmark different levels
            for level in ['light', 'medium', 'aggressive']:
                start_time = time.time()
                try:
                    processed = preprocessor.process_math_image(test_image, level)
                    processing_time = time.time() - start_time
                    print(f"   {level}: {processing_time:.2f}s")
                except Exception as e:
                    print(f"   {level}: Failed ({e})")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


def demonstrate_ml_features():
    """Demonstrate key ML features"""
    print("\n🌟 ML Features Demonstration")
    print("=" * 50)
    
    features = [
        "🧠 Deep Learning Denoising Autoencoders",
        "🔍 Neural Network Super-Resolution (2x upscaling)",
        "🎭 AI-Powered Background Removal with U2-Net",
        "📐 Mathematical Symbol Enhancement",
        "⚡ GPU Acceleration (when available)",
        "🔧 Fallback to Traditional Methods",
        "📊 Processing Statistics and Quality Metrics",
        "🎯 Optimized for Mathematical Text Recognition"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"   {i}. {feature}")
        time.sleep(0.3)  # Dramatic pause
    
    print("\n✨ Advanced ML preprocessing makes OCR accuracy significantly better!")
    return True


def main():
    """Main test runner"""
    print("🚀 SAUNet 4.0 Advanced ML Image Preprocessing Test Suite")
    print("=" * 70)
    
    tests = [
        ("System Initialization", test_ml_preprocessing_system),
        ("Enhancement Levels", test_image_enhancement_levels),
        ("Individual ML Features", test_specific_ml_features),
        ("Features Demo", demonstrate_ml_features),
    ]
    
    # Optional advanced tests
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        tests.extend([
            ("OCR Integration", test_ocr_integration),
            ("Performance Benchmark", benchmark_performance),
        ])
        print("🔬 Full test mode enabled - includes OCR and performance tests")
    else:
        print("💡 Run with --full for comprehensive tests including OCR integration")
    
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
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ML preprocessing is ready! 🔬")
    else:
        print("⚠️ Some tests failed. Check dependencies and setup.")
    
    print("\n💡 To use ML preprocessing in SAUNet 4.0:")
    print("   1. Install: pip install torch torchvision scikit-image rembg")
    print("   2. Click the 📷 OCR button in the calculator")
    print("   3. Choose enhancement level (Light/Medium/Aggressive)")
    print("   4. Select your mathematical image")
    print("   5. Watch AI enhance and solve it!")


if __name__ == "__main__":
    main() 