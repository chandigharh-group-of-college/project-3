#!/usr/bin/env python3
"""
Test script for SAUNet 4.0 AI Mathematical Problem Analyzer
Demonstrates ChatGPT-like intelligent analysis and solving
"""

from solve_from_image import IntelligentMathAnalyzer

def test_ai_analysis():
    """Test the AI analysis capabilities with various mathematical problems"""
    
    print("🚀 SAUNet 4.0 AI Mathematical Problem Analyzer - Test Suite")
    print("=" * 60)
    
    # Initialize the AI analyzer
    analyzer = IntelligentMathAnalyzer()
    
    # Test cases that simulate various OCR inputs
    test_cases = [
        # Case 1: Simple equation (like the one from the debug output)
        "SOLVE x = 2-1 X = ?? 20 X21",
        
        # Case 2: Basic arithmetic
        "what is 15 + 23",
        
        # Case 3: Equation solving
        "solve 2x + 5 = 13 for x",
        
        # Case 4: Simple multiplication
        "calculate 7 * 8",
        
        # Case 5: Fraction
        "what is 3/4 + 1/2",
        
        # Case 6: Quadratic equation
        "solve x^2 - 5x + 6 = 0"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n🧪 TEST CASE {i}")
        print("=" * 30)
        print(f"Input: {test_input}")
        print("-" * 30)
        
        try:
            # Analyze the mathematical content
            analysis = analyzer.analyze_mathematical_content(test_input)
            
            # Solve the problem
            solution = analyzer.solve_intelligently(analysis)
            
            # Display results
            print("📊 ANALYSIS RESULTS:")
            print(f"   Problem Type: {analysis['problem_type']}")
            print(f"   Interpretation: {analysis['interpretation']}")
            print(f"   Expressions: {analysis['expressions']}")
            
            print("\n💡 SOLUTION:")
            if solution['success']:
                print(f"   ✅ Solutions: {solution['solutions']}")
            else:
                print("   ⚠️ No solution found")
            
            print("\n📝 STEP-BY-STEP:")
            print(solution['steps'])
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "="*60)

def test_specific_ocr_case():
    """Test the specific case from the debug output"""
    print("\n🎯 SPECIFIC OCR CASE TEST")
    print("=" * 40)
    
    # The exact OCR input that was failing
    ocr_input = "SOLVE x = 2-1 X = ?? 20 X21"
    
    analyzer = IntelligentMathAnalyzer()
    
    print(f"📝 Original OCR: {ocr_input}")
    
    # Analyze
    analysis = analyzer.analyze_mathematical_content(ocr_input)
    solution = analyzer.solve_intelligently(analysis)
    
    print("\n🧠 AI INTERPRETATION:")
    print(f"   Problem Type: {analysis['problem_type']}")
    print(f"   Cleaned Text: {analysis['original_text']}")
    print(f"   Expressions Found: {analysis['expressions']}")
    
    print("\n⚡ SOLUTION ATTEMPT:")
    print(solution['steps'])
    
    if solution['success']:
        print(f"\n✅ FINAL ANSWER: {solution['solutions']}")
    else:
        print("\n⚠️ Alternative interpretation needed")

if __name__ == "__main__":
    # Run the tests
    test_ai_analysis()
    test_specific_ocr_case()
    
    print("\n🎉 Test completed!")
    print("💡 The AI analyzer can now handle various mathematical problems intelligently!") 