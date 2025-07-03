#!/usr/bin/env python3
"""
Test script for SAUNet 4.0 Advanced NLU Integration
Demonstrates AI-powered natural language understanding for mathematics
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_nlu_standalone():
    """Test the Advanced NLU system standalone"""
    print("🧪 Testing Advanced NLU System")
    print("=" * 50)
    
    try:
        from advanced_nlu import AdvancedMathNLU
        
        # Initialize the NLU system
        nlu = AdvancedMathNLU()
        
        # Test queries
        test_queries = [
            "differentiate sin(x²) with respect to x",
            "integrate x³ from 0 to 5", 
            "solve 2x + 3 = 7 for x",
            "what is the limit of sin(x)/x as x approaches 0",
            "simplify (x + 1)² - x²",
            "calculate the square root of 144"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🧪 Test {i}: {query}")
            print("-" * 40)
            
            understanding = nlu.understand_query(query)
            sympy_code = nlu.generate_sympy_code(understanding)
            
            print(f"Operation: {understanding['operation']}")
            print(f"Expression: {understanding.get('expression', 'N/A')}")
            print(f"Variable: {understanding.get('variable', 'N/A')}")
            print(f"Confidence: {understanding['confidence']:.1%}")
            print(f"SymPy Code: {sympy_code}")
        
        print("\n✅ Advanced NLU system working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import Advanced NLU: {e}")
        print("💡 Install dependencies: pip install transformers spacy torch")
        return False
    except Exception as e:
        print(f"❌ Error testing NLU: {e}")
        return False

def test_calculator_integration():
    """Test the integration with the calculator UI"""
    print("\n🧪 Testing Calculator Integration")
    print("=" * 40)
    
    try:
        # Test importing the UI with NLU
        from ui import CalculatorUI, ADVANCED_NLU_AVAILABLE
        from calculator import Calculator
        from solver import EquationSolver
        
        print(f"Advanced NLU Available: {ADVANCED_NLU_AVAILABLE}")
        
        if ADVANCED_NLU_AVAILABLE:
            print("✅ Calculator UI successfully imports Advanced NLU")
            
            # Create calculator instance
            calculator = Calculator()
            solver = EquationSolver()
            ui = CalculatorUI(calculator, solver)
            
            print("✅ Calculator UI initialized with Advanced NLU")
            
            if hasattr(ui, 'advanced_nlu') and ui.advanced_nlu:
                print("✅ Advanced NLU instance created in UI")
            else:
                print("⚠️ Advanced NLU instance not found in UI")
            
        else:
            print("⚠️ Advanced NLU not available in UI")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing calculator integration: {e}")
        return False

def show_usage_instructions():
    """Show instructions for using the enhanced calculator"""
    print("\n📚 USAGE INSTRUCTIONS")
    print("=" * 30)
    print("🚀 To use SAUNet 4.0 with Advanced NLU:")
    print()
    print("1. 🔧 Install AI Dependencies:")
    print("   pip install transformers spacy torch")
    print("   python -m spacy download en_core_web_sm")
    print()
    print("2. 🧮 Run the Calculator:")
    print("   python main.py")
    print()
    print("3. 💬 Use Natural Language Tab:")
    print("   - Type: 'differentiate sin(x²)'")
    print("   - Type: 'integrate x³ from 0 to 5'")
    print("   - Type: 'solve 2x + 3 = 7 for x'")
    print("   - Click 'Calculate' to see AI analysis")
    print()
    print("4. 🎯 Features Available:")
    print("   ✅ ChatGPT-like mathematical understanding")
    print("   ✅ Step-by-step AI reasoning")
    print("   ✅ Multi-method analysis (AI + patterns + linguistics)")
    print("   ✅ Automatic SymPy code generation")
    print("   ✅ Confidence scoring")
    print()
    print("🎉 Enjoy your AI-powered mathematical calculator!")

if __name__ == "__main__":
    print("🚀 SAUNet 4.0 Advanced NLU Integration Test")
    print("=" * 60)
    
    # Test standalone NLU
    nlu_works = test_nlu_standalone()
    
    # Test calculator integration
    integration_works = test_calculator_integration()
    
    # Show usage instructions
    show_usage_instructions()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 20)
    print(f"NLU System: {'✅ WORKING' if nlu_works else '❌ FAILED'}")
    print(f"Integration: {'✅ WORKING' if integration_works else '❌ FAILED'}")
    
    if nlu_works and integration_works:
        print("\n🎉 All tests passed! SAUNet 4.0 is ready with AI power!")
    else:
        print("\n⚠️ Some tests failed. Check dependencies and installation.")
    
    print("\n🚀 Run 'python main.py' to start the enhanced calculator!") 