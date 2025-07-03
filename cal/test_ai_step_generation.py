# test_ai_step_generation.py - Test AI Step Generation System
# SAUNet 4.0 - ChatGPT-like Mathematical Explanations

"""
Comprehensive test suite for the AI-powered step generation system.
Tests OpenAI integration, LangChain prompts, and conversational AI assistant.
"""

import os
import sys
import time
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing AI Step Generation Dependencies")
    print("=" * 50)
    
    dependencies = {
        'openai': 'OpenAI API library',
        'langchain': 'LangChain framework',
        'tiktoken': 'Token counting',
        'python-dotenv': 'Environment management'
    }
    
    available = {}
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            available[package] = True
            print(f"✅ {package}: {description}")
        except ImportError:
            available[package] = False
            print(f"❌ {package}: {description} - Not installed")
    
    return available


def test_environment_setup():
    """Test environment configuration"""
    print("\n🔧 Testing Environment Setup")
    print("=" * 50)
    
    # Check for .env file
    env_file_exists = os.path.exists('.env')
    print(f"📁 .env file: {'✅ Found' if env_file_exists else '⚠️ Not found'}")
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        masked_key = api_key[:8] + '*' * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else '*' * len(api_key)
        print(f"🔑 OpenAI API Key: ✅ Found ({masked_key})")
        return True
    else:
        print("🔑 OpenAI API Key: ❌ Not found")
        print("💡 Set your API key:")
        print("   - Environment variable: export OPENAI_API_KEY='your-key-here'")
        print("   - Or create .env file with: OPENAI_API_KEY=your-key-here")
        return False


def test_ai_step_generator():
    """Test the AI step generator system"""
    print("\n🧠 Testing AI Step Generator")
    print("=" * 50)
    
    try:
        from ai_step_generator import AIStepGenerator
        print("✅ AI Step Generator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AI Step Generator: {e}")
        return False
    
    try:
        # Initialize generator
        generator = AIStepGenerator(model="gpt-3.5-turbo")
        print("✅ AI Step Generator initialized")
        
        # Test different problem types
        test_problems = [
            ("x^2 + 3x + 2 = 0", "equation", "🔢 Equation Solving"),
            ("sin(x^2)", "derivative", "📈 Derivative"),
            ("x^3", "integral", "∫ Integration"),
            ("What is the chain rule?", "conversational", "💬 Conversational")
        ]
        
        results = {}
        
        for expression, problem_type, description in test_problems:
            print(f"\n{description}: {expression}")
            
            try:
                start_time = time.time()
                explanation = generator.generate_steps(expression, problem_type)
                end_time = time.time()
                
                if explanation:
                    print(f"   ✅ Generated in {end_time - start_time:.2f}s")
                    print(f"   📊 Topic: {explanation.topic}")
                    print(f"   📈 Difficulty: {explanation.difficulty}")
                    print(f"   🎯 Confidence: {explanation.confidence:.1%}")
                    print(f"   📝 Steps: {len(explanation.steps)} steps")
                    print(f"   💬 Solution: {explanation.solution[:50]}...")
                    
                    if hasattr(explanation, 'tokens_used'):
                        print(f"   🔢 Tokens: {explanation.tokens_used}")
                    
                    results[problem_type] = True
                else:
                    print("   ❌ No explanation generated")
                    results[problem_type] = False
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[problem_type] = False
        
        # Test statistics
        stats = generator.get_stats()
        print(f"\n📊 Generator Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return all(results.values())
        
    except Exception as e:
        print(f"❌ AI Step Generator test failed: {e}")
        return False


def test_ai_assistant():
    """Test the conversational AI assistant"""
    print("\n🤖 Testing AI Assistant")
    print("=" * 50)
    
    try:
        from ai_assistant import SAUNetAIAssistant
        print("✅ AI Assistant imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AI Assistant: {e}")
        return False
    
    try:
        # Initialize assistant
        assistant = SAUNetAIAssistant()
        print("✅ AI Assistant initialized")
        
        # Test capabilities
        capabilities = assistant.get_capabilities()
        print(f"🎯 Capabilities: {len(capabilities)} features")
        for capability in capabilities[:3]:  # Show first 3
            print(f"   {capability}")
        print("   ...")
        
        # Test conversational queries
        test_queries = [
            "Hello!",
            "What is the derivative of x^2?",
            "Solve 2x + 3 = 7",
            "What is calculus?"
        ]
        
        results = {}
        
        for query in test_queries:
            print(f"\n💬 Query: {query}")
            
            try:
                start_time = time.time()
                response = assistant.process_query(query)
                end_time = time.time()
                
                if response:
                    print(f"   ✅ Processed in {end_time - start_time:.2f}s")
                    print(f"   📂 Type: {response['type']}")
                    print(f"   📝 Response: {response['text'][:100]}...")
                    
                    results[query] = True
                else:
                    print("   ❌ No response generated")
                    results[query] = False
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[query] = False
        
        # Test conversation history
        history = assistant.get_conversation_history()
        print(f"\n📚 Conversation history: {len(history)} entries")
        
        return all(results.values())
        
    except Exception as e:
        print(f"❌ AI Assistant test failed: {e}")
        return False


def test_integration_with_calculator():
    """Test integration with the calculator system"""
    print("\n🧮 Testing Calculator Integration")
    print("=" * 50)
    
    try:
        from solve_from_image import IntelligentMathAnalyzer
        print("✅ IntelligentMathAnalyzer imported")
        
        # Test enhanced solver with AI
        analyzer = IntelligentMathAnalyzer()
        print("✅ Math analyzer initialized")
        
        # Test analysis with AI enhancement
        test_expression = "2x + 3 = 7"
        print(f"🔬 Testing: {test_expression}")
        
        analysis = analyzer.analyze_mathematical_content(test_expression)
        print(f"   📂 Problem type: {analysis['problem_type']}")
        print(f"   📝 Expressions: {len(analysis['expressions'])} found")
        
        # Test intelligent solving
        result = analyzer.solve_intelligently(analysis)
        print(f"   ✅ Success: {result['success']}")
        
        if 'ai_explanation' in result and result['ai_explanation']:
            print(f"   🤖 AI explanation: Generated")
            print(f"   📊 Confidence: {result['ai_explanation'].confidence:.1%}")
        else:
            print(f"   ⚠️ AI explanation: Not generated")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Calculator integration test failed: {e}")
        return False


def test_model_variations():
    """Test different OpenAI models"""
    print("\n🔄 Testing Model Variations")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ Skipping model tests - no API key")
        return True
    
    try:
        from ai_step_generator import AIStepGenerator
        
        models = ["gpt-3.5-turbo", "gpt-4"]  # Add more models as needed
        available_models = []
        
        for model in models:
            try:
                print(f"🧪 Testing {model}...")
                generator = AIStepGenerator(model=model)
                
                # Quick test
                explanation = generator.generate_steps("x^2", "derivative")
                if explanation:
                    print(f"   ✅ {model} working")
                    available_models.append(model)
                else:
                    print(f"   ⚠️ {model} no response")
                    
            except Exception as e:
                print(f"   ❌ {model} failed: {e}")
        
        print(f"\n📊 Available models: {len(available_models)}/{len(models)}")
        return len(available_models) > 0
        
    except Exception as e:
        print(f"❌ Model variation test failed: {e}")
        return False


def run_performance_benchmark():
    """Run performance benchmark"""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ Skipping benchmark - no API key")
        return True
    
    try:
        from ai_step_generator import AIStepGenerator
        
        generator = AIStepGenerator()
        
        # Benchmark different problem types
        problems = [
            ("x^2 + 5x + 6 = 0", "equation"),
            ("sin(x)", "derivative"),
            ("x^2", "integral"),
            ("2 + 3 * 4", "general")
        ]
        
        total_time = 0
        total_tokens = 0
        
        for expression, problem_type in problems:
            print(f"🔬 Benchmarking: {expression} ({problem_type})")
            
            start_time = time.time()
            explanation = generator.generate_steps(expression, problem_type)
            end_time = time.time()
            
            processing_time = end_time - start_time
            total_time += processing_time
            
            if explanation and hasattr(explanation, 'tokens_used'):
                total_tokens += explanation.tokens_used
                print(f"   ⏱️ Time: {processing_time:.2f}s")
                print(f"   🔢 Tokens: {explanation.tokens_used}")
            else:
                print(f"   ⏱️ Time: {processing_time:.2f}s (no token count)")
        
        avg_time = total_time / len(problems)
        print(f"\n📊 Benchmark Results:")
        print(f"   Average time: {avg_time:.2f}s")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Tokens per request: {total_tokens // len(problems) if total_tokens > 0 else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


def create_sample_env_file():
    """Create a sample .env file"""
    print("\n📝 Creating Sample .env File")
    print("=" * 50)
    
    env_content = """# SAUNet 4.0 AI Configuration
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Model Configuration
# OPENAI_MODEL=gpt-3.5-turbo
# OPENAI_TEMPERATURE=0.1

# Optional: Advanced Settings
# OPENAI_MAX_TOKENS=2000
# OPENAI_TIMEOUT=30
"""
    
    try:
        if not os.path.exists('.env'):
            with open('.env.sample', 'w') as f:
                f.write(env_content)
            print("✅ Created .env.sample file")
            print("💡 Copy .env.sample to .env and add your OpenAI API key")
        else:
            print("⚠️ .env file already exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env.sample: {e}")
        return False


def main():
    """Main test runner"""
    print("🚀 SAUNet 4.0 AI Step Generation Test Suite")
    print("=" * 70)
    print("Testing ChatGPT-like mathematical explanations and AI assistant")
    print()
    
    # Test configuration
    run_full_tests = len(sys.argv) > 1 and sys.argv[1] == "--full"
    api_tests_enabled = len(sys.argv) > 1 and "--no-api" not in sys.argv
    
    if run_full_tests:
        print("🔬 Full test mode enabled")
    if not api_tests_enabled:
        print("⚠️ API tests disabled")
    print()
    
    # Run tests
    test_results = {}
    
    # Basic tests (always run)
    test_results['Dependencies'] = test_dependencies()
    test_results['Environment'] = test_environment_setup()
    test_results['Sample .env'] = create_sample_env_file()
    
    # API-dependent tests
    if api_tests_enabled and os.getenv('OPENAI_API_KEY'):
        test_results['AI Step Generator'] = test_ai_step_generator()
        test_results['AI Assistant'] = test_ai_assistant()
        test_results['Calculator Integration'] = test_integration_with_calculator()
        
        if run_full_tests:
            test_results['Model Variations'] = test_model_variations()
            test_results['Performance Benchmark'] = run_performance_benchmark()
    else:
        print("\n⚠️ Skipping API tests - no OpenAI API key or API tests disabled")
    
    # Results summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI step generation is ready! 🤖")
    else:
        print("⚠️ Some tests failed. Check the requirements and setup.")
    
    # Setup instructions
    print("\n💡 Quick Setup Guide:")
    print("1. Install dependencies: pip install openai langchain langchain-openai python-dotenv tiktoken")
    print("2. Get OpenAI API key: https://platform.openai.com/api-keys")
    print("3. Set environment variable: export OPENAI_API_KEY='your-key'")
    print("4. Or create .env file with your API key")
    print("5. Test with: python test_ai_step_generation.py")
    
    print("\n🚀 Ready to use AI-powered mathematical explanations!")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 