# ai_step_generator.py - AI-Powered Step-by-Step Generator
# SAUNet 4.0 - ChatGPT-like Mathematical Explanations

"""
AI-powered mathematical step generator using OpenAI API and LangChain.
Provides ChatGPT-quality explanations for derivatives, integrals, equations, and more.
"""

import os
import re
import json
import warnings
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import sympy as sp

warnings.filterwarnings('ignore')

# OpenAI and LangChain imports with fallbacks
try:
    import openai
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import HumanMessage, SystemMessage
    import tiktoken
    OPENAI_AVAILABLE = True
    print("✅ OpenAI and LangChain available for AI step generation")
except ImportError as e:
    OPENAI_AVAILABLE = False
    print(f"⚠️ OpenAI/LangChain not available: {e}")
    print("💡 Install with: pip install openai langchain langchain-openai python-dotenv tiktoken")
    
    # Create dummy classes to prevent import errors
    class ChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            return None
    
    class HumanMessage:
        def __init__(self, content):
            self.content = content
    
    class SystemMessage:
        def __init__(self, content):
            self.content = content

# Environment management
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class StepExplanation:
    """Container for step-by-step mathematical explanations"""
    problem: str
    solution: str
    steps: List[str]
    explanation: str
    latex: str
    difficulty: str
    topic: str
    confidence: float
    tokens_used: int


class AIStepGenerator:
    """
    AI-powered mathematical step generator using OpenAI GPT models
    Provides ChatGPT-quality explanations for mathematical problems
    """
    
    def __init__(self, model="gpt-3.5-turbo", temperature=0.1):
        self.model = model
        self.temperature = temperature
        self.client = None
        self.chat_model = None
        self.token_encoder = None
        
        # Initialize OpenAI client
        self._initialize_openai()
        
        # Mathematical problem templates
        self.templates = {
            'derivative': self._get_derivative_template(),
            'integral': self._get_integral_template(),
            'equation': self._get_equation_template(),
            'general': self._get_general_template(),
            'conversational': self._get_conversational_template()
        }
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'total_tokens': 0,
            'average_response_time': 0
        }
    
    def _initialize_openai(self):
        """Initialize OpenAI client and LangChain models"""
        if not OPENAI_AVAILABLE:
            print("⚠️ OpenAI not available - using fallback explanations")
            return
        
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️ OPENAI_API_KEY not found in environment")
            print("💡 Set your API key: export OPENAI_API_KEY='your-key-here'")
            print("💡 Or create a .env file with: OPENAI_API_KEY=your-key-here")
            return
        
        try:
            # Initialize OpenAI client
            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
            
            # Initialize LangChain ChatOpenAI
            self.chat_model = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                openai_api_key=api_key
            )
            
            # Initialize token encoder
            self.token_encoder = tiktoken.encoding_for_model(self.model)
            
            print(f"✅ AI Step Generator initialized with {self.model}")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI: {e}")
            self.client = None
            self.chat_model = None
    
    def _get_derivative_template(self):
        """Template for derivative problems"""
        if not OPENAI_AVAILABLE:
            return None
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert mathematics tutor specializing in calculus. 
            Provide clear, step-by-step explanations for derivative problems. 
            Use proper mathematical notation and explain each rule used.
            Format your response with clear steps and reasoning."""),
            
            HumanMessage(content="""Please solve this derivative problem step-by-step:

Problem: Find the derivative of {expression}

Please provide:
1. A clear step-by-step solution
2. Explanation of each rule used (chain rule, product rule, etc.)
3. The final answer
4. Any important notes or alternative approaches

Make your explanation clear enough for a student to understand and follow along.""")
        ])
    
    def _get_integral_template(self):
        """Template for integral problems"""
        if not OPENAI_AVAILABLE:
            return None
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert mathematics tutor specializing in calculus integration. 
            Provide clear, step-by-step explanations for integral problems.
            Explain integration techniques like substitution, integration by parts, etc.
            Show all work clearly."""),
            
            HumanMessage(content="""Please solve this integral problem step-by-step:

Problem: Find the integral of {expression}

Please provide:
1. A detailed step-by-step solution
2. Explanation of the integration technique used
3. Any substitutions or transformations
4. The final answer with constant of integration (if indefinite)
5. Verification by differentiation (if appropriate)

Make your explanation educational and easy to follow.""")
        ])
    
    def _get_equation_template(self):
        """Template for equation solving"""
        if not OPENAI_AVAILABLE:
            return None
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert algebra tutor. 
            Solve equations step-by-step with clear explanations.
            Show each algebraic manipulation and explain the reasoning."""),
            
            HumanMessage(content="""Please solve this equation step-by-step:

Equation: {expression}

Please provide:
1. Clear step-by-step solution
2. Explanation of each algebraic operation
3. Check your answer by substitution
4. The final solution(s)

Make sure each step is justified and easy to understand.""")
        ])
    
    def _get_general_template(self):
        """Template for general mathematical problems"""
        if not OPENAI_AVAILABLE:
            return None
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert mathematics tutor with expertise in all areas of mathematics.
            Provide clear, educational explanations for any mathematical problem.
            Adapt your teaching style to the complexity of the problem."""),
            
            HumanMessage(content="""Please solve this mathematical problem step-by-step:

Problem: {expression}

Please provide:
1. Analysis of what type of problem this is
2. Step-by-step solution with explanations
3. Key concepts and techniques used
4. Final answer
5. Any useful insights or connections to other topics

Make your explanation comprehensive yet accessible.""")
        ])
    
    def _get_conversational_template(self):
        """Template for conversational AI assistant"""
        if not OPENAI_AVAILABLE:
            return None
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are SAUNet AI, an intelligent mathematical assistant integrated into a scientific calculator.
            You can help with any mathematical question or concept.
            Be friendly, helpful, and educational. Provide step-by-step explanations when solving problems.
            You can handle derivatives, integrals, equations, graphing, and general math questions."""),
            
            HumanMessage(content="""{user_query}

Please help the user with their mathematical question. If it's a problem to solve, provide step-by-step solutions.
If it's a concept question, provide clear explanations with examples.
Be conversational and educational.""")
        ])
    
    def generate_steps(self, expression: str, problem_type: str = 'general') -> StepExplanation:
        """
        Generate AI-powered step-by-step explanation for a mathematical expression
        
        Args:
            expression: Mathematical expression to solve
            problem_type: Type of problem ('derivative', 'integral', 'equation', 'general')
            
        Returns:
            StepExplanation object with detailed solution
        """
        if not self.client or not self.chat_model:
            return self._fallback_explanation(expression, problem_type)
        
        try:
            import time
            start_time = time.time()
            
            # Select appropriate template
            template = self.templates.get(problem_type, self.templates['general'])
            
            # Format the prompt
            if problem_type == 'conversational':
                messages = template.format_messages(user_query=expression)
            else:
                messages = template.format_messages(expression=expression)
            
            # Count input tokens
            input_text = ' '.join([msg.content for msg in messages])
            input_tokens = len(self.token_encoder.encode(input_text))
            
            # Generate response
            response = self.chat_model(messages)
            response_time = time.time() - start_time
            
            # Count output tokens
            output_tokens = len(self.token_encoder.encode(response.content))
            total_tokens = input_tokens + output_tokens
            
            # Parse response into structured format
            explanation = self._parse_ai_response(response.content, expression, problem_type)
            explanation.tokens_used = total_tokens
            
            # Update statistics
            self._update_stats(total_tokens, response_time, True)
            
            print(f"✅ AI explanation generated in {response_time:.2f}s ({total_tokens} tokens)")
            return explanation
            
        except Exception as e:
            print(f"⚠️ AI generation failed: {e}")
            self._update_stats(0, 0, False)
            return self._fallback_explanation(expression, problem_type)
    
    def _parse_ai_response(self, response: str, expression: str, problem_type: str) -> StepExplanation:
        """Parse AI response into structured StepExplanation"""
        
        # Extract steps (look for numbered or bulleted lists)
        steps = []
        step_patterns = [
            r'^\d+\.\s+(.+?)(?=^\d+\.|$)',
            r'^Step \d+[:.]?\s+(.+?)(?=^Step \d+|$)',
            r'^\*\s+(.+?)(?=^\*|$)',
            r'^-\s+(.+?)(?=^-|$)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
            if matches:
                steps = [step.strip() for step in matches]
                break
        
        # If no structured steps found, split by paragraphs
        if not steps:
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            steps = paragraphs[:10]  # Limit to 10 steps
        
        # Try to extract final answer
        solution = self._extract_solution(response)
        
        # Generate LaTeX if possible
        latex_expr = self._generate_latex(expression)
        
        # Determine difficulty and topic
        difficulty = self._assess_difficulty(expression, problem_type)
        topic = self._identify_topic(expression, problem_type)
        
        return StepExplanation(
            problem=expression,
            solution=solution,
            steps=steps,
            explanation=response,
            latex=latex_expr,
            difficulty=difficulty,
            topic=topic,
            confidence=0.9,  # High confidence for AI-generated content
            tokens_used=0  # Will be set by caller
        )
    
    def _extract_solution(self, response: str) -> str:
        """Extract the final solution from AI response"""
        # Look for common solution indicators
        patterns = [
            r'final answer[:\s]+(.+?)(?:\n|$)',
            r'solution[:\s]+(.+?)(?:\n|$)',
            r'answer[:\s]+(.+?)(?:\n|$)',
            r'therefore[,:\s]+(.+?)(?:\n|$)',
            r'result[:\s]+(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no explicit solution found, try to extract mathematical expressions
        math_patterns = [
            r'[fx]?\s*=\s*([^,\n]+)',
            r'∫[^=]*=\s*([^,\n]+)',
            r'd/dx[^=]*=\s*([^,\n]+)'
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1].strip()  # Return the last match
        
        return "See detailed explanation above"
    
    def _generate_latex(self, expression: str) -> str:
        """Generate LaTeX representation of the expression"""
        try:
            # Try to parse with SymPy and convert to LaTeX
            expr = sp.sympify(expression)
            return sp.latex(expr)
        except:
            # Fallback to basic LaTeX formatting
            latex = expression.replace('**', '^').replace('*', ' \\cdot ')
            latex = re.sub(r'(\d+)', r'{\1}', latex)  # Wrap numbers
            return latex
    
    def _assess_difficulty(self, expression: str, problem_type: str) -> str:
        """Assess the difficulty level of the problem"""
        complexity_indicators = {
            'basic': ['x', '+', '-', 'sin', 'cos', 'x^2', 'x^3'],
            'intermediate': ['chain rule', 'product rule', 'ln', 'log', 'tan', '^'],
            'advanced': ['integration by parts', 'substitution', 'partial fractions', 'complex']
        }
        
        expression_lower = expression.lower()
        
        if any(indicator in expression_lower for indicator in complexity_indicators['advanced']):
            return 'Advanced'
        elif any(indicator in expression_lower for indicator in complexity_indicators['intermediate']):
            return 'Intermediate'
        else:
            return 'Basic'
    
    def _identify_topic(self, expression: str, problem_type: str) -> str:
        """Identify the mathematical topic"""
        if problem_type == 'derivative':
            return 'Calculus - Derivatives'
        elif problem_type == 'integral':
            return 'Calculus - Integration'
        elif problem_type == 'equation':
            return 'Algebra - Equation Solving'
        else:
            # Try to identify from expression
            if any(word in expression.lower() for word in ['derivative', 'd/dx', 'differentiate']):
                return 'Calculus - Derivatives'
            elif any(word in expression.lower() for word in ['integral', '∫', 'integrate']):
                return 'Calculus - Integration'
            elif '=' in expression:
                return 'Algebra - Equations'
            else:
                return 'General Mathematics'
    
    def _fallback_explanation(self, expression: str, problem_type: str) -> StepExplanation:
        """Fallback explanation when AI is not available"""
        
        fallback_steps = {
            'derivative': [
                "Identify the function to differentiate",
                "Apply appropriate differentiation rules",
                "Simplify the result",
                "State the final derivative"
            ],
            'integral': [
                "Identify the function to integrate",
                "Choose appropriate integration technique", 
                "Apply the integration rules",
                "Add constant of integration (if indefinite)",
                "State the final result"
            ],
            'equation': [
                "Identify the type of equation",
                "Apply algebraic operations to isolate the variable",
                "Simplify both sides",
                "Check the solution by substitution"
            ],
            'general': [
                "Analyze the mathematical problem",
                "Apply relevant mathematical principles",
                "Perform necessary calculations",
                "Verify and state the final answer"
            ]
        }
        
        steps = fallback_steps.get(problem_type, fallback_steps['general'])
        
        return StepExplanation(
            problem=expression,
            solution="AI assistant unavailable - see steps for general approach",
            steps=steps,
            explanation=f"General approach for {problem_type} problems. For detailed AI-powered explanations, configure OpenAI API key.",
            latex=self._generate_latex(expression),
            difficulty="Unknown",
            topic=self._identify_topic(expression, problem_type),
            confidence=0.3,  # Low confidence for fallback
            tokens_used=0
        )
    
    def _update_stats(self, tokens: int, response_time: float, success: bool):
        """Update generation statistics"""
        self.stats['total_requests'] += 1
        if success:
            self.stats['successful_generations'] += 1
        self.stats['total_tokens'] += tokens
        
        # Update average response time
        prev_avg = self.stats['average_response_time']
        n = self.stats['total_requests']
        self.stats['average_response_time'] = (prev_avg * (n-1) + response_time) / n
    
    def chat_with_ai(self, query: str) -> StepExplanation:
        """
        Conversational AI assistant for mathematical questions
        
        Args:
            query: Natural language mathematical query
            
        Returns:
            StepExplanation with conversational response
        """
        return self.generate_steps(query, 'conversational')
    
    def get_stats(self) -> Dict:
        """Get generation statistics"""
        return self.stats.copy()
    
    def set_model(self, model: str):
        """Change the OpenAI model"""
        if model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview']:
            self.model = model
            if self.chat_model:
                self.chat_model.model_name = model
            print(f"✅ Switched to {model}")
        else:
            print(f"⚠️ Unsupported model: {model}")


def create_ai_generator(model="gpt-3.5-turbo") -> AIStepGenerator:
    """Factory function to create AI step generator"""
    return AIStepGenerator(model=model)


def test_ai_step_generator():
    """Test the AI step generation system"""
    print("🧠 Testing AI Step Generator")
    print("=" * 50)
    
    generator = AIStepGenerator()
    
    # Test problems
    test_problems = [
        ("x^2 + 3x + 2 = 0", "equation"),
        ("d/dx(sin(x^2))", "derivative"),
        ("∫ x^2 dx", "integral"),
        ("What is the chain rule?", "conversational")
    ]
    
    for expression, problem_type in test_problems:
        print(f"\n🔬 Testing {problem_type}: {expression}")
        
        try:
            explanation = generator.generate_steps(expression, problem_type)
            print(f"✅ Generated explanation:")
            print(f"   Topic: {explanation.topic}")
            print(f"   Difficulty: {explanation.difficulty}")
            print(f"   Steps: {len(explanation.steps)} steps")
            print(f"   Solution: {explanation.solution[:50]}...")
            print(f"   Confidence: {explanation.confidence}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Print statistics
    stats = generator.get_stats()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    test_ai_step_generator() 