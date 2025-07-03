# ai_assistant.py - Conversational AI Mathematical Assistant
# SAUNet 4.0 - ChatGPT-like Mathematical AI Assistant

"""
Conversational AI assistant for mathematical queries.
Provides natural language interface for mathematical problem solving.
"""

import re
import sympy as sp
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import AI step generator
try:
    from ai_step_generator import AIStepGenerator, StepExplanation
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class SAUNetAIAssistant:
    """
    Conversational AI assistant for mathematical problems
    Provides ChatGPT-like natural language mathematical understanding
    """
    
    def __init__(self):
        self.ai_generator = None
        self.conversation_history = []
        self.context = {}
        
        # Initialize AI generator
        if AI_AVAILABLE:
            try:
                self.ai_generator = AIStepGenerator()
                print("🤖 SAUNet AI Assistant initialized")
            except Exception as e:
                print(f"⚠️ AI Assistant initialization failed: {e}")
        else:
            print("⚠️ AI Assistant not available - install OpenAI dependencies")
        
        # Mathematical query patterns
        self.query_patterns = self._load_query_patterns()
        
        # Conversation starters
        self.greetings = [
            "Hello! I'm SAUNet AI, your mathematical assistant. How can I help you today?",
            "Hi there! Ready to solve some math problems together?",
            "Welcome! I can help with derivatives, integrals, equations, and more. What would you like to work on?",
            "Greetings! I'm here to help with your mathematical questions. What's on your mind?"
        ]
    
    def _load_query_patterns(self):
        """Load patterns for interpreting natural language mathematical queries"""
        return {
            # Derivative queries
            'derivative': [
                r'(?:find|calculate|what\s+is)?\s*(?:the\s+)?derivative\s+of\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w+))?',
                r'differentiate\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w+))?',
                r'd/d(\w+)\s*\(?(.+?)\)?',
                r'f\'?\s*\(\s*(\w+)\s*\)\s*=?\s*(.+)',
            ],
            
            # Integral queries
            'integral': [
                r'(?:find|calculate|what\s+is)?\s*(?:the\s+)?integral\s+of\s+(.+?)(?:\s+from\s+(.+?)\s+to\s+(.+?))?',
                r'integrate\s+(.+?)(?:\s+from\s+(.+?)\s+to\s+(.+?))?',
                r'∫\s*(.+?)(?:\s+d(\w+))?',
                r'∫_(\S+)\^(\S+)\s*(.+?)\s*d(\w+)',
            ],
            
            # Equation solving
            'equation': [
                r'solve\s+(.+?)\s*=\s*(.+?)(?:\s+for\s+(\w+))?',
                r'find\s+(\w+)\s+(?:when|if|where)\s+(.+)',
                r'what\s+is\s+(\w+)\s+(?:when|if|where)\s+(.+)',
                r'(.+?)\s*=\s*(.+?)(?:\s*solve\s*for\s*(\w+))?',
            ],
            
            # Evaluation queries
            'evaluate': [
                r'(?:calculate|evaluate|what\s+is)\s+(.+)',
                r'(.+?)\s*=\s*\?',
                r'find\s+the\s+value\s+of\s+(.+)',
            ],
            
            # Graphing queries
            'graph': [
                r'(?:plot|graph|draw)\s+(.+)',
                r'show\s+(?:me\s+)?(?:the\s+)?graph\s+of\s+(.+)',
                r'visualize\s+(.+)',
            ],
            
            # Conceptual queries
            'concept': [
                r'what\s+is\s+(?:the\s+)?(.+?)\s*\?',
                r'explain\s+(.+)',
                r'how\s+do\s+(?:you|i)\s+(.+?)\s*\?',
                r'tell\s+me\s+about\s+(.+)',
            ]
        }
    
    def process_query(self, query: str) -> Dict:
        """
        Process a natural language mathematical query
        
        Args:
            query: Natural language mathematical question
            
        Returns:
            Dictionary with response, type, and additional info
        """
        if not query.strip():
            return self._create_response("Please ask me a mathematical question!", "greeting")
        
        # Clean and normalize query
        cleaned_query = self._clean_query(query)
        
        # Add to conversation history
        self.conversation_history.append({"user": query, "timestamp": self._get_timestamp()})
        
        # Check for greetings
        if self._is_greeting(cleaned_query):
            response = self._handle_greeting()
        # Check for mathematical queries
        elif self._contains_math(cleaned_query):
            response = self._handle_mathematical_query(cleaned_query)
        # Check for conversational queries
        else:
            response = self._handle_conversational_query(cleaned_query)
        
        # Add response to history
        self.conversation_history.append({"assistant": response, "timestamp": self._get_timestamp()})
        
        return response
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the user query"""
        # Convert to lowercase for pattern matching
        cleaned = query.lower().strip()
        
        # Replace common mathematical symbols
        replacements = {
            'times': '*', 'divided by': '/', 'plus': '+', 'minus': '-',
            'squared': '**2', 'cubed': '**3', 'to the power of': '**',
            'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
            'ln': 'ln', 'log': 'log', 'sqrt': 'sqrt',
            'pi': 'pi', 'e': 'E'
        }
        
        for word, symbol in replacements.items():
            cleaned = cleaned.replace(word, symbol)
        
        return cleaned
    
    def _is_greeting(self, query: str) -> bool:
        """Check if the query is a greeting"""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                    'good evening', 'how are you', 'what can you do']
        return any(greeting in query for greeting in greetings)
    
    def _contains_math(self, query: str) -> bool:
        """Check if the query contains mathematical content"""
        math_indicators = [
            'derivative', 'integral', 'solve', 'calculate', 'evaluate',
            'differentiate', 'integrate', 'find', 'graph', 'plot',
            '=', '+', '-', '*', '/', '^', 'sin', 'cos', 'tan', 'ln', 'log',
            'x', 'y', 'equation', 'function', 'limit', 'series'
        ]
        return any(indicator in query for indicator in math_indicators)
    
    def _handle_greeting(self) -> Dict:
        """Handle greeting messages"""
        import random
        greeting = random.choice(self.greetings)
        
        return self._create_response(
            greeting + "\n\n💡 Try asking me:\n"
            "• 'Find the derivative of sin(x²)'\n"
            "• 'Solve x² + 3x + 2 = 0'\n"
            "• 'What is the integral of x³?'\n"
            "• 'Evaluate 2 + 3 × 4'",
            "greeting"
        )
    
    def _handle_mathematical_query(self, query: str) -> Dict:
        """Handle mathematical queries using AI"""
        # Try to identify query type and extract mathematical expression
        query_type, expression, extras = self._parse_mathematical_query(query)
        
        if not self.ai_generator:
            return self._create_response(
                "I'd love to help with that mathematical problem, but I need an OpenAI API key to provide detailed explanations.\n\n"
                "For now, you can:\n"
                "• Use the calculator tabs for basic operations\n"
                "• Try the OCR feature for image-based problems\n"
                "• Set up OpenAI API for AI-powered explanations",
                "error"
            )
        
        try:
            # Generate AI explanation
            print(f"🤖 Processing mathematical query: {query_type}")
            
            if query_type == 'unknown':
                # Use conversational AI for general queries
                explanation = self.ai_generator.chat_with_ai(query)
            else:
                # Use specific mathematical AI
                explanation = self.ai_generator.generate_steps(expression, query_type)
            
            # Format response
            response_text = self._format_ai_response(explanation, query_type)
            
            return self._create_response(
                response_text,
                query_type,
                {
                    'expression': expression,
                    'explanation': explanation,
                    'latex': getattr(explanation, 'latex', ''),
                    'steps': getattr(explanation, 'steps', []),
                    'confidence': getattr(explanation, 'confidence', 0.0)
                }
            )
            
        except Exception as e:
            print(f"⚠️ AI query processing failed: {e}")
            return self._create_response(
                f"I encountered an issue processing your mathematical query: {e}\n\n"
                "Please try rephrasing your question or check that your OpenAI API key is configured correctly.",
                "error"
            )
    
    def _handle_conversational_query(self, query: str) -> Dict:
        """Handle non-mathematical conversational queries"""
        if not self.ai_generator:
            return self._create_response(
                "I'm primarily designed to help with mathematical problems. "
                "Try asking me about derivatives, integrals, equations, or calculations!",
                "info"
            )
        
        try:
            # Use conversational AI
            explanation = self.ai_generator.chat_with_ai(query)
            response_text = explanation.explanation if explanation else "I'm not sure how to help with that."
            
            return self._create_response(response_text, "conversation")
            
        except Exception as e:
            return self._create_response(
                "I'm having trouble understanding your question. "
                "Could you try asking about a specific mathematical topic?",
                "error"
            )
    
    def _parse_mathematical_query(self, query: str) -> Tuple[str, str, Dict]:
        """Parse mathematical query to identify type and extract expression"""
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    if query_type == 'derivative':
                        expression = groups[0] if groups[0] else ""
                        variable = groups[1] if len(groups) > 1 and groups[1] else "x"
                        return 'derivative', expression, {'variable': variable}
                    
                    elif query_type == 'integral':
                        expression = groups[0] if groups[0] else ""
                        extras = {}
                        if len(groups) > 2 and groups[1] and groups[2]:
                            extras['from'] = groups[1]
                            extras['to'] = groups[2]
                        return 'integral', expression, extras
                    
                    elif query_type == 'equation':
                        if len(groups) >= 2:
                            expression = f"{groups[0]} = {groups[1]}"
                        else:
                            expression = groups[0] if groups[0] else ""
                        variable = groups[-1] if len(groups) > 2 and groups[-1] else "x"
                        return 'equation', expression, {'variable': variable}
                    
                    elif query_type == 'evaluate':
                        expression = groups[0] if groups[0] else ""
                        return 'general', expression, {}
                    
                    elif query_type == 'graph':
                        expression = groups[0] if groups[0] else ""
                        return 'general', f"Graph of {expression}", {}
                    
                    elif query_type == 'concept':
                        concept = groups[0] if groups[0] else ""
                        return 'conversational', f"Explain {concept}", {}
        
        # If no pattern matches, treat as general query
        return 'unknown', query, {}
    
    def _format_ai_response(self, explanation: StepExplanation, query_type: str) -> str:
        """Format AI explanation into readable response"""
        if not explanation:
            return "I couldn't generate an explanation for this problem."
        
        response_parts = []
        
        # Add topic and difficulty if available
        if hasattr(explanation, 'topic') and explanation.topic:
            response_parts.append(f"📚 **Topic**: {explanation.topic}")
        
        if hasattr(explanation, 'difficulty') and explanation.difficulty:
            response_parts.append(f"📊 **Difficulty**: {explanation.difficulty}")
        
        # Add main explanation
        if explanation.explanation:
            response_parts.append(f"\n{explanation.explanation}")
        
        # Add solution if available
        if explanation.solution and explanation.solution != "See detailed explanation above":
            response_parts.append(f"\n🎯 **Final Answer**: {explanation.solution}")
        
        # Add LaTeX if available
        if hasattr(explanation, 'latex') and explanation.latex:
            response_parts.append(f"\n📝 **LaTeX**: `{explanation.latex}`")
        
        # Add confidence if available
        if hasattr(explanation, 'confidence') and explanation.confidence:
            confidence_emoji = "🟢" if explanation.confidence > 0.8 else "🟡" if explanation.confidence > 0.5 else "🔴"
            response_parts.append(f"\n{confidence_emoji} **Confidence**: {explanation.confidence:.1%}")
        
        return '\n'.join(response_parts)
    
    def _create_response(self, text: str, response_type: str, extras: Optional[Dict] = None) -> Dict:
        """Create a formatted response dictionary"""
        return {
            'text': text,
            'type': response_type,
            'timestamp': self._get_timestamp(),
            'extras': extras or {}
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.context.clear()
    
    def get_capabilities(self) -> List[str]:
        """Get list of AI assistant capabilities"""
        return [
            "🧮 Solve mathematical equations",
            "📈 Find derivatives and integrals", 
            "📊 Evaluate expressions and calculations",
            "🎯 Step-by-step explanations",
            "💬 Natural language understanding",
            "📝 LaTeX formatting",
            "🤖 AI-powered mathematical reasoning",
            "📚 Concept explanations",
            "🔍 Problem type identification"
        ]


def create_ai_assistant() -> SAUNetAIAssistant:
    """Factory function to create AI assistant"""
    return SAUNetAIAssistant()


def test_ai_assistant():
    """Test the AI assistant functionality"""
    print("🤖 Testing SAUNet AI Assistant")
    print("=" * 50)
    
    assistant = SAUNetAIAssistant()
    
    # Test queries
    test_queries = [
        "Hello!",
        "What is the derivative of sin(x^2)?",
        "Solve x^2 + 3x + 2 = 0",
        "Find the integral of x^3",
        "Calculate 2 + 3 * 4",
        "What is the chain rule?",
        "Plot the function y = x^2"
    ]
    
    for query in test_queries:
        print(f"\n🔤 Query: {query}")
        response = assistant.process_query(query)
        print(f"🤖 Type: {response['type']}")
        print(f"📝 Response: {response['text'][:100]}...")
        
        if response['extras']:
            print(f"🔍 Extras: {list(response['extras'].keys())}")
    
    # Show capabilities
    print(f"\n🎯 AI Assistant Capabilities:")
    for capability in assistant.get_capabilities():
        print(f"   {capability}")


if __name__ == "__main__":
    test_ai_assistant() 