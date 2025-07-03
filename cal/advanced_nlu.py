# advanced_nlu.py - Advanced AI Natural Language Understanding for Mathematics
# SAUNet 4.0 - ChatGPT-like mathematical language processing

import re
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try importing AI/NLP libraries with fallbacks
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers library loaded successfully")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")

try:
    import spacy
    SPACY_AVAILABLE = True
    print("✅ spaCy library loaded successfully")
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️ spaCy not available. Install with: pip install spacy")


class AdvancedMathNLU:
    """
    Advanced Natural Language Understanding for Mathematical Queries
    Uses transformer models and NLP techniques for intelligent math parsing
    """
    
    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        print(f"🚀 Initializing Advanced Math NLU on {self.device}")
        
        # Initialize models
        self.text2math_model = None
        self.nlp = None
        self.math_patterns = self._load_math_patterns()
        self.operation_mappings = self._load_operation_mappings()
        
        # Try to load models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models for mathematical language understanding"""
        try:
            if TRANSFORMERS_AVAILABLE:
                print("🧠 Loading FLAN-T5 model for mathematical reasoning...")
                # Use a lightweight model for math reasoning
                model_name = "google/flan-t5-small"
                self.text2math_model = pipeline(
                    "text2text-generation", 
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    max_length=256,
                    temperature=0.1
                )
                print("✅ FLAN-T5 model loaded successfully")
            
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("✅ spaCy English model loaded")
                except OSError:
                    print("⚠️ spaCy English model not found. Download with: python -m spacy download en_core_web_sm")
                    self.nlp = None
        
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("💡 Using fallback pattern-based parsing")
    
    def _load_math_patterns(self) -> Dict[str, str]:
        """Load comprehensive mathematical language patterns"""
        return {
            # Differentiation patterns
            r'differentiate\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w+))?': 'derivative',
            r'find\s+(?:the\s+)?derivative\s+of\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w+))?': 'derivative',
            r'(?:d/d(\w+)|∂/∂(\w+))\s*\((.+?)\)': 'derivative',
            r'(?:d/d(\w+)|∂/∂(\w+))\s+(.+)': 'derivative',
            
            # Integration patterns
            r'integrate\s+(.+?)(?:\s+from\s+(.+?)\s+to\s+(.+?))?(?:\s+with\s+respect\s+to\s+(\w+))?': 'integral',
            r'find\s+(?:the\s+)?integral\s+of\s+(.+?)(?:\s+from\s+(.+?)\s+to\s+(.+?))?': 'integral',
            r'∫\s*(.+?)(?:\s+d(\w+))?': 'integral',
            
            # Equation solving patterns
            r'solve\s+(.+?)\s+for\s+(\w+)': 'solve_for',
            r'solve\s+(.+?)(?:\s*=\s*(.+?))?': 'solve',
            r'find\s+(?:the\s+)?(?:value\s+of\s+)?(\w+)\s+(?:when|if|such\s+that)\s+(.+)': 'solve',
            r'what\s+is\s+(\w+)\s+if\s+(.+)': 'solve',
            
            # Simplification patterns
            r'simplify\s+(.+)': 'simplify',
            r'expand\s+(.+)': 'expand',
            r'factor\s+(.+)': 'factor',
            
            # Evaluation patterns
            r'evaluate\s+(.+?)(?:\s+at\s+(\w+)\s*=\s*(.+?))?': 'evaluate',
            r'calculate\s+(.+)': 'calculate',
            r'what\s+is\s+(.+)': 'calculate',
            r'compute\s+(.+)': 'calculate',
            
            # Limit patterns
            r'(?:find\s+(?:the\s+)?)?limit\s+of\s+(.+?)\s+as\s+(\w+)\s+approaches\s+(.+)': 'limit',
            r'lim\s*(?:_{(\w+)\s*→\s*(.+?)})?\s*(.+)': 'limit',
        }
    
    def _load_operation_mappings(self) -> Dict[str, str]:
        """Load mathematical operation mappings"""
        return {
            # Function mappings
            'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
            'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
            'sine': 'sin', 'cosine': 'cos', 'tangent': 'tan',
            'log': 'log', 'ln': 'log', 'logarithm': 'log',
            'exp': 'exp', 'exponential': 'exp',
            'sqrt': 'sqrt', 'square root': 'sqrt',
            
            # Operation mappings
            'plus': '+', 'add': '+', 'added to': '+',
            'minus': '-', 'subtract': '-', 'subtracted from': '-',
            'times': '*', 'multiply': '*', 'multiplied by': '*',
            'divide': '/', 'divided by': '/',
            'power': '**', 'raised to': '**', 'to the power of': '**',
            'squared': '**2', 'cubed': '**3',
            
            # Constant mappings
            'pi': 'pi', 'π': 'pi', 'euler': 'E', 'e': 'E',
            'infinity': 'oo', '∞': 'oo', 'inf': 'oo',
        }
    
    def understand_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to understand mathematical queries using AI
        Returns structured interpretation of the mathematical problem
        """
        print(f"🧠 [NLU] Understanding query: {query}")
        
        # Clean and normalize the query
        cleaned_query = self._clean_query(query)
        print(f"🔧 [NLU] Cleaned query: {cleaned_query}")
        
        # Try AI model first if available
        ai_result = None
        if self.text2math_model:
            ai_result = self._use_ai_model(cleaned_query)
            print(f"🤖 [AI MODEL] Result: {ai_result}")
        
        # Use pattern-based parsing as primary or fallback
        pattern_result = self._pattern_based_parsing(cleaned_query)
        print(f"🔍 [PATTERNS] Result: {pattern_result}")
        
        # Use spaCy for additional analysis if available
        linguistic_result = None
        if self.nlp:
            linguistic_result = self._linguistic_analysis(cleaned_query)
            print(f"📝 [LINGUISTIC] Result: {linguistic_result}")
        
        # Combine results intelligently
        final_result = self._combine_analyses(ai_result, pattern_result, linguistic_result, cleaned_query)
        print(f"✅ [FINAL] Understanding: {final_result}")
        
        return final_result
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the mathematical query"""
        # Convert to lowercase for processing
        query = query.lower().strip()
        
        # Replace mathematical symbols and notations
        replacements = {
            '×': '*', '÷': '/', '−': '-', '–': '-', '—': '-',
            '²': '**2', '³': '**3', '^': '**',
            'x²': 'x**2', 'x³': 'x**3',
            ' squared': '**2', ' cubed': '**3',
            ' to the power of ': '**',
            ' raised to the ': '**',
            ' raised to ': '**',
        }
        
        for old, new in replacements.items():
            query = query.replace(old, new)
        
        # Clean up spacing
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def _use_ai_model(self, query: str) -> Optional[Dict[str, Any]]:
        """Use transformer model for mathematical understanding"""
        if not self.text2math_model:
            return None
        
        try:
            # Prepare prompt for mathematical reasoning
            prompt = f"Convert this mathematical problem to a symbolic expression: {query}"
            
            # Generate response
            response = self.text2math_model(prompt, max_length=128, num_return_sequences=1)
            
            if response and len(response) > 0:
                generated_text = response[0]['generated_text'].strip()
                
                # Parse the AI response
                return {
                    'ai_interpretation': generated_text,
                    'confidence': 0.8,  # Placeholder confidence
                    'source': 'ai_model'
                }
        
        except Exception as e:
            print(f"⚠️ [AI MODEL] Error: {e}")
            return None
    
    def _pattern_based_parsing(self, query: str) -> Dict[str, Any]:
        """Advanced pattern-based parsing of mathematical language"""
        result = {
            'operation': 'unknown',
            'expression': None,
            'variable': 'x',
            'bounds': None,
            'confidence': 0.5,
            'source': 'patterns'
        }
        
        # Check against mathematical patterns
        for pattern, operation in self.math_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                if operation == 'derivative':
                    result.update({
                        'operation': 'derivative',
                        'expression': groups[0] if groups[0] else groups[2],
                        'variable': groups[1] if len(groups) > 1 and groups[1] else 'x',
                        'confidence': 0.9
                    })
                
                elif operation == 'integral':
                    result.update({
                        'operation': 'integral',
                        'expression': groups[0],
                        'variable': groups[3] if len(groups) > 3 and groups[3] else 'x',
                        'bounds': (groups[1], groups[2]) if len(groups) > 2 and groups[1] and groups[2] else None,
                        'confidence': 0.9
                    })
                
                elif operation in ['solve', 'solve_for']:
                    result.update({
                        'operation': 'solve',
                        'expression': groups[0],
                        'variable': groups[1] if len(groups) > 1 and groups[1] else 'x',
                        'confidence': 0.9
                    })
                
                elif operation in ['calculate', 'evaluate']:
                    result.update({
                        'operation': 'evaluate',
                        'expression': groups[0],
                        'confidence': 0.8
                    })
                
                elif operation == 'limit':
                    result.update({
                        'operation': 'limit',
                        'expression': groups[0] if groups[0] else groups[2],
                        'variable': groups[1] if len(groups) > 1 and groups[1] else 'x',
                        'limit_point': groups[2] if len(groups) > 2 and groups[2] else groups[1],
                        'confidence': 0.9
                    })
                
                elif operation in ['simplify', 'expand', 'factor']:
                    result.update({
                        'operation': operation,
                        'expression': groups[0],
                        'confidence': 0.8
                    })
                
                break
        
        # Post-process expression
        if result['expression']:
            result['expression'] = self._normalize_expression(result['expression'])
        
        return result
    
    def _linguistic_analysis(self, query: str) -> Optional[Dict[str, Any]]:
        """Use spaCy for linguistic analysis of the query"""
        if not self.nlp:
            return None
        
        try:
            doc = self.nlp(query)
            
            # Extract mathematical entities and relationships
            entities = []
            verbs = []
            numbers = []
            
            for token in doc:
                if token.pos_ == "VERB":
                    verbs.append(token.lemma_)
                elif token.like_num or token.shape_ in ["dd", "d", "d.d"]:
                    numbers.append(token.text)
                elif token.ent_type_ in ["QUANTITY", "CARDINAL"]:
                    entities.append(token.text)
            
            # Identify mathematical operations from verbs
            math_verbs = {
                'differentiate': 'derivative',
                'derive': 'derivative',
                'integrate': 'integral',
                'solve': 'solve',
                'calculate': 'evaluate',
                'compute': 'evaluate',
                'find': 'solve',
                'simplify': 'simplify',
                'expand': 'expand',
                'factor': 'factor'
            }
            
            operation = 'unknown'
            for verb in verbs:
                if verb in math_verbs:
                    operation = math_verbs[verb]
                    break
            
            return {
                'operation': operation,
                'verbs': verbs,
                'numbers': numbers,
                'entities': entities,
                'confidence': 0.6,
                'source': 'linguistic'
            }
        
        except Exception as e:
            print(f"⚠️ [LINGUISTIC] Error: {e}")
            return None
    
    def _normalize_expression(self, expr: str) -> str:
        """Normalize mathematical expressions for SymPy"""
        # Replace word-based operations with symbols
        for word, symbol in self.operation_mappings.items():
            expr = re.sub(r'\b' + re.escape(word) + r'\b', symbol, expr, flags=re.IGNORECASE)
        
        # Handle implicit multiplication
        expr = re.sub(r'(\d)\s*([a-zA-Z])', r'\1*\2', expr)  # 2x -> 2*x
        expr = re.sub(r'([a-zA-Z])\s*(\d)', r'\1*\2', expr)  # x2 -> x*2
        expr = re.sub(r'\)\s*\(', ')*(', expr)  # )( -> )*(
        expr = re.sub(r'(\w)\s*\(', r'\1*(', expr)  # x( -> x*(
        
        # Clean up spaces
        expr = re.sub(r'\s+', '', expr)
        
        return expr.strip()
    
    def _combine_analyses(self, ai_result: Optional[Dict], pattern_result: Dict, 
                         linguistic_result: Optional[Dict], original_query: str) -> Dict[str, Any]:
        """Intelligently combine results from different analysis methods"""
        
        # Start with pattern result as base
        final_result = pattern_result.copy()
        
        # Update with higher confidence results
        if ai_result and ai_result.get('confidence', 0) > final_result.get('confidence', 0):
            # Try to extract useful information from AI result
            ai_text = ai_result.get('ai_interpretation', '')
            if ai_text:
                final_result['ai_suggestion'] = ai_text
                final_result['confidence'] = max(final_result['confidence'], 0.7)
        
        # Enhance with linguistic information
        if linguistic_result:
            if linguistic_result.get('operation') != 'unknown':
                # If linguistic analysis found a clear operation, use it if pattern didn't
                if final_result['operation'] == 'unknown':
                    final_result['operation'] = linguistic_result['operation']
                    final_result['confidence'] = max(final_result['confidence'], 0.6)
            
            # Add additional context
            final_result['linguistic_context'] = {
                'verbs': linguistic_result.get('verbs', []),
                'numbers': linguistic_result.get('numbers', []),
                'entities': linguistic_result.get('entities', [])
            }
        
        # Add metadata
        final_result.update({
            'original_query': original_query,
            'processing_methods': [
                'patterns',
                'ai_model' if ai_result else None,
                'linguistic' if linguistic_result else None
            ],
        })
        
        return final_result
    
    def generate_sympy_code(self, understanding: Dict[str, Any]) -> str:
        """Generate SymPy code from the understanding result"""
        operation = understanding.get('operation', 'unknown')
        expression = understanding.get('expression', '')
        variable = understanding.get('variable', 'x')
        
        if not expression:
            return "# No expression found"
        
        # Normalize expression for SymPy
        expr = self._normalize_expression(expression)
        
        if operation == 'derivative':
            return f"sp.diff({expr}, {variable})"
        
        elif operation == 'integral':
            bounds = understanding.get('bounds')
            if bounds:
                return f"sp.integrate({expr}, ({variable}, {bounds[0]}, {bounds[1]}))"
            else:
                return f"sp.integrate({expr}, {variable})"
        
        elif operation == 'solve':
            if '=' in expr:
                return f"sp.solve({expr}, {variable})"
            else:
                return f"sp.solve(sp.Eq({expr}, 0), {variable})"
        
        elif operation == 'limit':
            limit_point = understanding.get('limit_point', '0')
            return f"sp.limit({expr}, {variable}, {limit_point})"
        
        elif operation == 'simplify':
            return f"sp.simplify({expr})"
        
        elif operation == 'expand':
            return f"sp.expand({expr})"
        
        elif operation == 'factor':
            return f"sp.factor({expr})"
        
        elif operation == 'evaluate':
            return f"sp.simplify({expr})"
        
        else:
            return f"sp.sympify('{expr}')"


# Test and example usage
def test_advanced_nlu():
    """Test the advanced NLU system with various mathematical queries"""
    print("🧪 Testing Advanced Mathematical NLU")
    print("=" * 50)
    
    nlu = AdvancedMathNLU()
    
    test_queries = [
        "differentiate sin(x²) with respect to x",
        "find the derivative of cos(3x + 1)",
        "integrate x² from 0 to 5",
        "solve 2x + 3 = 7 for x",
        "what is the limit of sin(x)/x as x approaches 0",
        "simplify (x + 1)² - x²",
        "calculate the square root of 144",
        "find x when 3x - 5 = 16",
        "expand (a + b)³",
        "factor x² - 4"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}: {query}")
        print("-" * 40)
        
        understanding = nlu.understand_query(query)
        sympy_code = nlu.generate_sympy_code(understanding)
        
        print(f"Operation: {understanding['operation']}")
        print(f"Expression: {understanding.get('expression', 'N/A')}")
        print(f"Variable: {understanding.get('variable', 'N/A')}")
        print(f"Confidence: {understanding['confidence']:.2f}")
        print(f"SymPy Code: {sympy_code}")


if __name__ == "__main__":
    test_advanced_nlu() 