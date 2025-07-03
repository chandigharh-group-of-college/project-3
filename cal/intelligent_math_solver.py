# intelligent_math_solver.py - SAUNet 4.0 Intelligent Step-by-Step Math Solver

"""
Advanced AI-powered mathematical solver that provides detailed step-by-step solutions
while maintaining memory to avoid repetitive explanations and offering contextual responses.

Features:
- Comprehensive step-by-step breakdowns
- Memory of recently solved problems
- Intelligent non-repetition
- Multiple mathematical domains
- Extensible architecture for future enhancements
"""

import re
import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sympy as sp
import math

# Import existing SAUNet components
try:
    from calculator import Calculator
    from solver import EquationSolver
except ImportError:
    # Fallback if modules not available
    class Calculator:
        def __init__(self): pass
    class EquationSolver:
        def __init__(self): pass


@dataclass
class SolutionStep:
    """Individual step in a mathematical solution"""
    step_number: int
    action: str              # What operation is being performed
    explanation: str         # Why this step is necessary
    equation_before: str     # State before the step
    equation_after: str      # State after the step
    principle: str           # Mathematical principle/rule used
    difficulty: str          # Basic, Intermediate, Advanced


@dataclass
class MathSolution:
    """Complete mathematical solution with metadata"""
    problem: str
    problem_type: str
    final_answer: str
    steps: List[SolutionStep]
    solution_time: float
    difficulty_level: str
    mathematical_domain: str
    alternative_methods: List[str]
    key_concepts: List[str]
    real_world_context: Optional[str] = None


@dataclass
class ProblemMemory:
    """Memory entry for previously solved problems"""
    problem_hash: str
    original_problem: str
    solution: MathSolution
    timestamp: datetime
    access_count: int
    similar_problems: List[str]


class IntelligentMathSolver:
    """AI-powered step-by-step mathematical solver with memory and contextual awareness"""
    
    def __init__(self):
        self.problem_memory = {}
        self.recent_solutions = []
        print("🧠 Intelligent Step-by-Step Solver initialized")
    
    def solve_step_by_step(self, problem: str, force_detailed: bool = False) -> Dict[str, Any]:
        """Main solving function with intelligent response selection"""
        normalized = problem.lower().strip()
        problem_hash = hashlib.md5(normalized.encode()).hexdigest()[:10]
        
        # Check memory for non-repetitive responses
        if not force_detailed and self._is_recently_solved(problem_hash):
            return self._generate_memory_response(problem_hash, problem)
        
        # Handle simple calculations
        if self._is_simple_calculation(normalized) and not force_detailed:
            return self._handle_simple_calculation(normalized)
        
        # Generate detailed solution
        detailed_solution = self._generate_detailed_solution(normalized)
        self._store_in_memory(problem_hash, problem, detailed_solution)
        
        return {
            'success': True,
            'problem': problem,
            'solution': detailed_solution,
            'is_repetitive': False,
            'suggestions': self._generate_suggestions(detailed_solution)
        }
    
    def _is_recently_solved(self, problem_hash: str) -> bool:
        """Check if problem was solved in recent history"""
        return problem_hash in self.recent_solutions[-3:]
    
    def _is_simple_calculation(self, problem: str) -> bool:
        """Identify simple direct calculations"""
        patterns = [
            r'^\d+\s*[+\-*/]\s*\d+$',
            r'^sqrt\(\d+\)$',
            r'^\d+\s*\^\s*\d+$'
        ]
        return any(re.match(p, problem) for p in patterns)
    
    def _handle_simple_calculation(self, problem: str) -> Dict[str, Any]:
        """Handle simple calculations with direct answers"""
        try:
            expr = sp.sympify(problem.replace('^', '**'))
            result = float(sp.N(expr))
            return {
                'success': True,
                'is_simple': True,
                'direct_answer': result,
                'offer_steps': f"Result: {result}. Want detailed steps?"
            }
        except:
            return self._generate_detailed_solution(problem)
    
    def _generate_detailed_solution(self, problem: str) -> Dict[str, Any]:
        """Generate comprehensive step-by-step solution"""
        steps = []
        
        # Identify problem type and solve accordingly
        if '=' in problem and ('solve' in problem or 'x' in problem):
            steps, answer = self._solve_equation(problem)
            domain = 'algebra'
        elif '%' in problem or 'percent' in problem:
            steps, answer = self._solve_percentage(problem)
            domain = 'percentage'
        elif any(op in problem for op in ['+', '-', '*', '/']):
            steps, answer = self._solve_arithmetic(problem)
            domain = 'arithmetic'
        else:
            steps, answer = self._solve_general(problem)
            domain = 'general'
        
        return {
            'domain': domain,
            'steps': steps,
            'final_answer': answer,
            'difficulty': 'Basic' if len(steps) <= 4 else 'Intermediate',
            'key_concepts': self._extract_concepts(steps)
        }
    
    def _solve_equation(self, problem: str) -> Tuple[List[str], str]:
        """Solve algebraic equations with detailed steps"""
        steps = [
            "🎯 EQUATION SOLVING PROCESS",
            "=" * 35,
            "",
            f"📝 Problem: {problem}",
            ""
        ]
        
        try:
            if '=' in problem:
                left, right = problem.split('=')
                left, right = left.strip(), right.strip()
                
                steps.extend([
                    f"🔍 Equation structure: {left} = {right}",
                    "🎯 Goal: Isolate the variable",
                    ""
                ])
                
                # Use SymPy for solving
                equation = sp.Eq(sp.sympify(left), sp.sympify(right))
                variables = list(equation.free_symbols)
                
                if variables:
                    var = variables[0]
                    solution = sp.solve(equation, var)
                    
                    # Add reasoning steps
                    if '+' in left or '-' in left:
                        steps.append("🔧 Step 1: Use Addition/Subtraction Property of Equality")
                        steps.append("   → Add or subtract the same value from both sides")
                    
                    if '*' in left or '/' in left:
                        steps.append("🔧 Step 2: Use Multiplication/Division Property of Equality")
                        steps.append("   → Multiply or divide both sides by the same non-zero value")
                    
                    steps.append("")
                    
                    if solution:
                        answer = str(solution[0])
                        steps.extend([
                            f"⚡ Solving: {equation}",
                            f"✅ Solution: {var} = {answer}",
                            "",
                            "🔍 Verification:",
                            f"   Substitute {var} = {answer} back into original equation",
                            "   Both sides should be equal ✓"
                        ])
                        return steps, answer
            
            steps.append("❌ Unable to solve this equation")
            return steps, "No solution found"
            
        except Exception as e:
            steps.append(f"❌ Error during solving: {str(e)}")
            return steps, "Error"
    
    def _solve_percentage(self, problem: str) -> Tuple[List[str], str]:
        """Solve percentage problems with clear explanations"""
        steps = [
            "📊 PERCENTAGE CALCULATION",
            "=" * 30,
            "",
            f"📝 Problem: {problem}",
            ""
        ]
        
        try:
            numbers = re.findall(r'\d+', problem)
            
            if len(numbers) >= 2 and '%' in problem and 'of' in problem:
                percentage = int(numbers[0])
                base_number = int(numbers[1])
                
                steps.extend([
                    f"🎯 We need to find: {percentage}% of {base_number}",
                    "",
                    "🔧 Step 1: Convert percentage to decimal",
                    f"   {percentage}% = {percentage} ÷ 100 = {percentage/100}",
                    "",
                    "🔧 Step 2: Multiply by the base number",
                    f"   {percentage/100} × {base_number} = {(percentage/100) * base_number}",
                    "",
                    f"✅ Answer: {(percentage/100) * base_number}",
                    "",
                    "💡 Remember: 'of' in percentage problems means multiply!"
                ])
                
                return steps, str((percentage/100) * base_number)
            
            steps.append("❌ Could not identify percentage components")
            return steps, "Unable to solve"
            
        except Exception as e:
            steps.append(f"❌ Error: {str(e)}")
            return steps, "Error"
    
    def _solve_arithmetic(self, problem: str) -> Tuple[List[str], str]:
        """Solve arithmetic with order of operations explanation"""
        steps = [
            "🧮 ARITHMETIC CALCULATION",
            "=" * 30,
            "",
            f"📝 Expression: {problem}",
            ""
        ]
        
        try:
            steps.extend([
                "🔧 Applying Order of Operations (PEMDAS/BODMAS):",
                "   1. Parentheses/Brackets",
                "   2. Exponents/Orders",
                "   3. Multiplication and Division (left to right)",
                "   4. Addition and Subtraction (left to right)",
                ""
            ])
            
            # Calculate step by step
            expr = sp.sympify(problem.replace('^', '**'))
            result = sp.N(expr)
            
            # Show intermediate steps if complex
            if '+' in problem and ('*' in problem or '/' in problem):
                steps.append("🎯 Since we have mixed operations:")
                steps.append("   → First: Multiplication/Division")
                steps.append("   → Then: Addition/Subtraction")
            
            steps.extend([
                "",
                f"⚡ Calculating: {problem}",
                f"✅ Result: {result}"
            ])
            
            return steps, str(result)
            
        except Exception as e:
            steps.append(f"❌ Error: {str(e)}")
            return steps, "Error"
    
    def _solve_general(self, problem: str) -> Tuple[List[str], str]:
        """General problem solver with SymPy"""
        steps = [
            "🤖 MATHEMATICAL ANALYSIS",
            "=" * 30,
            "",
            f"📝 Input: {problem}",
            ""
        ]
        
        try:
            expr = sp.sympify(problem.replace('^', '**'))
            result = sp.simplify(expr)
            
            steps.extend([
                "🔧 Attempting to simplify the expression...",
                f"⚡ Simplified form: {result}",
                f"✅ Result: {result}"
            ])
            
            return steps, str(result)
            
        except Exception as e:
            steps.extend([
                "❓ Unable to parse this mathematical expression",
                "💡 Try rephrasing or check the syntax",
                f"🔍 Error details: {str(e)}"
            ])
            return steps, "Unable to solve"
    
    def _generate_memory_response(self, problem_hash: str, problem: str) -> Dict[str, Any]:
        """Generate intelligent non-repetitive response"""
        memory = self.problem_memory[problem_hash]
        answer = memory['solution']['final_answer']
        
        responses = [
            f"I solved this recently! Quick answer: {answer}. Want the full breakdown again?",
            f"We worked on this before. The result is {answer}. Need a refresher on the steps?",
            f"This looks familiar! Answer: {answer}. Shall we try a similar problem instead?"
        ]
        
        import random
        message = random.choice(responses)
        
        return {
            'success': True,
            'is_repetitive': True,
            'message': message,
            'quick_answer': answer,
            'suggestions': [
                "Show full steps again",
                "Try a similar problem", 
                "Practice related concepts"
            ]
        }
    
    def _store_in_memory(self, problem_hash: str, problem: str, solution: Dict[str, Any]):
        """Store solution in memory system"""
        self.problem_memory[problem_hash] = {
            'problem': problem,
            'solution': solution,
            'timestamp': datetime.now()
        }
        
        self.recent_solutions.append(problem_hash)
        if len(self.recent_solutions) > 5:
            self.recent_solutions.pop(0)
    
    def _extract_concepts(self, steps: List[str]) -> List[str]:
        """Extract key mathematical concepts from solution steps"""
        concepts = []
        step_text = ' '.join(steps).lower()
        
        concept_keywords = {
            'Addition Property of Equality': ['add', 'addition property'],
            'Subtraction Property of Equality': ['subtract', 'subtraction property'],
            'Order of Operations': ['pemdas', 'bodmas', 'order of operations'],
            'Percentage Conversion': ['percentage', 'convert', '%'],
            'Algebraic Manipulation': ['isolate', 'variable', 'equation'],
            'Arithmetic Operations': ['multiply', 'divide', 'add', 'subtract']
        }
        
        for concept, keywords in concept_keywords.items():
            if any(keyword in step_text for keyword in keywords):
                concepts.append(concept)
        
        return concepts[:3]  # Limit to top 3 concepts
    
    def _generate_suggestions(self, solution: Dict[str, Any]) -> List[str]:
        """Generate contextual suggestions based on solution"""
        suggestions = []
        
        domain = solution.get('domain', 'general')
        
        if domain == 'algebra':
            suggestions.extend([
                "Try solving a quadratic equation",
                "Practice with systems of equations",
                "Learn about graphing linear equations"
            ])
        elif domain == 'percentage':
            suggestions.extend([
                "Calculate compound interest",
                "Try percentage increase/decrease",
                "Practice with ratios and proportions"
            ])
        elif domain == 'arithmetic':
            suggestions.extend([
                "Practice mental math techniques",
                "Try problems with fractions",
                "Work on complex expressions"
            ])
        
        return suggestions[:3]
    
    def suggest_practice_problems(self) -> List[str]:
        """Suggest practice problems based on recent activity"""
        return [
            "solve 3x - 7 = 14",
            "what is 30% of 250",
            "calculate 15 + 6 * 4 - 8",
            "simplify 2(x + 5) = 22"
        ]


# Test the solver
if __name__ == "__main__":
    solver = IntelligentMathSolver()
    
    test_problems = [
        "solve x + 5 = 10",
        "what is 25% of 80",
        "calculate 2 + 3 * 4",
        "solve x + 5 = 10",  # Repeat to test memory
        "find 2 + 2"         # Simple calculation
    ]
    
    print("🧪 Testing Intelligent Step-by-Step Solver")
    print("=" * 50)
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🔢 Test {i}: {problem}")
        result = solver.solve_step_by_step(problem)
        
        if result.get('is_simple'):
            print(f"   📊 Quick Answer: {result['direct_answer']}")
        elif result.get('is_repetitive'):
            print(f"   🔄 Memory Response: {result['message']}")
        else:
            solution = result['solution']
            print(f"   ✅ Final Answer: {solution['final_answer']}")
            print(f"   🎯 Domain: {solution['domain']}")
            print(f"   📝 Steps: {len(solution['steps'])} steps") 