# step_by_step_solver.py - Step-by-Step Mathematical Solver
# SAUNet 4.0 - Intelligent Step-by-Step Problem Solver

"""
Step-by-Step Mathematical Solver for SAUNet 4.0
Provides detailed solutions with memory to avoid repetitive explanations
"""

import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sympy as sp


class StepByStepr:
    """Intelligent step-by-step mathematical solver with memory"""
    
    def __init__(self):
        self.problem_memory = {}  # Store recently solved problems
        self.recent_solutions = []  # Last 5 problem hashes
        
        print("🧠 Step-by-Step Solver initialized")
    
    def solve_with_steps(self, problem: str, force_detailed: bool = False) -> Dict[str, Any]:
        """
        Solve mathematical problem with detailed steps
        
        Args:
            problem: Mathematical problem string
            force_detailed: Force detailed steps even for repeated problems
            
        Returns:
            Dictionary with solution and steps
        """
        # Normalize and hash problem
        normalized = self._normalize_problem(problem)
        problem_hash = hashlib.md5(normalized.encode()).hexdigest()[:10]
        
        # Check for recent solutions
        if not force_detailed and self._is_recent(problem_hash):
            return self._get_non_repetitive_response(problem_hash, problem)
        
        # Check if simple calculation
        if self._is_simple_calc(normalized) and not force_detailed:
            return self._handle_simple_calc(normalized)
        
        # Generate detailed solution
        solution = self._generate_solution(normalized)
        
        # Store in memory
        self._store_memory(problem_hash, problem, solution)
        
        return {
            'success': True,
            'problem': problem,
            'solution': solution,
            'is_repetitive': False
        }
    
    def _normalize_problem(self, problem: str) -> str:
        """Normalize problem text"""
        normalized = problem.lower().strip()
        # Replace common variations
        replacements = {
            'what is': 'find',
            'solve for': 'solve', 
            'calculate': 'find',
            '×': '*',
            '÷': '/',
            '%': '/100'
        }
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized
    
    def _is_recent(self, problem_hash: str) -> bool:
        """Check if problem was solved recently"""
        return problem_hash in self.recent_solutions[-3:]
    
    def _is_simple_calc(self, problem: str) -> bool:
        """Check if problem is a simple calculation"""
        simple_patterns = [
            r'^\d+\s*[+\-*/]\s*\d+$',  # 2 + 3
            r'^sqrt\(\d+\)$',           # sqrt(16)
            r'^\d+\s*\^\s*\d+$'        # 2^3
        ]
        return any(re.match(pattern, problem) for pattern in simple_patterns)
    
    def _handle_simple_calc(self, problem: str) -> Dict[str, Any]:
        """Handle simple calculations"""
        try:
            expr = sp.sympify(problem.replace('^', '**'))
            result = float(sp.N(expr))
            
            return {
                'success': True,
                'problem': problem,
                'direct_answer': result,
                'is_simple': True,
                'offer_steps': f"The answer is {result}. Need detailed steps?"
            }
        except:
            return self._generate_solution(problem)
    
    def _generate_solution(self, problem: str) -> Dict[str, Any]:
        """Generate detailed step-by-step solution"""
        steps = []
        step_num = 1
        
        # Identify problem type
        if '=' in problem and ('solve' in problem or 'x' in problem):
            # Equation solving
            steps, final_answer = self._solve_equation(problem)
            problem_type = 'equation'
        elif any(op in problem for op in ['+', '-', '*', '/']):
            # Arithmetic
            steps, final_answer = self._solve_arithmetic(problem)
            problem_type = 'arithmetic'
        elif '%' in problem or 'percent' in problem:
            # Percentage
            steps, final_answer = self._solve_percentage(problem)
            problem_type = 'percentage'
        else:
            # General
            steps, final_answer = self._solve_general(problem)
            problem_type = 'general'
        
        return {
            'problem_type': problem_type,
            'steps': steps,
            'final_answer': final_answer,
            'difficulty': 'Basic' if len(steps) <= 3 else 'Intermediate'
        }
    
    def _solve_equation(self, problem: str) -> Tuple[List[str], str]:
        """Solve equations with steps"""
        steps = ["🎯 EQUATION SOLVING", "=" * 25, ""]
        
        try:
            if '=' in problem:
                left, right = problem.split('=')
                left, right = left.strip(), right.strip()
                
                steps.append(f"📝 Original equation: {left} = {right}")
                
                # Use SymPy to solve
                equation = sp.Eq(sp.sympify(left), sp.sympify(right))
                variables = list(equation.free_symbols)
                
                if variables:
                    var = variables[0]
                    solution = sp.solve(equation, var)
                    
                    steps.append(f"🔧 Isolating {var}...")
                    
                    if '+' in left or '-' in left:
                        steps.append("   → Subtract/add terms to both sides")
                    if '*' in left or '/' in left:
                        steps.append("   → Multiply/divide both sides")
                    
                    if solution:
                        answer = str(solution[0])
                        steps.append(f"✅ Solution: {var} = {answer}")
                        return steps, answer
            
            return steps + ["❌ Could not solve equation"], "No solution"
            
        except Exception as e:
            return steps + [f"❌ Error: {e}"], "Error"
    
    def _solve_arithmetic(self, problem: str) -> Tuple[List[str], str]:
        """Solve arithmetic with steps"""
        steps = ["🧮 ARITHMETIC CALCULATION", "=" * 30, ""]
        
        try:
            steps.append(f"📝 Expression: {problem}")
            
            # Apply order of operations
            steps.append("🔧 Applying order of operations (PEMDAS):")
            steps.append("   → Parentheses, Exponents, Multiplication/Division, Addition/Subtraction")
            
            # Calculate result
            expr = sp.sympify(problem.replace('^', '**'))
            result = sp.N(expr)
            
            steps.append(f"⚡ Calculating: {problem} = {result}")
            steps.append(f"✅ Result: {result}")
            
            return steps, str(result)
            
        except Exception as e:
            return steps + [f"❌ Error: {e}"], "Error"
    
    def _solve_percentage(self, problem: str) -> Tuple[List[str], str]:
        """Solve percentage problems with steps"""
        steps = ["📊 PERCENTAGE CALCULATION", "=" * 30, ""]
        
        try:
            numbers = re.findall(r'\d+', problem)
            
            if len(numbers) >= 2 and '%' in problem and 'of' in problem:
                percentage = int(numbers[0])
                base = int(numbers[1])
                
                steps.append(f"📝 Problem: {percentage}% of {base}")
                steps.append(f"🔧 Convert percentage to decimal: {percentage}% = {percentage}/100 = {percentage/100}")
                
                result = (percentage / 100) * base
                steps.append(f"⚡ Multiply: {percentage/100} × {base} = {result}")
                steps.append(f"✅ Answer: {result}")
                
                return steps, str(result)
            
            return steps + ["❌ Could not parse percentage problem"], "Error"
            
        except Exception as e:
            return steps + [f"❌ Error: {e}"], "Error"
    
    def _solve_general(self, problem: str) -> Tuple[List[str], str]:
        """General problem solving"""
        steps = ["🤖 GENERAL ANALYSIS", "=" * 25, ""]
        
        try:
            steps.append(f"📝 Problem: {problem}")
            
            # Try to simplify with SymPy
            expr = sp.sympify(problem.replace('^', '**'))
            result = sp.simplify(expr)
            
            steps.append(f"🔧 Simplifying expression...")
            steps.append(f"✅ Result: {result}")
            
            return steps, str(result)
            
        except Exception as e:
            steps.append("❓ Unable to determine specific operation")
            steps.append("💡 Try rephrasing or providing clearer input")
            return steps, "Unable to solve"
    
    def _get_non_repetitive_response(self, problem_hash: str, problem: str) -> Dict[str, Any]:
        """Generate non-repetitive response for recent problems"""
        memory = self.problem_memory[problem_hash]
        answer = memory['solution']['final_answer']
        
        messages = [
            f"You asked this recently. Quick answer: {answer}. Want full steps again?",
            f"I solved this before. The result is {answer}. Need a detailed breakdown?",
            f"This looks familiar! Answer: {answer}. Try a different approach?"
        ]
        
        import random
        message = random.choice(messages)
        
        return {
            'success': True,
            'problem': problem,
            'is_repetitive': True,
            'message': message,
            'quick_answer': answer,
            'suggestions': ["Show full steps", "Try similar problem", "Practice more"]
        }
    
    def _store_memory(self, problem_hash: str, problem: str, solution: Dict[str, Any]):
        """Store solution in memory"""
        self.problem_memory[problem_hash] = {
            'problem': problem,
            'solution': solution,
            'timestamp': datetime.now()
        }
        
        self.recent_solutions.append(problem_hash)
        if len(self.recent_solutions) > 5:
            self.recent_solutions.pop(0)
    
    def get_suggestions(self) -> List[str]:
        """Get practice suggestions based on recent problems"""
        suggestions = [
            "solve 2x + 5 = 15",
            "what is 20% of 150", 
            "calculate 12 + 8 * 3",
            "find 144 / 12 + 7"
        ]
        return suggestions[:3]


# Test function
def test_step_solver():
    """Test the step-by-step solver"""
    solver = StepByStepr()
    
    problems = [
        "solve x + 5 = 10",
        "what is 15% of 200", 
        "calculate 2 + 3 * 4",
        "solve x + 5 = 10",  # Repeat
        "find 2 + 2"         # Simple
    ]
    
    print("🧪 Testing Step-by-Step Solver")
    print("=" * 40)
    
    for i, problem in enumerate(problems, 1):
        print(f"\n🔢 Test {i}: {problem}")
        result = solver.solve_with_steps(problem)
        
        if result['success']:
            if result.get('is_simple'):
                print(f"   📊 Quick: {result['direct_answer']}")
            elif result.get('is_repetitive'):
                print(f"   🔄 Memory: {result['message']}")
            else:
                solution = result['solution']
                print(f"   ✅ Answer: {solution['final_answer']}")
                print(f"   🎯 Type: {solution['problem_type']}")
                print(f"   📝 Steps: {len(solution['steps'])}")


if __name__ == "__main__":
    test_step_solver() 