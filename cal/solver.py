import sympy as sp
from sympy import symbols, solve, Eq, diff, integrate, Matrix, latex
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import re

class EquationSolver:
    def __init__(self):
        """Initialize the equation solver with common variables."""
        # Common symbols
        self.x, self.y, self.z = sp.symbols('x y z')
        
    def solve_equation(self, equation_str, variable_str=None):
        """
        Solve an equation for a specified variable.
        
        Args:
            equation_str (str): The equation string, e.g., "x^2 - 5*x + 6 = 0"
            variable_str (str): The variable to solve for, e.g., "x"
            
        Returns:
            tuple: (solutions, steps)
        """
        try:
            # Parse the equation
            if '=' in equation_str:
                left_str, right_str = equation_str.split('=')
                left = parse_expr(left_str.strip())
                right = parse_expr(right_str.strip())
                expr = left - right
            else:
                expr = parse_expr(equation_str)
            
            # Determine the variable to solve for
            if variable_str:
                var = sp.symbols(variable_str)
            else:
                # Try to detect the variable from the equation
                vars_in_expr = list(expr.free_symbols)
                if len(vars_in_expr) == 0:
                    return [], "No variables found in the equation."
                var = vars_in_expr[0]  # Use the first variable found
            
            # Solve the equation
            solutions = sp.solve(expr, var)
            
            # Generate step-by-step solution
            steps = self._generate_equation_steps(expr, var, solutions)
            
            return solutions, steps
        except Exception as e:
            return [], f"Error solving equation: {str(e)}"
    
    def _generate_equation_steps(self, expr, var, solutions):
        """Generate step-by-step explanation for solving an equation."""
        steps = []
        steps.append(f"Equation: {expr} = 0")
        
        # Simplify if possible
        simplified = sp.expand(expr)
        if simplified != expr:
            steps.append(f"Expanding the equation: {simplified} = 0")
            expr = simplified
        
        # Check the polynomial degree
        try:
            degree = sp.degree(expr, var)
            steps.append(f"This is a polynomial equation of degree {degree} in {var}.")
            
            if degree == 1:
                # Linear equation: ax + b = 0
                a = expr.coeff(var, 1)
                b = expr.subs(var, 0)
                steps.append(f"Linear equation in the form a·{var} + b = 0")
                steps.append(f"where a = {a} and b = {b}")
                steps.append(f"Solving for {var}: {var} = -b/a = {-b}/{a} = {-b/a}")
                
            elif degree == 2:
                # Quadratic equation: ax² + bx + c = 0
                a = expr.coeff(var, 2)
                b = expr.coeff(var, 1)
                c = expr.subs(var, 0)
                
                steps.append(f"Quadratic equation in the form a·{var}² + b·{var} + c = 0")
                steps.append(f"where a = {a}, b = {b}, and c = {c}")
                
                # Try factoring first
                factors = sp.factor(expr)
                if factors != expr:
                    steps.append(f"Factoring the equation: {factors} = 0")
                    steps.append(f"Using the zero-product property: if A·B = 0, then either A = 0 or B = 0")
                    
                    if isinstance(factors, sp.Mul):
                        for i, factor in enumerate(factors.args, start=1):
                            if var in factor.free_symbols:
                                steps.append(f"Factor {i}: {factor} = 0")
                                sol = sp.solve(factor, var)
                                steps.append(f"Solution from factor {i}: {var} = {sol}")
                else:
                    # Use quadratic formula
                    discriminant = b**2 - 4*a*c
                    steps.append(f"Using the quadratic formula: {var} = (-b ± √(b² - 4ac)) / (2a)")
                    steps.append(f"Discriminant = b² - 4ac = {b}² - 4·{a}·{c} = {discriminant}")
                    
                    if discriminant > 0:
                        steps.append(f"Since the discriminant is positive, there are two real solutions:")
                        steps.append(f"{var}₁ = (-{b} + √{discriminant}) / (2·{a}) = {(-b + sp.sqrt(discriminant))/(2*a)}")
                        steps.append(f"{var}₂ = (-{b} - √{discriminant}) / (2·{a}) = {(-b - sp.sqrt(discriminant))/(2*a)}")
                    elif discriminant == 0:
                        steps.append(f"Since the discriminant is zero, there is one repeated solution:")
                        steps.append(f"{var} = -b / (2a) = -{b} / (2·{a}) = {-b/(2*a)}")
                    else:
                        steps.append(f"Since the discriminant is negative, there are two complex solutions.")
            
        except Exception as e:
            steps.append(f"(Error in polynomial analysis: {str(e)})")
        
        # Show the final solutions
        steps.append("\nFinal solution(s):")
        if solutions:
            for i, sol in enumerate(solutions, start=1):
                steps.append(f"  {var} = {sol}")
        else:
            steps.append("  No solutions found")
        
        return "\n".join(steps)
    
    def differentiate(self, expr_str, var_str='x', order=1):
        """
        Differentiate an expression with respect to a variable.
        
        Args:
            expr_str (str): The expression to differentiate
            var_str (str): The variable to differentiate with respect to
            order (int): The order of differentiation
            
        Returns:
            tuple: (derivative, steps)
        """
        try:
            # Clean the expression
            expr_str = expr_str.replace('^', '**')
            
            # Parse the expression
            try:
                expr = parse_expr(expr_str)
            except SyntaxError:
                return None, "Invalid syntax in expression. Please check your input."
            except Exception as e:
                return None, f"Error parsing expression: {str(e)}"
                
            var = sp.symbols(var_str)
            
            # Calculate the derivative
            derivative = sp.diff(expr, var, order)
            
            # Generate step-by-step solution
            steps = self._generate_differentiation_steps(expr, var, order, derivative)
            
            return derivative, steps
        except Exception as e:
            return None, f"Error calculating derivative: {str(e)}"
    
    def _generate_differentiation_steps(self, expr, var, order, derivative):
        """Generate step-by-step explanation for differentiation."""
        steps = []
        steps.append(f"Original expression: {expr}")
        steps.append(f"Taking the {self._ordinal(order)} derivative with respect to {var}")
        
        # For first order derivatives, show the rules used
        if order == 1:
            # Check for sums/differences
            if isinstance(expr, sp.Add):
                steps.append("\nUsing the sum/difference rule: d/dx[f(x) ± g(x)] = d/dx[f(x)] ± d/dx[g(x)]")
                for term in expr.args:
                    steps.append(f"  d/d{var}[{term}] = {sp.diff(term, var)}")
            
            # Check for products
            if isinstance(expr, sp.Mul) and len(expr.args) > 1:
                steps.append("\nUsing the product rule: d/dx[f(x)·g(x)] = f(x)·g'(x) + g(x)·f'(x)")
                
                # Split into factors for demonstration
                factors = list(expr.args)
                if len(factors) == 2:
                    f, g = factors
                    steps.append(f"  Let f({var}) = {f} and g({var}) = {g}")
                    steps.append(f"  f'({var}) = {sp.diff(f, var)}")
                    steps.append(f"  g'({var}) = {sp.diff(g, var)}")
                    steps.append(f"  f({var})·g'({var}) + g({var})·f'({var}) = {f}·{sp.diff(g, var)} + {g}·{sp.diff(f, var)}")
            
            # Check for powers
            if isinstance(expr, sp.Pow) and expr.args[1] != -1:
                base, exponent = expr.args
                if base == var:
                    steps.append(f"\nUsing the power rule: d/dx[x^n] = n·x^(n-1)")
                    steps.append(f"  d/d{var}[{var}^{exponent}] = {exponent}·{var}^{exponent-1} = {derivative}")
        
        # Show final result
        steps.append(f"\nFinal result: {derivative}")
        return "\n".join(steps)
    
    def integrate(self, expr_str, var_str='x', lower=None, upper=None):
        """
        Integrate an expression with respect to a variable.
        
        Args:
            expr_str (str): The expression to integrate
            var_str (str): The variable to integrate with respect to
            lower (str): Lower bound for definite integral (optional)
            upper (str): Upper bound for definite integral (optional)
            
        Returns:
            tuple: (integral, steps)
        """
        try:
            # Clean the expression
            expr_str = expr_str.replace('^', '**')
            
            # Parse the expression
            try:
                expr = parse_expr(expr_str)
            except SyntaxError:
                return None, "Invalid syntax in expression. Please check your input."
            except Exception as e:
                return None, f"Error parsing expression: {str(e)}"
                
            var = sp.symbols(var_str)
            
            # Calculate the integral
            if lower is not None and upper is not None:
                # Definite integral
                try:
                    lower_bound = parse_expr(str(lower))
                    upper_bound = parse_expr(str(upper))
                except SyntaxError:
                    return None, "Invalid syntax in integration limits. Please check your input."
                except Exception as e:
                    return None, f"Error parsing integration limits: {str(e)}"
                    
                integral = sp.integrate(expr, (var, lower_bound, upper_bound))
                is_definite = True
            else:
                # Indefinite integral
                integral = sp.integrate(expr, var)
                is_definite = False
            
            # Generate step-by-step solution
            steps = self._generate_integration_steps(expr, var, integral, lower, upper, is_definite)
            
            return integral, steps
        except Exception as e:
            return None, f"Error calculating integral: {str(e)}"
    
    def _generate_integration_steps(self, expr, var, integral, lower, upper, is_definite):
        """Generate step-by-step explanation for integration."""
        steps = []
        
        if is_definite:
            steps.append(f"Calculating the definite integral: ∫({expr}) d{var} from {lower} to {upper}")
        else:
            steps.append(f"Calculating the indefinite integral: ∫({expr}) d{var}")
        
        # Check for sums/differences
        if isinstance(expr, sp.Add):
            steps.append("\nUsing the sum/difference rule: ∫[f(x) ± g(x)] dx = ∫f(x) dx ± ∫g(x) dx")
            for term in expr.args:
                steps.append(f"  ∫({term}) d{var} = {sp.integrate(term, var)} + C")
        
        # Check for common integrals
        if isinstance(expr, sp.Pow) and expr.args[0] == var:
            exponent = expr.args[1]
            if exponent != -1:
                steps.append(f"\nUsing the power rule: ∫x^n dx = x^(n+1)/(n+1) + C for n ≠ -1")
                steps.append(f"  ∫({var}^{exponent}) d{var} = {var}^({exponent+1})/({exponent+1}) = {sp.integrate(expr, var)}")
            else:
                steps.append(f"\nUsing the logarithm rule: ∫(1/x) dx = ln|x| + C")
                steps.append(f"  ∫(1/{var}) d{var} = ln|{var}| = {sp.integrate(expr, var)}")
        
        # Show the final result
        if not is_definite:
            steps.append(f"\nFinal result: {integral} + C")
        else:
            steps.append(f"\nFinal result: {integral}")
        
        return "\n".join(steps)
    
    def matrix_operations(self, matrix_a_str, operation, matrix_b_str=None):
        """
        Perform matrix operations and provide step-by-step explanations.
        
        Args:
            matrix_a_str (str): String representation of matrix A
            operation (str): Operation to perform (add, multiply, inverse, determinant, etc.)
            matrix_b_str (str): String representation of matrix B (for binary operations)
            
        Returns:
            tuple: (result, steps)
        """
        try:
            # Parse matrix strings
            matrix_a = self._parse_matrix(matrix_a_str)
            matrix_b = self._parse_matrix(matrix_b_str) if matrix_b_str else None
            
            # Perform the operation
            if operation == 'add':
                if matrix_b is None:
                    return None, "Second matrix required for addition"
                result = matrix_a + matrix_b
                steps = self._generate_matrix_addition_steps(matrix_a, matrix_b, result)
            
            elif operation == 'multiply':
                if matrix_b is None:
                    return None, "Second matrix required for multiplication"
                result = matrix_a * matrix_b
                steps = self._generate_matrix_multiplication_steps(matrix_a, matrix_b, result)
            
            elif operation == 'determinant':
                result = matrix_a.det()
                steps = self._generate_matrix_determinant_steps(matrix_a, result)
            
            elif operation == 'inverse':
                if matrix_a.det() == 0:
                    return None, "Matrix is singular, no inverse exists"
                result = matrix_a.inv()
                steps = self._generate_matrix_inverse_steps(matrix_a, result)
            
            elif operation == 'transpose':
                result = matrix_a.transpose()
                steps = self._generate_matrix_transpose_steps(matrix_a, result)
            
            elif operation == 'eigenvalues':
                result = matrix_a.eigenvals()
                steps = self._generate_eigenvalues_steps(matrix_a, result)
            
            else:
                return None, f"Unknown operation: {operation}"
            
            return result, steps
        except Exception as e:
            return None, f"Error in matrix operation: {str(e)}"
    
    def _parse_matrix(self, matrix_str):
        """Parse a string representation into a sympy Matrix."""
        if matrix_str is None:
            return None
            
        try:
            # Check if it's a Python-style nested list representation
            if matrix_str.strip().startswith('[[') and matrix_str.strip().endswith(']]'):
                # Use Python's ast module to safely evaluate the nested list
                import ast
                matrix_data = ast.literal_eval(matrix_str)
                return sp.Matrix(matrix_data)
            
            # Clean up the string
            matrix_str = matrix_str.strip()
            
            # Check if it uses semicolons to separate rows
            if ';' in matrix_str:
                rows = []
                for row_str in matrix_str.split(';'):
                    # Remove brackets if present
                    row_str = row_str.strip()
                    if row_str.startswith('['):
                        row_str = row_str[1:]
                    if row_str.endswith(']'):
                        row_str = row_str[:-1]
                    
                    # Parse elements
                    elements = []
                    for elem_str in row_str.split(','):
                        elem_str = elem_str.strip()
                        try:
                            elem = float(elem_str) if '.' in elem_str else int(elem_str)
                        except ValueError:
                            elem = sp.Symbol(elem_str)
                        elements.append(elem)
                    
                    rows.append(elements)
                
                return sp.Matrix(rows)
            
            # Assume it's space-separated
            rows = []
            for row_str in matrix_str.split('\n'):
                row_str = row_str.strip()
                if not row_str:
                    continue
                    
                elements = []
                for elem_str in row_str.split():
                    elem_str = elem_str.strip()
                    try:
                        elem = float(elem_str) if '.' in elem_str else int(elem_str)
                    except ValueError:
                        elem = sp.Symbol(elem_str)
                    elements.append(elem)
                
                if elements:
                    rows.append(elements)
            
            return sp.Matrix(rows)
        
        except Exception as e:
            raise ValueError(f"Failed to parse matrix: {str(e)}")
    
    def _generate_matrix_addition_steps(self, matrix_a, matrix_b, result):
        """Generate step-by-step explanation for matrix addition."""
        steps = []
        steps.append("Matrix Addition:")
        steps.append(f"A = {matrix_a}")
        steps.append(f"B = {matrix_b}")
        
        if matrix_a.shape != matrix_b.shape:
            steps.append(f"Error: Cannot add matrices of different sizes ({matrix_a.shape} and {matrix_b.shape})")
            return "\n".join(steps)
        
        steps.append("\nAdding corresponding elements:")
        
        # For small matrices, show element-by-element addition
        if matrix_a.shape[0] <= 4 and matrix_a.shape[1] <= 4:
            element_steps = []
            for i in range(matrix_a.shape[0]):
                for j in range(matrix_a.shape[1]):
                    element_steps.append(f"  C[{i+1},{j+1}] = A[{i+1},{j+1}] + B[{i+1},{j+1}] = {matrix_a[i,j]} + {matrix_b[i,j]} = {result[i,j]}")
            steps.append("\n".join(element_steps))
        
        steps.append(f"\nResult = {result}")
        return "\n".join(steps)
    
    def _generate_matrix_multiplication_steps(self, matrix_a, matrix_b, result):
        """Generate step-by-step explanation for matrix multiplication."""
        steps = []
        steps.append("Matrix Multiplication:")
        steps.append(f"A = {matrix_a}")
        steps.append(f"B = {matrix_b}")
        
        if matrix_a.shape[1] != matrix_b.shape[0]:
            steps.append(f"Error: Cannot multiply matrices with incompatible dimensions")
            steps.append(f"Matrix A has size {matrix_a.shape} and Matrix B has size {matrix_b.shape}")
            steps.append(f"For multiplication, the number of columns in A must equal the number of rows in B")
            return "\n".join(steps)
        
        steps.append(f"\nCalculating each element of the product C = A×B")
        
        # For small matrices, show calculation for each element
        if matrix_a.shape[0] <= 3 and matrix_b.shape[1] <= 3:
            element_steps = []
            for i in range(matrix_a.shape[0]):
                for j in range(matrix_b.shape[1]):
                    calc = " + ".join(f"{matrix_a[i,k]}·{matrix_b[k,j]}" for k in range(matrix_a.shape[1]))
                    element_steps.append(f"  C[{i+1},{j+1}] = {calc} = {result[i,j]}")
            steps.append("\n".join(element_steps))
        else:
            steps.append("  Each element C[i,j] is the dot product of row i from A with column j from B")
        
        steps.append(f"\nResult = {result}")
        return "\n".join(steps)
    
    def _generate_matrix_determinant_steps(self, matrix, result):
        """Generate step-by-step explanation for determinant calculation."""
        steps = []
        steps.append(f"Calculating determinant of matrix:\n{matrix}")
        
        if matrix.shape[0] != matrix.shape[1]:
            steps.append("Error: Determinant is only defined for square matrices")
            return "\n".join(steps)
        
        size = matrix.shape[0]
        
        if size == 1:
            steps.append(f"For a 1×1 matrix, the determinant is simply the single element: {matrix[0,0]}")
        
        elif size == 2:
            a, b = matrix[0,0], matrix[0,1]
            c, d = matrix[1,0], matrix[1,1]
            steps.append(f"For a 2×2 matrix, the determinant is calculated as:")
            steps.append(f"det(A) = a·d - b·c = {a}·{d} - {b}·{c} = {a*d} - {b*c} = {result}")
        
        elif size == 3:
            steps.append("For a 3×3 matrix, we'll use the cofactor expansion along the first row:")
            
            # Cofactor expansion along first row
            expansion_terms = []
            for j in range(3):
                minor = matrix.minor_matrix(0, j)
                cofactor = ((-1)**(0+j)) * minor.det()
                expansion_terms.append(f"{matrix[0,j]} · ({(-1)**(0+j)}) · det({minor}) = {matrix[0,j]} · {cofactor}")
            
            steps.append("  " + " + ".join(expansion_terms))
        
        else:
            steps.append(f"For a {size}×{size} matrix, the determinant is calculated using cofactor expansion.")
            steps.append("The calculation is complex; displaying only the final result.")
        
        steps.append(f"\nFinal determinant = {result}")
        return "\n".join(steps)
    
    def _generate_matrix_inverse_steps(self, matrix, result):
        """Generate step-by-step explanation for matrix inversion."""
        steps = []
        steps.append(f"Calculating inverse of matrix:\n{matrix}")
        
        if matrix.shape[0] != matrix.shape[1]:
            steps.append("Error: Matrix must be square to have an inverse")
            return "\n".join(steps)
        
        det = matrix.det()
        steps.append(f"\nStep 1: Calculate the determinant = {det}")
        
        if det == 0:
            steps.append("The determinant is zero, so the matrix is singular and has no inverse.")
            return "\n".join(steps)
        
        size = matrix.shape[0]
        
        if size == 2:
            a, b = matrix[0,0], matrix[0,1]
            c, d = matrix[1,0], matrix[1,1]
            
            steps.append(f"\nStep 2: For a 2×2 matrix [[a, b], [c, d]], the inverse is:")
            steps.append(f"A⁻¹ = (1/det(A)) · [[d, -b], [-c, a]]")
            steps.append(f"A⁻¹ = (1/{det}) · [[{d}, {-b}], [{-c}, {a}]]")
        else:
            steps.append(f"\nStep 2: Calculate the adjugate matrix (matrix of cofactors, transposed)")
            steps.append(f"Step 3: Divide the adjugate matrix by the determinant ({det})")
        
        steps.append(f"\nFinal inverse = {result}")
        return "\n".join(steps)
    
    def _generate_matrix_transpose_steps(self, matrix, result):
        """Generate step-by-step explanation for matrix transpose."""
        steps = []
        steps.append(f"Calculating transpose of matrix:\n{matrix}")
        steps.append("\nTo transpose a matrix, we swap rows and columns.")
        steps.append(f"Size before: {matrix.shape[0]}×{matrix.shape[1]}")
        steps.append(f"Size after: {result.shape[0]}×{result.shape[1]}")
        
        if matrix.shape[0] <= 4 and matrix.shape[1] <= 4:
            steps.append("\nElement mapping:")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    steps.append(f"  A[{i+1},{j+1}] = {matrix[i,j]} → A^T[{j+1},{i+1}] = {result[j,i]}")
        
        steps.append(f"\nFinal transpose = {result}")
        return "\n".join(steps)
    
    def _generate_eigenvalues_steps(self, matrix, eigenvalues):
        """Generate step-by-step explanation for eigenvalue calculation."""
        steps = []
        steps.append(f"Calculating eigenvalues of matrix:\n{matrix}")
        
        if matrix.shape[0] != matrix.shape[1]:
            steps.append("Error: Eigenvalues are only defined for square matrices")
            return "\n".join(steps)
        
        steps.append("\nStep 1: Set up the characteristic equation: det(A - λI) = 0")
        steps.append("Step 2: Solve for λ")
        
        # For small matrices, show the characteristic polynomial
        if matrix.shape[0] <= 3:
            λ = sp.symbols('λ')
            identity = sp.eye(matrix.shape[0])
            char_matrix = matrix - λ * identity
            char_poly = char_matrix.det()
            
            steps.append(f"\nCharacteristic matrix (A - λI):\n{char_matrix}")
            steps.append(f"Characteristic polynomial: {char_poly} = 0")
        
        steps.append("\nEigenvalues with multiplicities:")
        for value, multiplicity in eigenvalues.items():
            steps.append(f"  λ = {value} (multiplicity: {multiplicity})")
        
        return "\n".join(steps)
    
    @staticmethod
    def _ordinal(n):
        """Convert a number to its ordinal representation (1st, 2nd, 3rd, etc.)."""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
