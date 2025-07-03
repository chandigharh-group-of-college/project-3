import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, diff, integrate, solve, Matrix, latex
import scipy.stats as stats
import re
import math

class Calculator:
    """Core calculator engine that performs calculations and symbolic operations."""
    
    def __init__(self):
        self.x, self.y, self.z = sp.symbols('x y z')
        self.memory = 0
        self.history = []
        self.last_result = None
        
    def evaluate(self, expression, use_sympy=False):
        """Evaluate a mathematical expression."""
        try:
            # Clean the expression
            expression = expression.replace('^', '**')
            
            if use_sympy:
                # Use sympy for symbolic calculations
                if not re.match(r'^[0-9a-zA-Z\+\-\*\/\(\)\.]+$', expression):
                    print("Cleaned expression contains unsupported characters.")
                    return None, None
                result = parse_expr(expression)
                self.last_result = result
                return result, None
            else:
                # Use Python's eval for numerical calculations
                # Replace common mathematical functions with their numpy equivalents
                for func in ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'pi']:
                    if func in expression:
                        expression = expression.replace(func, f"np.{func}")
                
                # Basic security check
                if any(word in expression for word in ['import', 'eval', 'exec', 'open', '__']):
                    raise ValueError("Invalid expression: potentially unsafe operations detected")
                
                # Evaluate the expression
                result = eval(expression, {"np": np, "math": math})
                self.last_result = result
                return result, None
                
        except Exception as e:
            return None, str(e)
    
    def solve_equation(self, equation, variable='x'):
        """Solve an equation for the specified variable."""
        try:
            if '=' in equation:
                left_side, right_side = equation.split('=')
                equation = f"({left_side})-({right_side})"
            
            expr = parse_expr(equation)
            var = sp.symbols(variable)
            
            # Solve the equation
            solutions = sp.solve(expr, var)
            steps = self._generate_steps_for_equation(equation, variable, solutions)
            
            self.last_result = solutions
            return solutions, steps
        except Exception as e:
            return None, str(e)
    
    def differentiate(self, expression, variable='x', order=1):
        """Differentiate an expression with respect to the specified variable."""
        try:
            # Clean the expression
            expression = expression.replace('^', '**')
            
            # Parse the expression
            try:
                expr = parse_expr(expression)
            except SyntaxError:
                return None, "Invalid syntax in expression. Please check your input."
            except Exception as e:
                return None, f"Error parsing expression: {str(e)}"
                
            var = sp.symbols(variable)
            
            # Calculate the derivative
            result = sp.diff(expr, var, order)
            steps = self._generate_steps_for_differentiation(expression, variable, order, result)
            
            self.last_result = result
            return result, steps
        except Exception as e:
            return None, str(e)
    
    def integrate_expr(self, expression, variable='x', lower=None, upper=None):
        """Integrate an expression with respect to the specified variable."""
        try:
            # Clean the expression
            expression = expression.replace('^', '**')
            
            # Parse the expression
            try:
                expr = parse_expr(expression)
            except SyntaxError:
                return None, "Invalid syntax in expression. Please check your input."
            except Exception as e:
                return None, f"Error parsing expression: {str(e)}"
                
            var = sp.symbols(variable)
            
            # Calculate the integral
            if lower is not None and upper is not None:
                try:
                    lower_bound = float(lower) if lower.replace('.', '', 1).isdigit() else parse_expr(lower)
                    upper_bound = float(upper) if upper.replace('.', '', 1).isdigit() else parse_expr(upper)
                except SyntaxError:
                    return None, "Invalid syntax in integration limits. Please check your input."
                except Exception as e:
                    return None, f"Error parsing integration limits: {str(e)}"
                    
                result = sp.integrate(expr, (var, lower_bound, upper_bound))
                integral_type = "definite"
            else:
                result = sp.integrate(expr, var)
                integral_type = "indefinite"
            
            steps = self._generate_steps_for_integration(expression, variable, result, integral_type, lower, upper)
            self.last_result = result
            return result, steps
        except Exception as e:
            return None, str(e)
    
    def parse_matrix_input(self, input_str):
        """Parse matrix input in various formats and return a SymPy Matrix."""
        try:
            # Clean up input string
            input_str = input_str.strip()
            
            # Check if it's in Python list format: [[1, 2], [3, 4]]
            if input_str.startswith('[[') and input_str.endswith(']]'):
                try:
                    import ast
                    matrix_data = ast.literal_eval(input_str)
                    if isinstance(matrix_data, list) and all(isinstance(row, list) for row in matrix_data):
                        return Matrix(matrix_data), None
                except Exception as e:
                    return None, f"Invalid matrix format: {str(e)}"
            
            # Check if rows are separated by semicolons: [1, 2; 3, 4]
            if ';' in input_str:
                if input_str.startswith('[') and input_str.endswith(']'):
                    input_str = input_str[1:-1]
                
                rows = []
                for row_str in input_str.split(';'):
                    row_str = row_str.strip()
                    if row_str.startswith('[') and row_str.endswith(']'):
                        row_str = row_str[1:-1]
                    
                    elements = []
                    for elem_str in row_str.split(','):
                        elem_str = elem_str.strip()
                        try:
                            # Try to parse as number or symbolic expression
                            if elem_str.replace('.', '', 1).isdigit():
                                elements.append(float(elem_str))
                            else:
                                elements.append(parse_expr(elem_str))
                        except Exception:
                            # If parsing fails, treat as symbolic variable
                            elements.append(sp.Symbol(elem_str))
                    
                    rows.append(elements)
                
                return Matrix(rows), None
            
            # Try parsing as multiline matrix with space-separated elements
            rows = []
            for line in input_str.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('[') and line.endswith(']'):
                    line = line[1:-1]
                
                elements = []
                for elem_str in line.split():
                    elem_str = elem_str.strip()
                    try:
                        # Try to parse as number or symbolic expression
                        if elem_str.replace('.', '', 1).isdigit():
                            elements.append(float(elem_str))
                        else:
                            elements.append(parse_expr(elem_str))
                    except Exception:
                        # If parsing fails, treat as symbolic variable
                        elements.append(sp.Symbol(elem_str))
                
                if elements:
                    rows.append(elements)
            
            if rows:
                return Matrix(rows), None
            
            raise ValueError("Could not parse matrix format")
            
        except Exception as e:
            return None, f"Error parsing matrix: {str(e)}"

    def matrix_operation(self, matrix_a, matrix_b=None, operation='determinant'):
        """Perform matrix operations."""
        try:
            # Convert string representation to sympy Matrix if needed
            if isinstance(matrix_a, str):
                matrix_a, err = self.parse_matrix_input(matrix_a)
                if err:
                    return None, err
            
            if matrix_b and isinstance(matrix_b, str):
                matrix_b, err = self.parse_matrix_input(matrix_b)
                if err:
                    return None, err
            
            # Perform the requested operation
            if operation == 'determinant':
                result = matrix_a.det()
                steps = self._matrix_determinant_steps(matrix_a, result)
            elif operation == 'inverse':
                if matrix_a.det() == 0:
                    return None, "Matrix is singular (determinant is zero), inverse does not exist."
                result = matrix_a.inv()
                steps = self._matrix_inverse_steps(matrix_a, result)
            elif operation == 'transpose':
                result = matrix_a.transpose()
                steps = f"Computing transpose of matrix:\n{self._format_matrix(matrix_a)}\n\nResult:\n{self._format_matrix(result)}"
            elif operation == 'multiply' and matrix_b:
                if matrix_a.shape[1] != matrix_b.shape[0]:
                    return None, f"Matrix dimensions incompatible for multiplication. A is {matrix_a.shape} and B is {matrix_b.shape}."
                result = matrix_a * matrix_b
                steps = self._matrix_multiply_steps(matrix_a, matrix_b, result)
            elif operation == 'add' and matrix_b:
                if matrix_a.shape != matrix_b.shape:
                    return None, f"Matrix dimensions must match for addition. A is {matrix_a.shape} and B is {matrix_b.shape}."
                result = matrix_a + matrix_b
                steps = f"Adding matrices:\n{self._format_matrix(matrix_a)}\n+\n{self._format_matrix(matrix_b)}\n\nResult:\n{self._format_matrix(result)}"
            elif operation == 'subtract' and matrix_b:
                if matrix_a.shape != matrix_b.shape:
                    return None, f"Matrix dimensions must match for subtraction. A is {matrix_a.shape} and B is {matrix_b.shape}."
                result = matrix_a - matrix_b
                steps = f"Subtracting matrices:\n{self._format_matrix(matrix_a)}\n-\n{self._format_matrix(matrix_b)}\n\nResult:\n{self._format_matrix(result)}"
            elif operation == 'eigenvalues':
                eigenvals = matrix_a.eigenvals()
                result = list(eigenvals.keys())
                steps = f"Computing eigenvalues of matrix:\n{self._format_matrix(matrix_a)}\n\nEigenvalues: {result}"
            elif operation == 'eigenvectors':
                result = matrix_a.eigenvects()
                steps = f"Computing eigenvectors of matrix:\n{self._format_matrix(matrix_a)}\n\nResults:{self._format_eigenvectors(result)}"
            else:
                raise ValueError(f"Unsupported matrix operation: {operation}")
                
            self.last_result = result
            return result, steps
        except Exception as e:
            return None, str(e)
    
    def convert_base(self, number, from_base=10, to_base=2):
        """Convert a number between different bases."""
        try:
            # Convert the input to base 10 first
            if from_base != 10:
                decimal = int(str(number), from_base)
            else:
                decimal = int(number)
            
            # Then convert from base 10 to the target base
            if to_base == 10:
                result = decimal
            elif to_base == 16:
                result = hex(decimal)[2:]  # Remove '0x' prefix
            elif to_base == 8:
                result = oct(decimal)[2:]  # Remove '0o' prefix
            elif to_base == 2:
                result = bin(decimal)[2:]  # Remove '0b' prefix
            else:
                result = ""
                temp = decimal
                while temp > 0:
                    remainder = temp % to_base
                    if remainder < 10:
                        result = str(remainder) + result
                    else:
                        result = chr(remainder + 55) + result  # A-F for bases > 10
                    temp //= to_base
            
            steps = f"Converting {number} from base {from_base} to base {to_base}:\n"
            steps += f"Step 1: Convert to decimal → {decimal}\n"
            if to_base != 10:
                steps += f"Step 2: Convert decimal to base {to_base} → {result}"
            
            self.last_result = result
            return result, steps
        except Exception as e:
            return None, str(e)
    
    def unit_conversion(self, value, from_unit, to_unit):
        """Convert between different units of measurement."""
        # Dictionary of conversion factors to SI units
        unit_to_si = {
            # Length
            "mm": 0.001, "cm": 0.01, "m": 1, "km": 1000, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.34,
            # Mass
            "mg": 0.000001, "g": 0.001, "kg": 1, "oz": 0.0283495, "lb": 0.453592, "ton": 1000,
            # Volume
            "ml": 0.000001, "l": 0.001, "m3": 1, "gal": 0.00378541, "qt": 0.000946353,
            # Time
            "s": 1, "min": 60, "hr": 3600, "day": 86400,
            # Temperature requires special handling
            "c": "celsius", "f": "fahrenheit", "k": "kelvin",
            # Area
            "mm2": 0.000001, "cm2": 0.0001, "m2": 1, "km2": 1000000, "in2": 0.00064516, "ft2": 0.092903, "acre": 4046.86,
            # Energy
            "j": 1, "cal": 4.184, "kcal": 4184, "kwh": 3600000,
            # Pressure
            "pa": 1, "kpa": 1000, "atm": 101325, "bar": 100000, "psi": 6894.76
        }
        
        try:
            # Handle temperature conversions separately
            if from_unit.lower() in ["c", "f", "k"] and to_unit.lower() in ["c", "f", "k"]:
                if from_unit.lower() == "c" and to_unit.lower() == "f":
                    result = (value * 9/5) + 32
                    formula = "(value × 9/5) + 32"
                elif from_unit.lower() == "f" and to_unit.lower() == "c":
                    result = (value - 32) * 5/9
                    formula = "(value - 32) × 5/9"
                elif from_unit.lower() == "c" and to_unit.lower() == "k":
                    result = value + 273.15
                    formula = "value + 273.15"
                elif from_unit.lower() == "k" and to_unit.lower() == "c":
                    result = value - 273.15
                    formula = "value - 273.15"
                elif from_unit.lower() == "f" and to_unit.lower() == "k":
                    result = (value - 32) * 5/9 + 273.15
                    formula = "(value - 32) × 5/9 + 273.15"
                elif from_unit.lower() == "k" and to_unit.lower() == "f":
                    result = (value - 273.15) * 9/5 + 32
                    formula = "(value - 273.15) × 9/5 + 32"
                else:
                    result = value  # Same unit
                    formula = "value"
            else:
                # Standard conversion through SI units
                if from_unit.lower() not in unit_to_si or to_unit.lower() not in unit_to_si:
                    raise ValueError(f"Unsupported unit: {from_unit} or {to_unit}")
                
                # Convert from source unit to SI
                si_value = value * unit_to_si[from_unit.lower()]
                
                # Convert from SI to target unit
                result = si_value / unit_to_si[to_unit.lower()]
                formula = f"value × {unit_to_si[from_unit.lower()]} ÷ {unit_to_si[to_unit.lower()]}"
            
            steps = (f"Converting {value} {from_unit} to {to_unit}:\n"
                     f"Formula: {formula}\n"
                     f"Result: {result} {to_unit}")
            
            self.last_result = result
            return result, steps
        except Exception as e:
            return None, str(e)
    
    def statistical_analysis(self, data, operation):
        """Perform statistical analysis on a dataset."""
        try:
            if isinstance(data, str):
                # Parse string into a list of numbers
                data = [float(x) for x in re.findall(r'-?\d+\.?\d*', data)]
            
            if not data:
                raise ValueError("No valid data provided")
            
            steps = f"Statistical analysis of data: {data}\n\n"
            
            if operation == "mean":
                result = np.mean(data)
                steps += f"Mean = sum of all values / number of values\n"
                steps += f"Mean = {sum(data)} / {len(data)} = {result}"
            
            elif operation == "median":
                result = np.median(data)
                steps += f"Median = middle value of the sorted data\n"
                steps += f"Sorted data: {sorted(data)}\n"
                steps += f"Median = {result}"
            
            elif operation == "mode":
                # Find the mode using a frequency count
                value_counts = {}
                for num in data:
                    value_counts[num] = value_counts.get(num, 0) + 1
                
                max_count = max(value_counts.values())
                modes = [k for k, v in value_counts.items() if v == max_count]
                
                result = modes
                steps += f"Mode = most frequent value(s) in the data\n"
                steps += f"Value counts: {value_counts}\n"
                steps += f"Mode(s): {modes}"
            
            elif operation == "stdev":
                result = np.std(data, ddof=1)  # Sample standard deviation
                steps += f"Standard Deviation = sqrt(sum((x - mean)²) / (n - 1))\n"
                steps += f"Mean = {np.mean(data)}\n"
                steps += f"Sum of squared differences = {sum((x - np.mean(data))**2 for x in data)}\n"
                steps += f"Standard Deviation = {result}"
            
            elif operation == "variance":
                result = np.var(data, ddof=1)  # Sample variance
                steps += f"Variance = sum((x - mean)²) / (n - 1)\n"
                steps += f"Mean = {np.mean(data)}\n"
                steps += f"Sum of squared differences = {sum((x - np.mean(data))**2 for x in data)}\n"
                steps += f"Variance = {result}"
            
            else:
                raise ValueError(f"Unsupported statistical operation: {operation}")
            
            self.last_result = result
            return result, steps
        except Exception as e:
            return None, str(e)
    
    def process_text_command(self, command):
        """Process natural language commands to perform calculations with enhanced parsing."""
        command = command.lower().strip()
        
        # Check for matrix operations
        matrix_patterns = [
            (r"matrix\s+(.*?)(?:\s+and\s+|\s+with\s+)(.*?)(?:\s+add|\s+sum)", "add"),
            (r"add\s+(?:matrix\s+)?(.*?)(?:\s+and\s+|\s+with\s+|\s+to\s+)(?:matrix\s+)?(.*)", "add"),
            (r"matrix\s+(.*?)(?:\s+and\s+|\s+with\s+)(.*?)(?:\s+multiply|\s+product)", "multiply"),
            (r"multiply\s+(?:matrix\s+)?(.*?)(?:\s+and\s+|\s+with\s+|\s+by\s+)(?:matrix\s+)?(.*)", "multiply"),
            (r"(?:inverse|invert)(?:\s+of)?\s+(?:matrix\s+)?(.*)", "inverse"),
            (r"(?:determinant|det)(?:\s+of)?\s+(?:matrix\s+)?(.*)", "determinant"),
            (r"transpose(?:\s+of)?\s+(?:matrix\s+)?(.*)", "transpose"),
            (r"eigenvalues(?:\s+of)?\s+(?:matrix\s+)?(.*)", "eigenvalues"),
        ]
        
        # Try to match matrix patterns
        for pattern, op in matrix_patterns:
            match = re.search(pattern, command)
            if match:
                if op in ["add", "multiply"]:
                    matrix_a_str = match.group(1).strip()
                    matrix_b_str = match.group(2).strip()
                    return self.matrix_operation(matrix_a_str, matrix_b_str, op)
                else:
                    matrix_str = match.group(1).strip()
                    return self.matrix_operation(matrix_str, operation=op)
        
        # Check for symbolic operations using existing patterns
        if "differentiate" in command or "derivative" in command:
            # Extract the expression and variable
            expr_match = re.search(r'(differentiate|derivative of|d/dx)[\s]*(.*?)(\s+with respect to\s+|\s+w\.r\.t\.?\s+|$)', command)
            var_match = re.search(r'with respect to\s+(\w+)|w\.r\.t\.?\s+(\w+)', command)
            
            if expr_match:
                expr = expr_match.group(2).strip()
                var = var_match.group(1) if var_match else 'x'
                return self.differentiate(expr, var)
        
        elif "integrate" in command:
            # Check for definite integral
            if "from" in command and "to" in command:
                expr_match = re.search(r'integrate\s+(.*?)\s+from\s+([-+]?\d*\.?\d+|\w+)\s+to\s+([-+]?\d*\.?\d+|\w+)', command)
                if expr_match:
                    expr = expr_match.group(1).strip()
                    lower = expr_match.group(2).strip()
                    upper = expr_match.group(3).strip()
                    return self.integrate_expr(expr, 'x', lower, upper)
            else:
                # Indefinite integral
                expr_match = re.search(r'integrate\s+(.*?)($|\s+with|\s+dx)', command)
                if expr_match:
                    expr = expr_match.group(1).strip()
                    return self.integrate_expr(expr)
        
        elif "solve" in command and "equation" in command:
            eq_match = re.search(r'solve(?:\s+the)?\s+equation\s+(.*?)(?:\s+for\s+(\w+)|$)', command)
            if eq_match:
                equation = eq_match.group(1).strip()
                var = eq_match.group(2) if eq_match.group(2) else 'x'
                return self.solve_equation(equation, var)
        
        # If no specific command matched, try to evaluate as a direct expression
        return self.evaluate(command)
    
    def _format_matrix(self, matrix):
        """Format a matrix for display with proper alignment."""
        # Convert matrix to string representation
        str_matrix = str(matrix)
        # Remove 'Matrix' prefix and clean up formatting
        str_matrix = str_matrix.replace("Matrix", "")
        return str_matrix
    
    def _format_eigenvectors(self, eigenvects):
        """Format eigenvectors result for display."""
        result = "\n"
        for eigenvalue, multiplicity, eigenvects in eigenvects:
            result += f"\nEigenvalue: {eigenvalue} (multiplicity: {multiplicity})\n"
            result += "Eigenvectors:\n"
            for vec in eigenvects:
                result += f"  {vec}\n"
        return result
    
    def _matrix_determinant_steps(self, matrix, result):
        """Generate step-by-step explanation for determinant calculation."""
        steps = [f"Computing determinant of matrix:\n{self._format_matrix(matrix)}"]
        
        if matrix.shape == (1, 1):
            steps.append(f"For a 1×1 matrix, the determinant is just the single element: {matrix[0, 0]}")
        elif matrix.shape == (2, 2):
            steps.append(f"For a 2×2 matrix [[a, b], [c, d]], the determinant is ad - bc")
            steps.append(f"det = {matrix[0, 0]} × {matrix[1, 1]} - {matrix[0, 1]} × {matrix[1, 0]}")
            steps.append(f"det = {matrix[0, 0] * matrix[1, 1]} - {matrix[0, 1] * matrix[1, 0]}")
        elif matrix.shape == (3, 3):
            steps.append("For a 3×3 matrix, we can use the cofactor expansion along the first row:")
            
            # Cofactor expansion explanation
            cofactors = []
            for j in range(3):
                minor = matrix.minor_matrix(0, j)
                cofactor = (-1)**(0+j) * minor.det()
                cofactors.append(f"{matrix[0, j]} × ({(-1)**(0+j)}) × det{self._format_matrix(minor)} = {matrix[0, j] * cofactor}")
            
            steps.append("\n".join(cofactors))
        
        steps.append(f"\nFinal result: det = {result}")
        return "\n\n".join(steps)
    
    def _matrix_inverse_steps(self, matrix, result):
        """Generate step-by-step explanation for matrix inverse calculation."""
        steps = [f"Computing inverse of matrix:\n{self._format_matrix(matrix)}"]
        
        det = matrix.det()
        steps.append(f"Step 1: Calculate the determinant = {det}")
        
        if det == 0:
            steps.append("The determinant is zero, so the matrix is not invertible.")
            return "\n\n".join(steps)
        
        if matrix.shape == (2, 2):
            steps.append("Step 2: For a 2×2 matrix [[a, b], [c, d]], the inverse is:")
            steps.append("1/det × [[d, -b], [-c, a]]")
            
            adj_matrix = Matrix([
                [matrix[1, 1], -matrix[0, 1]],
                [-matrix[1, 0], matrix[0, 0]]
            ])
            
            steps.append(f"Adjugate matrix = {self._format_matrix(adj_matrix)}")
            steps.append(f"Inverse = (1/{det}) × {self._format_matrix(adj_matrix)}")
        else:
            steps.append(f"Step 2: Calculate the adjugate matrix (matrix of cofactors, transposed)")
            steps.append(f"Step 3: Divide the adjugate matrix by the determinant")
        
        steps.append(f"\nFinal result:\n{self._format_matrix(result)}")
        return "\n\n".join(steps)
    
    def _matrix_multiply_steps(self, matrix_a, matrix_b, result):
        """Generate step-by-step explanation for matrix multiplication."""
        steps = []
        steps.append(f"Multiplying matrices:")
        steps.append(f"Matrix A ({matrix_a.shape[0]}×{matrix_a.shape[1]}):\n{self._format_matrix(matrix_a)}")
        steps.append(f"Matrix B ({matrix_b.shape[0]}×{matrix_b.shape[1]}):\n{self._format_matrix(matrix_b)}")
        
        if matrix_a.shape[1] != matrix_b.shape[0]:
            steps.append(f"Error: Cannot multiply these matrices. The number of columns in A ({matrix_a.shape[1]}) must equal the number of rows in B ({matrix_b.shape[0]}).")
            return "\n\n".join(steps)
        
        steps.append(f"To multiply matrices, we compute the dot product of each row of A with each column of B.")
        
        # For small matrices, show the calculation for each element
        if matrix_a.shape[0] <= 3 and matrix_b.shape[1] <= 3:
            element_steps = []
            for i in range(matrix_a.shape[0]):
                for j in range(matrix_b.shape[1]):
                    dot_product = " + ".join([f"({matrix_a[i, k]} × {matrix_b[k, j]})" for k in range(matrix_a.shape[1])])
                    element_steps.append(f"C[{i+1},{j+1}] = {dot_product} = {result[i, j]}")
            steps.append("\n".join(element_steps))
        
        steps.append(f"\nResult ({result.shape[0]}×{result.shape[1]}):\n{self._format_matrix(result)}")
        return "\n\n".join(steps)

    def _generate_steps_for_differentiation(self, expression, variable, order, result):
        """Generate step-by-step explanation for differentiation."""
        steps = f"Step 1: Identify the function f({variable}) = {expression}\n\n"
        
        if order == 1:
            steps += f"Step 2: Find the derivative f'({variable}) using differentiation rules\n\n"
        else:
            steps += f"Step 2: Find the {order}th derivative f^({order})({variable}) using repeated differentiation\n\n"
        
        steps += f"Step 3: Apply differentiation rules to get:\n"
        steps += f"f"
        steps += "'" * order
        steps += f"({variable}) = {result}\n\n"
        
        # Add the final result
        steps += f"Result: {result}"
        return steps
    
    def _generate_steps_for_integration(self, expression, variable, result, integral_type, lower=None, upper=None):
        """Generate step-by-step explanation for integration."""
        steps = f"Step 1: Identify the integrand: {expression}\n\n"
        
        if integral_type == "indefinite":
            steps += f"Step 2: Find the antiderivative with respect to {variable}\n\n"
            steps += f"Step 3: Apply integration rules to get:\n"
            steps += f"∫ {expression} d{variable} = {result} + C\n\n"
        else:  # definite
            steps += f"Step 2: Find the indefinite integral with respect to {variable}\n\n"
            steps += f"Step 3: Apply the Fundamental Theorem of Calculus:\n"
            steps += f"∫({lower}, {upper}) {expression} d{variable} = F({upper}) - F({lower})\n\n"
            steps += f"Step 4: Substitute the limits into the antiderivative:\n"
            steps += f"Result: {result}\n"
        
        # Add the final result
        if integral_type == "indefinite":
            steps += f"Result: {result} + C  (where C is an arbitrary constant)"
        else:
            steps += f"Result: {result}"
        return steps
    
    def _generate_steps_for_equation(self, equation, variable, solutions):
        """Generate detailed step-by-step explanation for equation solving."""
        steps = []
        
        # Step 1: Format the equation
        if '=' in equation:
            left_side, right_side = equation.split('=')
            steps.append(f"Step 1: Original equation: {left_side} = {right_side}")
            
            # Step 2: Move all terms to one side
            steps.append(f"Step 2: Move all terms to the left side: {left_side} - ({right_side}) = 0")
            
            # Parse both sides
            left_expr, _ = parse_expr(left_side), parse_expr(right_side)
            expr = parse_expr(f"({left_side})-({right_side})")
        else:
            # Assume it's already in the form expr = 0
            steps.append(f"Step 1: Original equation: {equation} = 0")
            expr = parse_expr(equation)
        
        # Simplify the expression
        simplified = sp.expand(expr)
        if simplified != expr:
            steps.append(f"Step 3: Expand the expression: {simplified} = 0")
        
        # Check for specific equation types and provide more detailed steps
        degree = sp.degree(simplified, variable)
        
        if degree == 1:
            # Linear equation
            steps.append("Step 4: This is a linear equation in the form ax + b = 0")
            
            # Collect terms with the variable and constants
            var_coeff = simplified.coeff(variable, 1)
            constant_term = simplified - var_coeff * variable
            
            steps.append(f"Step 5: Rewrite as {var_coeff}*{variable} + {constant_term} = 0")
            steps.append(f"Step 6: Isolate the variable by subtracting {constant_term} from both sides:")
            steps.append(f"{var_coeff}*{variable} = {-constant_term}")
            
            if var_coeff != 1:
                steps.append(f"Step 7: Divide both sides by {var_coeff}:")
                steps.append(f"{variable} = {-constant_term/var_coeff}")
            
        elif degree == 2:
            # Quadratic equation
            steps.append("Step 4: This is a quadratic equation in the form ax² + bx + c = 0")
            
            # Collect coefficients
            a = simplified.coeff(variable, 2)
            b = simplified.coeff(variable, 1)
            c = simplified.as_coeff_add(variable)[0]
            
            steps.append(f"Step 5: Identify coefficients: a = {a}, b = {b}, c = {c}")
            
            # Try to factor if simple integers
            if all(coef.is_Integer for coef in [a, b, c]):
                # Check if trinomial can be factored nicely
                try:
                    factored = sp.factor(simplified)
                    if factored != simplified:
                        steps.append(f"Step 6: Factor the expression: {factored} = 0")
                        steps.append("Step 7: Set each factor equal to zero and solve:")
                        
                        # Extract factors and solve each one
                        if isinstance(factored, sp.Mul):
                            factors = factored.args
                            for i, factor in enumerate(factors, 1):
                                if variable in factor.free_symbols:  # Only process factors containing the variable
                                    steps.append(f"Factor {i}: {factor} = 0")
                                    if factor.is_Pow:
                                        base = factor.args[0]
                                        exp = factor.args[1]
                                        steps.append(f"   {base} = 0 (with multiplicity {exp})")
                                    else:
                                        steps.append(f"   Solving for {variable}")
                except Exception:
                    # If factoring fails, fallback to quadratic formula
                    pass
            
            # Use quadratic formula
            discriminant = b**2 - 4*a*c
            steps.append(f"Step 6: Calculate the discriminant: Δ = b² - 4ac = {b}² - 4({a})({c}) = {discriminant}")
            
            if discriminant > 0:
                steps.append(f"Step 7: Since Δ > 0, there are two real solutions. Using the quadratic formula:")
                steps.append(f"x = (-b ± √Δ) / (2a) = ({-b} ± √{discriminant}) / (2 × {a})")
            elif discriminant == 0:
                steps.append(f"Step 7: Since Δ = 0, there is one repeated real solution. Using the quadratic formula:")
                steps.append(f"x = -b / (2a) = {-b} / (2 × {a}) = {-b/(2*a)}")
            else:
                steps.append(f"Step 7: Since Δ < 0, there are two complex solutions. Using the quadratic formula:")
                steps.append(f"x = (-b ± √Δ) / (2a) = ({-b} ± √{discriminant}) / (2 × {a})")
        
        # Final step: List the solutions
        if solutions:
            steps.append(f"\nFinal solution(s):")
            for i, sol in enumerate(solutions, 1):
                steps.append(f"  Solution {i}: {variable} = {sol}")
        else:
            steps.append("\nNo solutions found")
        
        return "\n".join(steps) 