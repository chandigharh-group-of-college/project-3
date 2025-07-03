# solver_ui_integration.py - Intelligent Solver UI Integration
# SAUNet 4.0 - Step-by-Step Solver Integration

"""
UI integration for the Intelligent Math Solver in SAUNet 4.0.
Adds step-by-step solving capability to the existing calculator interface.
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

try:
    from intelligent_math_solver import IntelligentMathSolver
    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False
    print("⚠️ Intelligent Math Solver not available")


class StepBySteTabWidget(QWidget):
    """Tab widget for step-by-step mathematical solutions"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        if SOLVER_AVAILABLE:
            self.solver = IntelligentMathSolver()
        else:
            self.solver = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the step-by-step solver UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🧠 Intelligent Step-by-Step Solver")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #667eea, stop:1 #764ba2);
                color: white; 
                padding: 12px; 
                border-radius: 8px;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(title)
        
        # Input section
        input_frame = QFrame()
        input_frame.setFrameStyle(QFrame.StyledPanel)
        input_layout = QVBoxLayout(input_frame)
        
        # Problem input
        self.problem_input = QLineEdit()
        self.problem_input.setPlaceholderText("Enter your math problem (e.g., 'solve x + 5 = 10', 'what is 15% of 200')")
        self.problem_input.setFont(QFont("Arial", 11))
        self.problem_input.setMinimumHeight(35)
        self.problem_input.returnPressed.connect(self.solve_problem)
        input_layout.addWidget(QLabel("💭 Ask me anything mathematical:"))
        input_layout.addWidget(self.problem_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.solve_btn = QPushButton("🔍 Solve Step-by-Step")
        self.solve_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.solve_btn.clicked.connect(self.solve_problem)
        
        self.detailed_btn = QPushButton("🔧 Force Detailed")
        self.detailed_btn.clicked.connect(lambda: self.solve_problem(True))
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_solution)
        
        button_layout.addWidget(self.solve_btn)
        button_layout.addWidget(self.detailed_btn)
        button_layout.addWidget(self.clear_btn)
        
        input_layout.addLayout(button_layout)
        layout.addWidget(input_frame)
        
        # Solution display
        self.solution_display = QTextEdit()
        self.solution_display.setReadOnly(True)
        self.solution_display.setFont(QFont("Consolas", 10))
        self.solution_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.solution_display)
        
        # Quick examples
        examples_frame = QFrame()
        examples_frame.setMaximumHeight(120)
        examples_layout = QVBoxLayout(examples_frame)
        examples_layout.addWidget(QLabel("📚 Quick Examples:"))
        
        examples_grid = QGridLayout()
        examples = [
            "solve 2x + 3 = 11",
            "what is 25% of 80", 
            "simplify (x + 3)²",
            "find 144 / 12 + 5"
        ]
        
        for i, example in enumerate(examples):
            btn = QPushButton(example)
            btn.setMaximumHeight(25)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 4px;
                    border: 1px solid #ccc;
                    background: white;
                }
                QPushButton:hover { background: #f0f0f0; }
            """)
            btn.clicked.connect(lambda checked, ex=example: self.load_example(ex))
            examples_grid.addWidget(btn, i // 2, i % 2)
        
        examples_layout.addLayout(examples_grid)
        layout.addWidget(examples_frame)
        
        # Status
        self.status_label = QLabel("Ready to solve mathematical problems!")
        self.status_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(self.status_label)
    
    def solve_problem(self, force_detailed=False):
        """Solve the mathematical problem"""
        if not SOLVER_AVAILABLE:
            self.solution_display.setHtml("""
                <div style='color: #e74c3c;'>
                <h3>❌ Intelligent Solver Not Available</h3>
                <p>The intelligent math solver module is not installed.</p>
                <p>This is a preview of what the interface would look like.</p>
                </div>
            """)
            return
        
        problem = self.problem_input.text().strip()
        if not problem:
            self.status_label.setText("⚠️ Please enter a mathematical problem")
            return
        
        self.status_label.setText("🔄 Analyzing problem...")
        QApplication.processEvents()
        
        try:
            result = self.solver.solve_step_by_step(problem, force_detailed)
            self.display_result(result)
        except Exception as e:
            self.solution_display.setHtml(f"""
                <div style='color: #e74c3c;'>
                <h3>❌ Error</h3>
                <p>Failed to solve: {str(e)}</p>
                </div>
            """)
            self.status_label.setText("❌ Error occurred")
    
    def display_result(self, result):
        """Display the solution result"""
        if not result['success']:
            self.solution_display.setHtml("""
                <div style='color: #e74c3c;'>
                <h3>❌ Unable to Solve</h3>
                <p>Please check your problem format and try again.</p>
                </div>
            """)
            return
        
        html = "<div style='font-family: Arial; font-size: 12px;'>"
        
        # Handle different response types
        if result.get('is_simple'):
            html += f"""
                <div style='background: #d4edda; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                <h3 style='color: #155724; margin: 0;'>🎯 Quick Answer</h3>
                <p style='font-size: 16px; font-weight: bold; margin: 5px 0;'>{result['direct_answer']}</p>
                <p style='margin: 0;'>{result['offer_steps']}</p>
                </div>
            """
        
        elif result.get('is_repetitive'):
            html += f"""
                <div style='background: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                <h3 style='color: #856404; margin: 0;'>🔄 Smart Memory</h3>
                <p style='margin: 5px 0;'>{result['message']}</p>
                <p style='font-weight: bold; margin: 0;'>Quick Answer: {result['quick_answer']}</p>
                </div>
            """
        
        else:
            # Detailed solution
            solution = result['solution']
            
            # Problem header
            html += f"""
                <div style='background: #cce5ff; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                <h3 style='color: #004085; margin: 0;'>📋 Problem Analysis</h3>
                <p style='margin: 5px 0;'><strong>Problem:</strong> {solution.problem}</p>
                <p style='margin: 5px 0;'><strong>Domain:</strong> {solution.mathematical_domain.title()}</p>
                <p style='margin: 5px 0;'><strong>Difficulty:</strong> {solution.difficulty_level}</p>
                <p style='margin: 0;'><strong>Key Concepts:</strong> {', '.join(solution.key_concepts)}</p>
                </div>
            """
            
            # Steps
            html += """
                <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                <h3 style='color: #495057; margin: 0 0 10px 0;'>🔧 Step-by-Step Solution</h3>
            """
            
            for step in solution.steps:
                html += f"""
                    <div style='background: white; padding: 8px; border-left: 4px solid #667eea; margin: 5px 0;'>
                    <p style='margin: 0; font-weight: bold; color: #667eea;'>Step {step.step_number}: {step.action}</p>
                    <p style='margin: 5px 0 0 0;'><strong>Explanation:</strong> {step.explanation}</p>
                    <p style='margin: 5px 0 0 0;'><strong>Principle:</strong> {step.principle}</p>
                    {f'<p style="margin: 5px 0 0 0;"><code>{step.equation_before} → {step.equation_after}</code></p>' if step.equation_before != step.equation_after else ''}
                    </div>
                """
            
            html += "</div>"
            
            # Final answer
            html += f"""
                <div style='background: #d4edda; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                <h3 style='color: #155724; margin: 0;'>🎉 Final Answer</h3>
                <p style='font-size: 16px; font-weight: bold; margin: 5px 0;'>{solution.final_answer}</p>
                </div>
            """
            
            # Real-world context
            if solution.real_world_context:
                html += f"""
                    <div style='background: #e2e3e5; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                    <h3 style='color: #383d41; margin: 0;'>🌍 Real-World Connection</h3>
                    <p style='margin: 5px 0 0 0;'>{solution.real_world_context}</p>
                    </div>
                """
            
            # Alternative methods
            if solution.alternative_methods:
                alt_list = ''.join(f'<li>{method}</li>' for method in solution.alternative_methods)
                html += f"""
                    <div style='background: #ffeeba; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                    <h3 style='color: #856404; margin: 0;'>🔀 Alternative Methods</h3>
                    <ul style='margin: 5px 0 0 20px;'>{alt_list}</ul>
                    </div>
                """
        
        # Suggestions
        if 'suggestions' in result and result['suggestions']:
            suggestions_list = ''.join(f'<li>{suggestion}</li>' for suggestion in result['suggestions'])
            html += f"""
                <div style='background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                <h3 style='color: #6c757d; margin: 0;'>💡 Suggestions & Next Steps</h3>
                <ul style='margin: 5px 0 0 20px;'>{suggestions_list}</ul>
                </div>
            """
        
        html += "</div>"
        
        self.solution_display.setHtml(html)
        self.status_label.setText("✅ Problem solved successfully!")
    
    def load_example(self, example):
        """Load an example problem"""
        self.problem_input.setText(example)
        self.solve_problem()
    
    def clear_solution(self):
        """Clear the solution display"""
        self.solution_display.clear()
        self.problem_input.clear()
        self.status_label.setText("Ready to solve mathematical problems!")


def integrate_intelligent_solver_tab(calculator_ui):
    """
    Integration function to add the intelligent solver tab to SAUNet 4.0
    
    Args:
        calculator_ui: The main calculator UI instance
    """
    if hasattr(calculator_ui, 'tabs'):
        # Add the step-by-step solver as a new tab
        solver_tab = StepBySteTabWidget(calculator_ui)
        calculator_ui.tabs.addTab(solver_tab, "🧠 Step-by-Step")
        
        print("✅ Intelligent Step-by-Step Solver integrated into SAUNet 4.0")
    else:
        print("⚠️ Cannot integrate - calculator UI doesn't have tabs")


# Test function
def test_solver_ui():
    """Test the solver UI component"""
    import sys
    app = QApplication(sys.argv)
    
    widget = StepBySteTabWidget()
    widget.setWindowTitle("SAUNet 4.0 - Intelligent Solver Test")
    widget.resize(800, 600)
    widget.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    test_solver_ui() 