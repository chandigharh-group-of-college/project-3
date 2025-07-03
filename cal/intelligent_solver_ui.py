# intelligent_solver_ui.py - UI Integration for Intelligent Math Solver
# SAUNet 4.0 - Interactive Step-by-Step Solution Interface

"""
UI Integration component for the Intelligent Math Solver.
Provides rich interactive interface for step-by-step mathematical solutions
with memory, suggestions, and contextual responses.
"""

import sys
from typing import Dict, List, Any, Optional
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

try:
    from intelligent_math_solver import IntelligentMathSolver, SolutionStep, MathSolution
except ImportError:
    print("⚠️ Intelligent Math Solver not available")
    # Create dummy classes
    class IntelligentMathSolver:
        def __init__(self): pass
        def solve_step_by_step(self, problem, force=False): 
            return {'success': False, 'error': 'Solver not available'}


class StepByStepWidget(QWidget):
    """Widget for displaying step-by-step mathematical solutions"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.solver = IntelligentMathSolver()
        self.current_solution = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the step-by-step solver UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("🧠 Intelligent Step-by-Step Solver")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                      stop:0 #667eea, stop:1 #764ba2);
            color: white; 
            padding: 10px; 
            border-radius: 8px;
            margin-bottom: 10px;
        """)
        layout.addWidget(header)
        
        # Input section
        input_group = QGroupBox("💭 Ask me anything mathematical!")
        input_layout = QVBoxLayout(input_group)
        
        self.problem_input = QLineEdit()
        self.problem_input.setPlaceholderText("Enter your math problem (e.g., 'solve x + 5 = 10', 'what is 15% of 200')")
        self.problem_input.setFont(QFont("Arial", 11))
        self.problem_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 6px;
                font-size: 11pt;
            }
            QLineEdit:focus {
                border-color: #667eea;
            }
        """)
        self.problem_input.returnPressed.connect(self.solve_problem)
        input_layout.addWidget(self.problem_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.solve_button = QPushButton("🔍 Solve Step-by-Step")
        self.solve_button.setFont(QFont("Arial", 10, QFont.Bold))
        self.solve_button.setStyleSheet("""
            QPushButton {
                background: #667eea;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #764ba2;
            }
        """)
        self.solve_button.clicked.connect(self.solve_problem)
        
        self.force_detailed_button = QPushButton("🔧 Force Detailed Steps")
        self.force_detailed_button.setFont(QFont("Arial", 10))
        self.force_detailed_button.setStyleSheet("""
            QPushButton {
                background: #f093fb;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #f5576c;
            }
        """)
        self.force_detailed_button.clicked.connect(lambda: self.solve_problem(force_detailed=True))
        
        button_layout.addWidget(self.solve_button)
        button_layout.addWidget(self.force_detailed_button)
        input_layout.addLayout(button_layout)
        
        layout.addWidget(input_group)
        
        # Solution display area
        self.solution_area = QScrollArea()
        self.solution_widget = QWidget()
        self.solution_layout = QVBoxLayout(self.solution_widget)
        self.solution_area.setWidget(self.solution_widget)
        self.solution_area.setWidgetResizable(True)
        self.solution_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: #fafafa;
            }
        """)
        layout.addWidget(self.solution_area)
        
        # Suggestions panel
        self.suggestions_group = QGroupBox("💡 Suggestions & Next Steps")
        self.suggestions_layout = QVBoxLayout(self.suggestions_group)
        self.suggestions_group.hide()  # Initially hidden
        layout.addWidget(self.suggestions_group)
        
        # Quick examples
        examples_group = QGroupBox("📚 Quick Examples")
        examples_layout = QVBoxLayout(examples_group)
        
        example_problems = [
            "solve 2x + 3 = 11",
            "what is 25% of 80",
            "simplify (x + 3)²",
            "find 15 + 23 * 2"
        ]
        
        for problem in example_problems:
            example_btn = QPushButton(f"Try: {problem}")
            example_btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 6px;
                    border: 1px solid #ddd;
                    background: white;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background: #f0f0f0;
                }
            """)
            example_btn.clicked.connect(lambda checked, p=problem: self.load_example(p))
            examples_layout.addWidget(example_btn)
        
        layout.addWidget(examples_group)
    
    def solve_problem(self, force_detailed: bool = False):
        """Solve the mathematical problem with step-by-step breakdown"""
        problem = self.problem_input.text().strip()
        if not problem:
            QMessageBox.warning(self, "Input Required", "Please enter a mathematical problem to solve.")
            return
        
        # Clear previous solution
        self.clear_solution_display()
        
        # Show loading indicator
        loading_label = QLabel("🔄 Analyzing your problem...")
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setFont(QFont("Arial", 12))
        loading_label.setStyleSheet("color: #667eea; padding: 20px;")
        self.solution_layout.addWidget(loading_label)
        
        # Process with solver
        QApplication.processEvents()  # Update UI
        
        try:
            result = self.solver.solve_step_by_step(problem, force_detailed)
            self.display_solution(result)
        except Exception as e:
            self.display_error(f"Error solving problem: {str(e)}")
        
        # Remove loading indicator
        loading_label.deleteLater()
    
    def display_solution(self, result: Dict[str, Any]):
        """Display the solution result in the UI"""
        if not result['success']:
            self.display_error("Unable to solve the problem. Please check your input.")
            return
        
        # Handle different types of responses
        if result.get('is_simple'):
            self.display_simple_answer(result)
        elif result.get('is_repetitive'):
            self.display_repetitive_response(result)
        else:
            self.display_detailed_solution(result)
        
        # Show suggestions
        if 'suggestions' in result:
            self.display_suggestions(result['suggestions'])
    
    def display_simple_answer(self, result: Dict[str, Any]):
        """Display simple calculation result"""
        # Answer card
        answer_card = self.create_card("🎯 Quick Answer", f"**{result['direct_answer']}**")
        self.solution_layout.addWidget(answer_card)
        
        # Offer detailed steps
        offer_widget = QWidget()
        offer_layout = QHBoxLayout(offer_widget)
        
        offer_label = QLabel(result['offer_steps'])
        offer_label.setWordWrap(True)
        
        show_steps_btn = QPushButton("Show Detailed Steps")
        show_steps_btn.setStyleSheet("""
            QPushButton {
                background: #4ecdc4;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
        """)
        show_steps_btn.clicked.connect(lambda: self.solve_problem(force_detailed=True))
        
        offer_layout.addWidget(offer_label)
        offer_layout.addWidget(show_steps_btn)
        self.solution_layout.addWidget(offer_widget)
    
    def display_repetitive_response(self, result: Dict[str, Any]):
        """Display non-repetitive response for recently solved problems"""
        # Non-repetitive message
        message_card = self.create_card("🔄 Smart Memory", result['message'])
        self.solution_layout.addWidget(message_card)
        
        # Quick answer
        answer_card = self.create_card("⚡ Quick Answer", f"**{result['quick_answer']}**")
        self.solution_layout.addWidget(answer_card)
        
        # Action buttons
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        
        show_full_btn = QPushButton("📖 Show Full Steps")
        show_full_btn.clicked.connect(lambda: self.solve_problem(force_detailed=True))
        
        try_different_btn = QPushButton("🔄 Try Different Approach")
        try_different_btn.clicked.connect(self.suggest_alternative)
        
        actions_layout.addWidget(show_full_btn)
        actions_layout.addWidget(try_different_btn)
        self.solution_layout.addWidget(actions_widget)
    
    def display_detailed_solution(self, result: Dict[str, Any]):
        """Display detailed step-by-step solution"""
        solution = result['solution']
        self.current_solution = solution
        
        # Problem header
        problem_card = self.create_card("📋 Problem", solution.problem)
        self.solution_layout.addWidget(problem_card)
        
        # Metadata
        metadata_text = f"""
        **Domain:** {solution.mathematical_domain.title()}  
        **Difficulty:** {solution.difficulty_level}  
        **Solution Time:** {solution.solution_time:.2f}s  
        **Key Concepts:** {', '.join(solution.key_concepts)}
        """
        metadata_card = self.create_card("📊 Analysis", metadata_text)
        self.solution_layout.addWidget(metadata_card)
        
        # Steps
        steps_group = QGroupBox("🔧 Step-by-Step Solution")
        steps_layout = QVBoxLayout(steps_group)
        
        for step in solution.steps:
            step_widget = self.create_step_widget(step)
            steps_layout.addWidget(step_widget)
        
        self.solution_layout.addWidget(steps_group)
        
        # Final answer
        final_card = self.create_card("🎉 Final Answer", f"**{solution.final_answer}**", "#27ae60")
        self.solution_layout.addWidget(final_card)
        
        # Real-world context
        if solution.real_world_context:
            context_card = self.create_card("🌍 Real-World Connection", solution.real_world_context)
            self.solution_layout.addWidget(context_card)
        
        # Alternative methods
        if solution.alternative_methods:
            alt_text = "• " + "\n• ".join(solution.alternative_methods)
            alt_card = self.create_card("🔀 Alternative Methods", alt_text)
            self.solution_layout.addWidget(alt_card)
    
    def create_card(self, title: str, content: str, color: str = "#667eea") -> QGroupBox:
        """Create a styled card widget"""
        card = QGroupBox(title)
        card.setFont(QFont("Arial", 10, QFont.Bold))
        
        layout = QVBoxLayout(card)
        content_label = QLabel(content)
        content_label.setWordWrap(True)
        content_label.setFont(QFont("Arial", 10))
        
        # Support basic markdown-style formatting
        formatted_content = content.replace("**", "<b>").replace("**", "</b>")
        content_label.setText(formatted_content)
        
        layout.addWidget(content_label)
        
        card.setStyleSheet(f"""
            QGroupBox {{
                background: white;
                border: 2px solid {color};
                border-radius: 8px;
                margin-top: 1ex;
                padding: 5px;
            }}
            QGroupBox::title {{
                color: {color};
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        
        return card
    
    def create_step_widget(self, step: SolutionStep) -> QWidget:
        """Create a widget for displaying a solution step"""
        step_widget = QFrame()
        step_widget.setFrameStyle(QFrame.Box)
        step_widget.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                margin: 2px;
                padding: 5px;
            }
        """)
        
        layout = QVBoxLayout(step_widget)
        
        # Step header
        header_layout = QHBoxLayout()
        
        step_number = QLabel(f"Step {step.step_number}")
        step_number.setFont(QFont("Arial", 10, QFont.Bold))
        step_number.setStyleSheet("color: #667eea;")
        
        principle = QLabel(f"[{step.principle}]")
        principle.setFont(QFont("Arial", 9))
        principle.setStyleSheet("color: #888; font-style: italic;")
        
        header_layout.addWidget(step_number)
        header_layout.addStretch()
        header_layout.addWidget(principle)
        
        layout.addLayout(header_layout)
        
        # Step action
        action_label = QLabel(f"**Action:** {step.action}")
        action_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(action_label)
        
        # Step explanation
        explanation_label = QLabel(f"**Why:** {step.explanation}")
        explanation_label.setWordWrap(True)
        explanation_label.setFont(QFont("Arial", 10))
        layout.addWidget(explanation_label)
        
        # Before/After equations
        if step.equation_before != step.equation_after:
            equation_layout = QHBoxLayout()
            
            before_label = QLabel(step.equation_before)
            before_label.setFont(QFont("Courier", 10))
            before_label.setStyleSheet("background: #fff3cd; padding: 4px; border-radius: 3px;")
            
            arrow_label = QLabel("→")
            arrow_label.setFont(QFont("Arial", 12, QFont.Bold))
            arrow_label.setStyleSheet("color: #667eea;")
            
            after_label = QLabel(step.equation_after)
            after_label.setFont(QFont("Courier", 10))
            after_label.setStyleSheet("background: #d4edda; padding: 4px; border-radius: 3px;")
            
            equation_layout.addWidget(before_label)
            equation_layout.addWidget(arrow_label)
            equation_layout.addWidget(after_label)
            equation_layout.addStretch()
            
            layout.addLayout(equation_layout)
        
        return step_widget
    
    def display_suggestions(self, suggestions: List[str]):
        """Display suggestions and next steps"""
        # Clear previous suggestions
        for i in reversed(range(self.suggestions_layout.count())):
            self.suggestions_layout.itemAt(i).widget().setParent(None)
        
        for suggestion in suggestions:
            suggestion_btn = QPushButton(f"💡 {suggestion}")
            suggestion_btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 8px;
                    border: 1px solid #ddd;
                    background: #f8f9fa;
                    border-radius: 4px;
                    margin: 2px;
                }
                QPushButton:hover {
                    background: #e9ecef;
                    border-color: #667eea;
                }
            """)
            suggestion_btn.clicked.connect(lambda checked, s=suggestion: self.handle_suggestion(s))
            self.suggestions_layout.addWidget(suggestion_btn)
        
        self.suggestions_group.show()
    
    def handle_suggestion(self, suggestion: str):
        """Handle suggestion button clicks"""
        if "Show detailed steps" in suggestion:
            self.solve_problem(force_detailed=True)
        elif "Try a similar problem" in suggestion:
            problems = self.solver.suggest_practice_problems()
            if problems:
                self.problem_input.setText(problems[0])
        elif "alternative" in suggestion.lower() or "different" in suggestion.lower():
            self.suggest_alternative()
        else:
            # Generic suggestion handling
            QMessageBox.information(self, "Suggestion", f"Explore: {suggestion}")
    
    def suggest_alternative(self):
        """Suggest alternative approaches"""
        if self.current_solution and self.current_solution.alternative_methods:
            methods = "\n".join(f"• {method}" for method in self.current_solution.alternative_methods)
            QMessageBox.information(self, "Alternative Methods", 
                                  f"Here are other ways to solve this problem:\n\n{methods}")
        else:
            QMessageBox.information(self, "Alternative Methods", 
                                  "Try graphing the equation or using a different algebraic approach!")
    
    def load_example(self, problem: str):
        """Load an example problem"""
        self.problem_input.setText(problem)
        self.solve_problem()
    
    def display_error(self, error_message: str):
        """Display error message"""
        error_card = self.create_card("❌ Error", error_message, "#e74c3c")
        self.solution_layout.addWidget(error_card)
    
    def clear_solution_display(self):
        """Clear the solution display area"""
        for i in reversed(range(self.solution_layout.count())):
            child = self.solution_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        self.suggestions_group.hide()


class IntelligentSolverDialog(QDialog):
    """Standalone dialog for the intelligent solver"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🧠 SAUNet 4.0 - Intelligent Step-by-Step Solver")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        layout = QVBoxLayout(self)
        
        # Add the step-by-step widget
        self.solver_widget = StepByStepWidget(self)
        layout.addWidget(self.solver_widget)
        
        # Add dialog buttons
        button_layout = QHBoxLayout()
        
        self.session_btn = QPushButton("📊 Session Summary")
        self.session_btn.clicked.connect(self.show_session_summary)
        
        self.practice_btn = QPushButton("🎯 Practice Problems")
        self.practice_btn.clicked.connect(self.show_practice_problems)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(self.session_btn)
        button_layout.addWidget(self.practice_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def show_session_summary(self):
        """Show session summary"""
        summary = self.solver_widget.solver.get_session_summary()
        summary_text = f"""
        **Problems Solved:** {summary['problems_solved']}
        **Domains Practiced:** {', '.join(summary['domains_practiced'])}
        **Most Common Domain:** {summary['most_common_domain'] or 'None'}
        **Total Patterns:** {summary['total_patterns']}
        """
        QMessageBox.information(self, "Session Summary", summary_text)
    
    def show_practice_problems(self):
        """Show practice problems"""
        problems = self.solver_widget.solver.suggest_practice_problems()
        if problems:
            problems_text = "\n".join(f"• {problem}" for problem in problems)
            msg = QMessageBox(self)
            msg.setWindowTitle("Practice Problems")
            msg.setText("Here are some practice problems based on your session:")
            msg.setDetailedText(problems_text)
            msg.exec_()
        else:
            QMessageBox.information(self, "Practice Problems", 
                                  "Solve a few problems first, and I'll suggest practice exercises!")


def test_intelligent_solver_ui():
    """Test the intelligent solver UI"""
    app = QApplication(sys.argv)
    
    dialog = IntelligentSolverDialog()
    dialog.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    test_intelligent_solver_ui() 