import sys
import math
import re
import threading
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# PyQt5 imports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Import calculator and solver modules
try:
    from calculator import Calculator
    from solver import EquationSolver
    import sympy as sp
except ImportError as e:
    print(f"⚠️ Calculator modules not available: {e}")
    # Create dummy classes to prevent errors
    class Calculator:
        def __init__(self): 
            self.memory = 0
        def process_text_command(self, text):
            return "Calculator not available"
    class EquationSolver:
        def __init__(self): pass
        def solve_equation(self, equation):
            return "No solution", "Solver not available"
    import sympy as sp

# Try importing OCR dependencies with better error handling
OCR_AVAILABLE = False
try:
    import easyocr
    import cv2
    from solve_from_image import solve_from_image
    OCR_AVAILABLE = True
    print("✅ OCR libraries available")
except ImportError as e:
    print(f"⚠️ OCR not available: {e}")
    print("💡 Install with: pip install easyocr opencv-python")

# Import MathOCR class for UI compatibility  
try:
    from ocr import MathOCR
except ImportError:
    # Create a dummy MathOCR class if the module is missing
    class MathOCR:
        def extract(self, image_path):
            return "OCR module not available"

# Voice recognition availability
VOICE_AVAILABLE = False
try:
    from voice_recognition import VoiceControlledCalculator
    VOICE_AVAILABLE = True
    print("✅ Voice recognition available")
except ImportError as e:
    print(f"⚠️ Voice recognition not available: {e}")
    print("💡 Install with: pip install speechrecognition pyaudio pyttsx3")

# Try importing the advanced NLU system
try:
    from advanced_nlu import AdvancedMathNLU
    ADVANCED_NLU_AVAILABLE = True
    print("🚀 Advanced NLU system loaded successfully")
except ImportError:
    ADVANCED_NLU_AVAILABLE = False
    print("⚠️ Advanced NLU not available. Install dependencies: pip install transformers spacy torch")

# Add intelligent solver import at the top with other imports
try:
    from intelligent_math_solver import IntelligentMathSolver
    INTELLIGENT_SOLVER_AVAILABLE = True
    print("🧠 Intelligent Step-by-Step Solver available")
except ImportError:
    INTELLIGENT_SOLVER_AVAILABLE = False
    print("⚠️ Intelligent Step-by-Step Solver not available")
    # Create dummy class
    class IntelligentMathSolver:
        def __init__(self): pass
        def solve_step_by_step(self, problem, force=False):
            return {'success': False, 'error': 'Solver not available'}

# Font settings
FONT_FAMILY = "Roboto"  # or "Fira Code"
FONT_SIZE = 12
ICON_SIZE = 24

# Color scheme for light mode
LIGHT_COLORS = {
    "background": "#FFFFFF",
    "text": "#212121",
    "button": "#F5F5F5",
    "button_text": "#212121",
    "highlight": "#1976D2",
    "secondary": "#BBDEFB",
    "error": "#D32F2F",
    "success": "#388E3C"
}

# Color scheme for dark mode
DARK_COLORS = {
    "background": "#1E1E2E",        # Darker background for better contrast
    "text": "#FFFFFF",              # Pure white for maximum text visibility
    "button": "#313244",            # Slightly lighter than background
    "button_text": "#FFFFFF",       # Pure white for button text
    "highlight": "#74C7EC",         # Brighter blue for highlighted items
    "secondary": "#45475A",         # Mid-tone for secondary elements
    "error": "#F38BA8",             # Brighter red for errors
    "success": "#A6E3A1"            # Brighter green for success messages
}





class CalculatorUI(QMainWindow):
    """Main window of the scientific calculator application."""
    
    def __init__(self, calculator=None, solver=None):
        super().__init__()
        
        # Initialize calculator and solver instances
        self.calculator = calculator or Calculator()
        self.solver = solver or EquationSolver()
        
        # Thread management
        self.active_threads = []
        
        # Initialize OCR if available
        if OCR_AVAILABLE:
            self.ocr = MathOCR()
        else:
            self.ocr = None
        
        # Initialize Advanced NLU if available
        if ADVANCED_NLU_AVAILABLE:
            print("🧠 Initializing Advanced NLU system...")
            self.advanced_nlu = AdvancedMathNLU()
            print("✅ Advanced NLU system ready!")
        else:
            self.advanced_nlu = None
        
        # Initialize Voice Recognition if available
        if VOICE_AVAILABLE:
            print("🎤 Initializing Voice Recognition system...")
            self.voice_controller = VoiceControlledCalculator(self)
            # Connect voice input callback
            self.voice_controller.voice_recognizer.set_callback(self.process_voice_input)
            print("✅ Voice Recognition system ready!")
        else:
            self.voice_controller = None
        
        # Initialize Intelligent Step-by-Step Solver
        if INTELLIGENT_SOLVER_AVAILABLE:
            print("🧠 Initializing Intelligent Step-by-Step Solver...")
            self.intelligent_solver = IntelligentMathSolver()
            print("✅ Intelligent Step-by-Step Solver ready!")
        else:
            self.intelligent_solver = None
        
        # Animation and transition setup
        self.fade_animation = None
        self.current_tab_index = 0
        self.auto_clear_on_tab_switch = True
        
        # SAUNet 4.0 New Features
        self.calculation_history = []
        self.history_index = -1
        self.version = "4.0"
        
        # Set window properties
        self.setWindowTitle("SAUNet 4.0 Advanced Scientific Calculator")
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)  # Better default size
        
        # Ensure status bar and menu bar are created
        if self.statusBar() is None:
            self.setStatusBar(QStatusBar(self))
        if self.menuBar() is None:
            self.setMenuBar(QMenuBar(self))
        
        # Create UI components
        self.init_ui()
        
        # Set light theme as default
        self.is_dark_mode = False
        self.apply_theme()
        
        # Show welcome message for SAUNet 4.0
        self.show_welcome_message()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Create the central widget and main layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create top toolbar with image and voice features
        self.create_top_toolbar()
        
        # Create the top display area
        self.create_display_area()
        
        # Create results and steps area
        self.create_results_area()
        
        # Create tabs for different calculator functions
        self.create_tabs()
        
        # Add status bar
        status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
        if status_bar:
            status_bar.showMessage("Ready")
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        if menubar is None:
            menubar = QMenuBar(self)
            self.setMenuBar(menubar)
        
        # File menu
        file_menu = menubar.addMenu("File") if menubar else None
        
        # Clear action
        clear_action = QAction("Clear All", self)
        clear_action.setShortcut("Ctrl+N")
        clear_action.triggered.connect(self.clear_all)
        if file_menu:
            file_menu.addAction(clear_action)
        
        # Export results action
        export_action = QAction("Export Results", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self.export_results)
        if file_menu:
            file_menu.addAction(export_action)
        
        if file_menu:
            file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(lambda: (self.close(), None)[-1])
        if file_menu:
            file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit") if menubar else None
        
        # Copy action
        copy_action = QAction("Copy Result", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.copy_result)
        if edit_menu:
            edit_menu.addAction(copy_action)
        
        # Paste action
        paste_action = QAction("Paste", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self.paste_expression)
        if edit_menu:
            edit_menu.addAction(paste_action)
        
        # View menu
        view_menu = menubar.addMenu("View") if menubar else None
        
        # Toggle theme action
        self.theme_action = QAction("Toggle Dark Mode", self)
        self.theme_action.setShortcut("Ctrl+T")
        self.theme_action.triggered.connect(self.toggle_theme)
        if view_menu:
            view_menu.addAction(self.theme_action)
        
        # History menu (NEW in SAUNet 4.0)
        history_menu = menubar.addMenu("History") if menubar else None
        
        # Previous calculation action
        prev_action = QAction("Previous Calculation", self)
        prev_action.setShortcut("Ctrl+Up")
        prev_action.triggered.connect(self.previous_calculation)
        if history_menu:
            history_menu.addAction(prev_action)
        
        # Next calculation action
        next_action = QAction("Next Calculation", self)
        next_action.setShortcut("Ctrl+Down")
        next_action.triggered.connect(self.next_calculation)
        if history_menu:
            history_menu.addAction(next_action)
        
        if history_menu:
            history_menu.addSeparator()
        
        # Clear history action
        clear_history_action = QAction("Clear History", self)
        clear_history_action.setShortcut("Ctrl+Shift+H")
        clear_history_action.triggered.connect(self.clear_history)
        if history_menu:
            history_menu.addAction(clear_history_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help") if menubar else None
        
        # About action
        about_action = QAction("About SAUNet 4.0", self)
        about_action.triggered.connect(self.show_about)
        if help_menu:
            help_menu.addAction(about_action)
    
    def create_top_toolbar(self):
        """Create a mobile-like toolbar with image and voice features in the top right."""
        # Create toolbar widget
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        # Add title/logo on the left
        title_label = QLabel("🧮 SAUNet 4.0")
        title_label.setFont(QFont(FONT_FAMILY, FONT_SIZE + 3, QFont.Bold))
        title_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        toolbar_layout.addWidget(title_label)
        
        # Add stretch to push buttons to the right
        toolbar_layout.addStretch()
        
        # Create buttons container for the right side (mobile-like)
        buttons_container = QWidget()
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)
        
        # Voice Recognition button (mobile-style)
        self.toolbar_voice_button = QPushButton("🎤")
        self.toolbar_voice_button.setFixedSize(45, 45)
        self.toolbar_voice_button.setToolTip("Voice Recognition\nSpeak mathematical questions")
        self.toolbar_voice_button.setStyleSheet("""
            QPushButton {
                border: 2px solid #4CAF50;
                border-radius: 22px;
                background-color: #E8F5E8;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:pressed {
                background-color: #45a049;
            }
            QPushButton:checked {
                background-color: #FF5722;
                border-color: #FF5722;
                color: white;
            }
        """)
        if VOICE_AVAILABLE:
            self.toolbar_voice_button.setCheckable(True)
            self.toolbar_voice_button.clicked.connect(self.toggle_voice_recognition)
        else:
            self.toolbar_voice_button.setEnabled(False)
            self.toolbar_voice_button.setToolTip("Voice recognition not available")
        buttons_layout.addWidget(self.toolbar_voice_button)
        
        # Camera/OCR button (mobile-style)
        self.toolbar_camera_button = QPushButton("📷")
        self.toolbar_camera_button.setFixedSize(45, 45)
        self.toolbar_camera_button.setToolTip("Image Recognition\nUpload math images")
        self.toolbar_camera_button.setStyleSheet("""
            QPushButton {
                border: 2px solid #2196F3;
                border-radius: 22px;
                background-color: #E3F2FD;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2196F3;
                color: white;
            }
            QPushButton:pressed {
                background-color: #1976D2;
            }
        """)
        if OCR_AVAILABLE:
            self.toolbar_camera_button.clicked.connect(self.show_ocr_dialog)
        else:
            self.toolbar_camera_button.setEnabled(False)
            self.toolbar_camera_button.setToolTip("OCR not available")
        buttons_layout.addWidget(self.toolbar_camera_button)
        
        # Settings/Menu button (mobile-style)
        self.toolbar_menu_button = QPushButton("⚙️")
        self.toolbar_menu_button.setFixedSize(45, 45)
        self.toolbar_menu_button.setToolTip("Settings & Options")
        self.toolbar_menu_button.setStyleSheet("""
            QPushButton {
                border: 2px solid #9C27B0;
                border-radius: 22px;
                background-color: #F3E5F5;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9C27B0;
                color: white;
            }
            QPushButton:pressed {
                background-color: #7B1FA2;
            }
        """)
        self.toolbar_menu_button.clicked.connect(self.show_settings_menu)
        buttons_layout.addWidget(self.toolbar_menu_button)
        
        # History button (mobile-style)
        self.toolbar_history_button = QPushButton("📋")
        self.toolbar_history_button.setFixedSize(45, 45)
        self.toolbar_history_button.setToolTip("Calculation History")
        self.toolbar_history_button.setStyleSheet("""
            QPushButton {
                border: 2px solid #FF9800;
                border-radius: 22px;
                background-color: #FFF3E0;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF9800;
                color: white;
            }
            QPushButton:pressed {
                background-color: #F57C00;
            }
        """)
        self.toolbar_history_button.clicked.connect(self.show_history_dialog)
        buttons_layout.addWidget(self.toolbar_history_button)
        
        # Add buttons container to main toolbar
        toolbar_layout.addWidget(buttons_container)
        
        # Style the toolbar
        toolbar_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                border-bottom: 2px solid #dee2e6;
            }
        """)
        
        # Add toolbar to main layout
        self.main_layout.addWidget(toolbar_widget)
    
    def show_settings_menu(self):
        """Show settings popup menu (mobile-style)."""
        menu = QMenu(self)
        
        # Theme toggle
        theme_action = QAction("🌙 Toggle Dark Mode", self)
        theme_action.triggered.connect(self.toggle_theme)
        menu.addAction(theme_action)
        
        menu.addSeparator()
        
        # Voice settings
        if VOICE_AVAILABLE:
            voice_help_action = QAction("🎤 Voice Help", self)
            voice_help_action.triggered.connect(self.show_voice_help)
            menu.addAction(voice_help_action)
        
        # OCR settings
        if OCR_AVAILABLE:
            ocr_enhancement_action = QAction("📷 OCR Enhancement", self)
            ocr_enhancement_action.triggered.connect(self.show_enhancement_dialog)
            menu.addAction(ocr_enhancement_action)
        
        menu.addSeparator()
        
        # Export options
        export_action = QAction("💾 Export Results", self)
        export_action.triggered.connect(self.export_results)
        menu.addAction(export_action)
        
        # Clear all
        clear_action = QAction("🗑️ Clear All", self)
        clear_action.triggered.connect(self.clear_all)
        menu.addAction(clear_action)
        
        menu.addSeparator()
        
        # About
        about_action = QAction("ℹ️ About SAUNet 4.0", self)
        about_action.triggered.connect(self.show_about)
        menu.addAction(about_action)
        
        # Show menu at button position
        button_rect = self.toolbar_menu_button.geometry()
        menu.exec_(self.toolbar_menu_button.mapToGlobal(button_rect.bottomLeft()))
    
    def show_history_dialog(self):
        """Show calculation history in a mobile-style dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("📋 Calculation History")
        dialog.setMinimumSize(400, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("📋 Recent Calculations")
        title.setFont(QFont(FONT_FAMILY, FONT_SIZE + 2, QFont.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # History list
        history_list = QListWidget()
        history_list.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        
        # Add history items
        if self.calculation_history:
            for i, (expression, result) in enumerate(reversed(self.calculation_history)):
                item_text = f"{expression} = {result}"
                history_list.addItem(item_text)
        else:
            history_list.addItem("No calculations yet...")
        
        layout.addWidget(history_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Clear history button
        clear_btn = QPushButton("🗑️ Clear History")
        def clear_and_close():
            self.clear_history()
            dialog.close()
        clear_btn.clicked.connect(clear_and_close)
        button_layout.addWidget(clear_btn)
        
        # Use selected button
        use_btn = QPushButton("📝 Use Selected")
        use_btn.clicked.connect(lambda: self.use_history_item(history_list, dialog))
        button_layout.addWidget(use_btn)
        
        # Close button
        close_btn = QPushButton("❌ Close")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # Style the dialog
        dialog.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
                border-radius: 10px;
            }
            QListWidget {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #007bff;
            }
            QPushButton {
                padding: 8px 16px;
                border: 2px solid #007bff;
                border-radius: 6px;
                background-color: #007bff;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        
        dialog.exec_()
    
    def use_history_item(self, history_list, dialog):
        """Use the selected history item."""
        current_item = history_list.currentItem()
        if current_item:
            item_text = current_item.text()
            if "=" in item_text:
                expression = item_text.split("=")[0].strip()
                self.input_field.setText(expression)
                dialog.close()
    
    def create_display_area(self):
        """Create the display area for input and results with enhanced scrolling."""
        display_layout = QVBoxLayout()
        
        # Expression input field with history support and scroll wheel functionality
        self.input_field = ScrollableLineEdit()
        self.input_field.setPlaceholderText("Enter expression or equation... (🖱️ Use scroll wheel for navigation, Ctrl+Scroll to modify numbers)")
        self.input_field.setFont(QFont(FONT_FAMILY, FONT_SIZE + 2))
        self.input_field.returnPressed.connect(self.evaluate_expression)
        self.input_field.setMinimumHeight(40)
        
        # Style the input field
        colors = DARK_COLORS if hasattr(self, 'is_dark_mode') and self.is_dark_mode else LIGHT_COLORS
        self.input_field.setStyleSheet(f"""
            ScrollableLineEdit {{
                background-color: {colors.get('background', '#FFFFFF')};
                color: {colors.get('text', '#212121')};
                border: 2px solid {colors.get('secondary', '#BBDEFB')};
                border-radius: 8px;
                padding: 10px;
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE + 2}px;
            }}
            ScrollableLineEdit:focus {{
                border-color: {colors.get('highlight', '#1976D2')};
                background-color: {colors.get('button', '#F5F5F5')};
            }}
        """)
        
        display_layout.addWidget(self.input_field)
        
        # Buttons below the input field
        buttons_layout = QHBoxLayout()
        
        # Evaluate button
        self.eval_button = QPushButton("Evaluate")
        self.eval_button.clicked.connect(self.evaluate_expression)
        buttons_layout.addWidget(self.eval_button)
        
        # Clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_input)
        buttons_layout.addWidget(clear_button)
        
        # OCR button
        ocr_button = QPushButton("📷 OCR")
        if OCR_AVAILABLE:
            ocr_button.setToolTip("Upload image to extract mathematical expressions")
        else:
            ocr_button.setToolTip("OCR not available - install easyocr and pytesseract")
            ocr_button.setEnabled(False)
        ocr_button.clicked.connect(self.show_ocr_dialog)
        buttons_layout.addWidget(ocr_button)
        
        # Voice Recognition button
        self.voice_button = QPushButton("🎤 Voice")
        if VOICE_AVAILABLE:
            self.voice_button.setToolTip("🎤 Voice Recognition - Click to start speaking mathematical questions!")
            self.voice_button.setCheckable(True)  # Toggle button
            self.voice_button.clicked.connect(self.toggle_voice_recognition)
        else:
            self.voice_button.setToolTip("Voice recognition not available - install speechrecognition and pyaudio")
            self.voice_button.setEnabled(False)
        buttons_layout.addWidget(self.voice_button)
        
        # Add voice help button
        if VOICE_AVAILABLE:
            voice_help_button = QPushButton("❓")
            voice_help_button.setToolTip("Voice Recognition Help & Examples")
            voice_help_button.setMaximumWidth(30)
            voice_help_button.clicked.connect(self.show_voice_help)
            buttons_layout.addWidget(voice_help_button)
        
        display_layout.addLayout(buttons_layout)
        self.main_layout.addLayout(display_layout)
    
    def create_results_area(self):
        """Create the enhanced area for displaying calculation results with popup functionality."""
        # Create a splitter to divide results and steps
        self.results_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Enhanced results text display with clickable elements
        self.results_display = QTextBrowser()  # Changed from QTextEdit to QTextBrowser for clickable links
        self.results_display.setReadOnly(True)
        self.results_display.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        self.results_display.setPlaceholderText("Results will appear here...\n💡 Click on results to view detailed solutions!")
        self.results_display.setMinimumHeight(120)
        self.results_display.setOpenExternalLinks(False)  # Handle links internally
        self.results_display.anchorClicked.connect(self.handle_result_click)
        
        # Enhanced steps text display with smooth scrolling
        self.steps_display = EnhancedTextEdit()
        self.steps_display.setReadOnly(True)
        self.steps_display.setFont(QFont("Consolas", FONT_SIZE - 1))
        self.steps_display.setPlaceholderText("Step-by-step solution will appear here...\n💡 Use mouse wheel for smooth scrolling!")
        self.steps_display.setMinimumHeight(150)
        
        # Style the results and steps displays
        colors = DARK_COLORS if hasattr(self, 'is_dark_mode') and self.is_dark_mode else LIGHT_COLORS
        
        results_style = f"""
            QTextBrowser {{
                background-color: {colors.get('background', '#FFFFFF')};
                color: {colors.get('text', '#212121')};
                border: 2px solid {colors.get('secondary', '#BBDEFB')};
                border-radius: 8px;
                padding: 12px;
                selection-background-color: {colors.get('highlight', '#1976D2')};
            }}
            QTextBrowser:focus {{
                border-color: {colors.get('highlight', '#1976D2')};
            }}
        """
        self.results_display.setStyleSheet(results_style)
        
        # Add widgets to splitter
        self.results_splitter.addWidget(self.results_display)
        self.results_splitter.addWidget(self.steps_display)
        self.results_splitter.setSizes([150, 250])
        
        # Store current calculation data for popup
        self.current_calculation = {
            'expression': '',
            'result': '',
            'steps': '',
            'type': 'General'
        }
        
        self.main_layout.addWidget(self.results_splitter, 1)  # Give it a stretch factor of 1
    
    def handle_result_click(self, url):
        """Handle clicking on result links to show detailed popup."""
        if url.toString() == "show_detailed_solution":
            self.show_detailed_solution_popup()
    
    def show_detailed_solution_popup(self):
        """Show the detailed solution popup dialog."""
        if not self.current_calculation['expression']:
            QMessageBox.information(self, "No Solution", "No calculation available to show detailed solution.")
            return
        
        # Create and show the solution popup
        popup = ResultDetailDialog(
            self,
            self.current_calculation['expression'],
            self.current_calculation['result'],
            self.current_calculation['steps'],
            self.current_calculation['type']
        )
        popup.exec_()
    
    def create_tabs(self):
        """Create tabs for different calculator functions."""
        self.tabs = QTabWidget()
        
        # Set up tab style and animations
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)
        self.tabs.setUsesScrollButtons(True)
        
        # Connect tab change signal
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # Basic calculator tab
        self.basic_tab = QWidget()
        self.create_basic_tab()
        self.tabs.addTab(self.basic_tab, "🔢 Basic")
        
        # Scientific calculator tab
        self.scientific_tab = QWidget()
        self.create_scientific_tab()
        self.tabs.addTab(self.scientific_tab, "🧮 Scientific")
        
        # Natural language tab
        self.nl_tab = QWidget()
        self.create_nl_tab()
        self.tabs.addTab(self.nl_tab, "💬 Text Input")
        
        # Intelligent Step-by-Step Solver tab
        if INTELLIGENT_SOLVER_AVAILABLE:
            self.step_solver_tab = QWidget()
            self.create_step_solver_tab()
            self.tabs.addTab(self.step_solver_tab, "🧠 Step-by-Step")
        
        # Other operations tab
        self.other_tab = QWidget()
        self.create_other_tab()
        self.tabs.addTab(self.other_tab, "⚙️ Others")
        
        # Add clear input toggle
        self.create_tab_options()
        
        self.main_layout.addWidget(self.tabs)
    
    def create_basic_tab(self):
        """Create the basic calculator tab."""
        layout = QVBoxLayout(self.basic_tab)
        
        # Create grid for buttons
        grid_layout = QGridLayout()
        
        # Define button texts
        button_texts = [
            ['7', '8', '9', '/', 'C'],
            ['4', '5', '6', '*', '←'],
            ['1', '2', '3', '-', '±'],
            ['0', '.', '=', '+', '%']
        ]
        
        # Create and add buttons to grid
        for row, texts in enumerate(button_texts):
            for col, text in enumerate(texts):
                button = QPushButton(text)
                button.setFont(QFont(FONT_FAMILY, FONT_SIZE))
                button.setMinimumSize(50, 50)
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                
                if text == '=':
                    button.clicked.connect(self.evaluate_expression)
                elif text == 'C':
                    button.clicked.connect(self.clear_input)
                elif text == '←':
                    button.clicked.connect(self.backspace)
                elif text == '±':
                    button.clicked.connect(self.toggle_sign)
                else:
                    # Use a lambda to capture the button text
                    button.clicked.connect(lambda checked, text=text: self.append_to_input(text))
                
                grid_layout.addWidget(button, row, col)
        
        # Memory buttons
        memory_layout = QHBoxLayout()
        
        mc_button = QPushButton("MC")
        mc_button.clicked.connect(self.memory_clear)
        memory_layout.addWidget(mc_button)
        
        mr_button = QPushButton("MR")
        mr_button.clicked.connect(self.memory_recall)
        memory_layout.addWidget(mr_button)
        
        m_plus_button = QPushButton("M+")
        m_plus_button.clicked.connect(self.memory_add)
        memory_layout.addWidget(m_plus_button)
        
        m_minus_button = QPushButton("M-")
        m_minus_button.clicked.connect(self.memory_subtract)
        memory_layout.addWidget(m_minus_button)
        
        # Add layouts to main layout
        layout.addLayout(grid_layout)
        layout.addLayout(memory_layout)
    
    def create_scientific_tab(self):
        """Create the scientific calculator tab."""
        layout = QVBoxLayout(self.scientific_tab)
        
        # Create grid for buttons
        grid_layout = QGridLayout()
        
        # Define button texts
        button_texts = [
            ['sin', 'cos', 'tan', '(', ')'],
            ['asin', 'acos', 'atan', 'π', 'e'],
            ['log', 'ln', '√', '^', '!'],
            ['sinh', 'cosh', 'tanh', '|x|', 'mod'],
            ['x²', 'x³', 'x^y', '1/x', 'x!']
        ]
        
        # Create and add buttons to grid
        for row, texts in enumerate(button_texts):
            for col, text in enumerate(texts):
                button = QPushButton(text)
                button.setFont(QFont(FONT_FAMILY, FONT_SIZE - 1))
                button.setMinimumSize(50, 40)
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                
                # Connect button to appropriate function
                if text == 'π':
                    button.clicked.connect(lambda: self.append_to_input('pi'))
                elif text == 'e':
                    button.clicked.connect(lambda: self.append_to_input('e'))
                elif text == '√':
                    button.clicked.connect(lambda: self.append_to_input('sqrt('))
                elif text == 'x²':
                    button.clicked.connect(lambda: self.append_to_input('**2'))
                elif text == 'x³':
                    button.clicked.connect(lambda: self.append_to_input('**3'))
                elif text == 'x^y':
                    button.clicked.connect(lambda: self.append_to_input('**'))
                elif text == '1/x':
                    button.clicked.connect(self.reciprocal)
                elif text == '|x|':
                    button.clicked.connect(lambda: self.append_to_input('abs('))
                elif text == 'mod':
                    button.clicked.connect(lambda: self.append_to_input('%'))
                elif text == '!':
                    button.clicked.connect(lambda: self.append_to_input('factorial('))
                elif text == 'x!':
                    button.clicked.connect(lambda: self.append_to_input('factorial('))
                else:
                    # For trig functions and other functions
                    button.clicked.connect(lambda checked, t=text: self.append_function(t))
                
                grid_layout.addWidget(button, row, col)
        
        # Add calculus function buttons
        calculus_layout = QHBoxLayout()
        
        derivative_button = QPushButton("Derivative")
        derivative_button.clicked.connect(lambda: self.show_calculus_dialog("derivative"))
        calculus_layout.addWidget(derivative_button)
        
        integral_button = QPushButton("Integral")
        integral_button.clicked.connect(lambda: self.show_calculus_dialog("integral"))
        calculus_layout.addWidget(integral_button)
        
        limit_button = QPushButton("Limit")
        limit_button.clicked.connect(lambda: self.show_calculus_dialog("limit"))
        calculus_layout.addWidget(limit_button)
        
        solve_button = QPushButton("Solve")
        solve_button.clicked.connect(lambda: self.show_calculus_dialog("solve"))
        calculus_layout.addWidget(solve_button)
        
        # Add layouts to main layout
        layout.addLayout(grid_layout)
        layout.addLayout(calculus_layout)
    
    def create_step_solver_tab(self):
        """Create the intelligent step-by-step solver tab."""
        layout = QVBoxLayout(self.step_solver_tab)
        
        # Title and description
        title_label = QLabel("🧠 Intelligent Step-by-Step Mathematical Solver")
        title_label.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #667eea, stop:1 #764ba2);
                color: white; 
                padding: 12px; 
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title_label)
        
        # Problem input section
        input_group = QGroupBox("💭 Ask me anything mathematical!")
        input_layout = QVBoxLayout(input_group)
        
        # Problem input field
        self.step_problem_input = QLineEdit()
        self.step_problem_input.setPlaceholderText("Enter your math problem (e.g., 'solve x + 5 = 10', 'what is 15% of 200', 'simplify 2x + 3x')")
        self.step_problem_input.setFont(QFont(FONT_FAMILY, 11))
        self.step_problem_input.setMinimumHeight(35)
        self.step_problem_input.returnPressed.connect(self.solve_step_by_step)
        input_layout.addWidget(self.step_problem_input)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.solve_step_btn = QPushButton("🔍 Solve Step-by-Step")
        self.solve_step_btn.setFont(QFont(FONT_FAMILY, 10, QFont.Bold))
        self.solve_step_btn.clicked.connect(self.solve_step_by_step)
        
        self.force_detailed_btn = QPushButton("🔧 Force Detailed Steps")
        self.force_detailed_btn.clicked.connect(lambda: self.solve_step_by_step(True))
        
        self.clear_step_btn = QPushButton("🗑️ Clear")
        self.clear_step_btn.clicked.connect(self.clear_step_solution)
        
        button_layout.addWidget(self.solve_step_btn)
        button_layout.addWidget(self.force_detailed_btn)
        button_layout.addWidget(self.clear_step_btn)
        
        input_layout.addLayout(button_layout)
        layout.addWidget(input_group)
        
        # Solution display area
        self.step_solution_display = QTextEdit()
        self.step_solution_display.setReadOnly(True)
        self.step_solution_display.setFont(QFont("Consolas", 10))
        self.step_solution_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.step_solution_display)
        
        # Quick examples section
        examples_group = QGroupBox("📚 Quick Examples - Click to Try")
        examples_layout = QGridLayout(examples_group)
        
        examples = [
            "solve 2x + 3 = 11",
            "what is 25% of 80",
            "simplify (x + 3)²", 
            "find 144 / 12 + 5",
            "solve 3x - 7 = 14",
            "calculate 15 + 6 * 4"
        ]
        
        for i, example in enumerate(examples):
            btn = QPushButton(example)
            btn.setMaximumHeight(30)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 6px;
                    border: 1px solid #ccc;
                    background: white;
                    border-radius: 4px;
                }
                QPushButton:hover { 
                    background: #e3f2fd; 
                    border-color: #667eea;
                }
            """)
            btn.clicked.connect(lambda checked, ex=example: self.load_step_example(ex))
            examples_layout.addWidget(btn, i // 2, i % 2)
        
        examples_group.setMaximumHeight(120)
        layout.addWidget(examples_group)
        
        # Status label
        self.step_status_label = QLabel("Ready to solve mathematical problems with detailed explanations!")
        self.step_status_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(self.step_status_label)
    
    def create_nl_tab(self):
        """Create the natural language input tab with enhanced scrolling."""
        layout = QVBoxLayout(self.nl_tab)
        
        # Title with AI indicator
        title_text = "🧠 AI-Powered Natural Language Mathematics" if ADVANCED_NLU_AVAILABLE else "Natural Language Input"
        instructions = QLabel(title_text)
        instructions.setFont(QFont(FONT_FAMILY, FONT_SIZE + 1, QFont.Bold))
        layout.addWidget(instructions)
        
        # AI Status indicator
        if ADVANCED_NLU_AVAILABLE:
            ai_status = QLabel("✅ Advanced NLU with Transformers & spaCy Ready")
            ai_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            ai_status = QLabel("⚠️ Basic NLU (Install: pip install transformers spacy torch)")
            ai_status.setStyleSheet("color: #FF9800;")
        layout.addWidget(ai_status)
        
        # Enhanced Examples
        voice_indicator = " 🎤" if VOICE_AVAILABLE else ""
        examples_text = (f"🎯 Advanced AI Examples{voice_indicator}:\n"
                        "🔸 Calculus: 'differentiate sin(x²) with respect to x'\n"
                        "🔸 Integration: 'integrate x³ from 0 to 5'\n"
                        "🔸 Equations: 'solve 2x + 3 = 7 for x'\n"
                        "🔸 Limits: 'find the limit of sin(x)/x as x approaches 0'\n"
                        "🔸 Simplify: 'simplify (x + 1)² - x²'\n"
                        "🔸 Expand: 'expand (a + b)³'\n"
                        "🔸 Factor: 'factor x² - 4'\n"
                        "🔸 Calculate: 'what is the square root of 144'")
        examples = QLabel(examples_text)
        examples.setFont(QFont(FONT_FAMILY, FONT_SIZE - 1))
        layout.addWidget(examples)
        
        # Enhanced text input area with smooth scrolling
        self.nl_input = AdvancedTextEdit()
        self.nl_input.setPlaceholderText("Enter your mathematical query here...\n\n💡 Tip: Use the mouse wheel for smooth scrolling and Ctrl+Wheel to zoom!")
        self.nl_input.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        self.nl_input.setMinimumHeight(120)
        self.nl_input.setMaximumHeight(200)
        
        # Add enhanced styling for the text input
        colors = DARK_COLORS if hasattr(self, 'is_dark_mode') and self.is_dark_mode else LIGHT_COLORS
        self.nl_input.setStyleSheet(f"""
            AdvancedTextEdit {{
                background-color: {colors.get('background', '#FFFFFF')};
                color: {colors.get('text', '#212121')};
                border: 2px solid {colors.get('secondary', '#BBDEFB')};
                border-radius: 8px;
                padding: 10px;
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE}px;
                line-height: 1.4;
            }}
            AdvancedTextEdit:focus {{
                border-color: {colors.get('highlight', '#1976D2')};
                background-color: {colors.get('button', '#F5F5F5')};
            }}
        """)
        layout.addWidget(self.nl_input)
        
        # Submit button with enhanced styling
        submit_button = QPushButton("🧮 Calculate with AI")
        submit_button.clicked.connect(self.process_nl_query)
        submit_button.setFont(QFont(FONT_FAMILY, FONT_SIZE, QFont.Bold))
        submit_button.setMinimumHeight(40)
        submit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors.get('highlight', '#1976D2')};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {colors.get('secondary', '#BBDEFB')};
            }}
            QPushButton:pressed {{
                background-color: #1565C0;
            }}
        """)
        layout.addWidget(submit_button)
        
        # Add some stretching space
        layout.addStretch()
    
    def create_other_tab(self):
        """Create the tab for other operations."""
        layout = QVBoxLayout(self.other_tab)
        
        # Create sub-tabs
        sub_tabs = QTabWidget()
        
        # Matrix operations tab
        matrix_tab = QWidget()
        self.create_matrix_tab(matrix_tab)
        sub_tabs.addTab(matrix_tab, "Matrices")
        
        # Base conversion tab
        base_tab = QWidget()
        self.create_base_conversion_tab(base_tab)
        sub_tabs.addTab(base_tab, "Base Conversion")
        
        # Statistics tab
        stats_tab = QWidget()
        self.create_statistics_tab(stats_tab)
        sub_tabs.addTab(stats_tab, "Statistics")
        
        # Unit conversion tab
        unit_tab = QWidget()
        self.create_unit_conversion_tab(unit_tab)
        sub_tabs.addTab(unit_tab, "Unit Conversion")
        
        layout.addWidget(sub_tabs)
    
    def create_tab_options(self):
        """Create options for tab behavior."""
        # Add a small widget above tabs for options
        options_layout = QHBoxLayout()
        
        # Auto-clear checkbox
        self.auto_clear_checkbox = QCheckBox("Clear input when switching tabs")
        self.auto_clear_checkbox.setChecked(self.auto_clear_on_tab_switch)
        self.auto_clear_checkbox.stateChanged.connect(self.toggle_auto_clear)
        options_layout.addWidget(self.auto_clear_checkbox)
        
        # Add some spacing
        options_layout.addStretch()
        
        # Tab transition indicator
        self.transition_indicator = QLabel("Ready")
        self.transition_indicator.setStyleSheet("color: #666; font-style: italic;")
        options_layout.addWidget(self.transition_indicator)
        
        # Insert before tabs
        self.main_layout.insertLayout(self.main_layout.count() - 1, options_layout)
    
    def toggle_auto_clear(self, state):
        """Toggle auto-clear functionality."""
        self.auto_clear_on_tab_switch = state == 2  # Qt.Checked = 2
        status_msg = "Auto-clear enabled" if self.auto_clear_on_tab_switch else "Auto-clear disabled"
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(status_msg, 2000)
    
    def on_tab_changed(self, index):
        """Handle tab change with smooth transitions and auto-clear."""
        if index == self.current_tab_index:
            return
        
        # Get tab names for status
        tab_names = ["Basic", "Scientific", "Text Input", "Others"]
        new_tab_name = tab_names[index] if index < len(tab_names) else "Unknown"
        
        # Update transition indicator
        self.transition_indicator.setText(f"Switching to {new_tab_name}...")
        
        # Apply fade animation
        self.apply_tab_transition_effect(index)
        
        # Auto-clear input if enabled
        if self.auto_clear_on_tab_switch:
            self.clear_input()
            self.clear_results()
        
        # Update current tab index
        self.current_tab_index = index
        
        # Update status bar
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(f"Switched to {new_tab_name} calculator", 3000)
        
        # Reset indicator after transition
        QTimer.singleShot(500, lambda: self.transition_indicator.setText("Ready"))
    
    def apply_tab_transition_effect(self, new_index):
        """Apply smooth transition effect when switching tabs."""
        current_widget = self.tabs.widget(self.current_tab_index)
        new_widget = self.tabs.widget(new_index)
        
        if current_widget and new_widget:
            # Create fade-out effect for current tab
            self.fade_out_effect = QGraphicsOpacityEffect()
            current_widget.setGraphicsEffect(self.fade_out_effect)
            
            # Create fade-in effect for new tab
            self.fade_in_effect = QGraphicsOpacityEffect()
            new_widget.setGraphicsEffect(self.fade_in_effect)
            
            # Fade out animation
            self.fade_out_animation = QPropertyAnimation(self.fade_out_effect, b"opacity")
            self.fade_out_animation.setDuration(200)
            self.fade_out_animation.setStartValue(1.0)
            self.fade_out_animation.setEndValue(0.3)
            self.fade_out_animation.setEasingCurve(QEasingCurve.InOutQuad)
            
            # Fade in animation
            self.fade_in_animation = QPropertyAnimation(self.fade_in_effect, b"opacity")
            self.fade_in_animation.setDuration(300)
            self.fade_in_animation.setStartValue(0.3)
            self.fade_in_animation.setEndValue(1.0)
            self.fade_in_animation.setEasingCurve(QEasingCurve.InOutQuad)
            
            # Start animations
            self.fade_out_animation.start()
            QTimer.singleShot(100, self.fade_in_animation.start)
            
            # Clean up effects after animation
            QTimer.singleShot(400, lambda: self.cleanup_transition_effects())
    
    def cleanup_transition_effects(self):
        """Clean up transition effects after animation completes."""
        try:
            for i in range(self.tabs.count()):
                widget = self.tabs.widget(i)
                if widget:
                    widget.setGraphicsEffect(None)
        except:
            pass  # Ignore cleanup errors
    
    def clear_results(self):
        """Clear the results display areas."""
        self.results_display.clear()
        self.steps_display.clear()
    

    
    def add_to_history(self, expression, result):
        """Add a calculation to the history."""
        if expression and expression.strip():
            # Avoid duplicate consecutive entries
            if not self.calculation_history or self.calculation_history[-1][0] != expression:
                self.calculation_history.append((expression, str(result)))
                # Keep only last 50 calculations
                if len(self.calculation_history) > 50:
                    self.calculation_history.pop(0)
            self.history_index = len(self.calculation_history)
    
    def previous_calculation(self):
        """Navigate to the previous calculation in history."""
        if self.calculation_history and self.history_index > 0:
            self.history_index -= 1
            expression, result = self.calculation_history[self.history_index]
            self.input_field.setText(expression)
            self.results_display.setText(f"Previous: {result}")
            
            status_bar = self.statusBar()
            if status_bar:
                status_bar.showMessage(f"History: {self.history_index + 1}/{len(self.calculation_history)}", 2000)
    
    def next_calculation(self):
        """Navigate to the next calculation in history."""
        if self.calculation_history and self.history_index < len(self.calculation_history) - 1:
            self.history_index += 1
            expression, result = self.calculation_history[self.history_index]
            self.input_field.setText(expression)
            self.results_display.setText(f"Next: {result}")
            
            status_bar = self.statusBar()
            if status_bar:
                status_bar.showMessage(f"History: {self.history_index + 1}/{len(self.calculation_history)}", 2000)
        elif self.history_index >= len(self.calculation_history) - 1:
            # At the end of history, clear the input for new calculation
            self.input_field.clear()
            self.history_index = len(self.calculation_history)
    
    def clear_history(self):
        """Clear the calculation history."""
        self.calculation_history.clear()
        self.history_index = -1
        
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage("Calculation history cleared", 2000)
    
    def show_welcome_message(self):
        """Show welcome message for SAUNet 4.0."""
        voice_feature = "• 🎤 Voice Recognition - Speak math questions naturally!\n" if VOICE_AVAILABLE else ""
        voice_examples = ""
        if VOICE_AVAILABLE:
            voice_examples = (
                "\n🗣️ Voice Examples:\n"
                "• 'Solve two x plus three equals seven'\n"
                "• 'What is the square root of sixteen'\n"
                "• 'Differentiate x squared'\n"
                "• Click the ❓ button for more voice help\n"
            )
        
        welcome_text = (
            "🎉 Welcome to SAUNet 4.0 Advanced Scientific Calculator!\n\n"
            "🆕 New Features in v4.0:\n"
            "• Calculation History (History menu)\n"
            "• Smooth Tab Transitions with auto-clear\n"
            "• Enhanced OCR with image processing\n"
            "• Professional themes and animations\n"
            f"{voice_feature}"
            "• Improved user experience\n\n"
            "🚀 Quick Start:\n"
            "• Use tabs to switch between calculator modes\n"
            "• Try the OCR button to solve math from images\n"
            "• Access calculation history via History menu\n"
            "• Toggle dark mode with Ctrl+T\n"
            f"{voice_examples}\n"
            "Ready to calculate! 🧮✨"
        )
        
        self.results_display.setTextColor(QColor(LIGHT_COLORS["highlight"] if not self.is_dark_mode else DARK_COLORS["highlight"]))
        self.results_display.setText(welcome_text)
        self.results_display.setTextColor(QColor(LIGHT_COLORS["text"] if not self.is_dark_mode else DARK_COLORS["text"]))
        
        # Update status bar
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage("SAUNet 4.0 - Ready for advanced calculations!")
    
    def create_matrix_tab(self, parent):
        """Create the matrix operations tab."""
        layout = QVBoxLayout(parent)
        
        # Matrix A input
        layout.addWidget(QLabel("Matrix A:"))
        self.matrix_a_input = QTextEdit()
        self.matrix_a_input.setPlaceholderText("Enter matrix A (e.g., [1, 2; 3, 4])")
        self.matrix_a_input.setMaximumHeight(100)
        layout.addWidget(self.matrix_a_input)
        
        # Matrix B input
        layout.addWidget(QLabel("Matrix B:"))
        self.matrix_b_input = QTextEdit()
        self.matrix_b_input.setPlaceholderText("Enter matrix B (needed for operations like multiply)")
        self.matrix_b_input.setMaximumHeight(100)
        layout.addWidget(self.matrix_b_input)
        
        # Operation selection
        layout.addWidget(QLabel("Operation:"))
        self.matrix_op_combo = QComboBox()
        self.matrix_op_combo.addItems([
            "Determinant", "Inverse", "Transpose", "Multiply", 
            "Add", "Subtract", "Eigenvalues"
        ])
        layout.addWidget(self.matrix_op_combo)
        
        # Calculate button
        matrix_calc_button = QPushButton("Calculate")
        matrix_calc_button.clicked.connect(self.perform_matrix_operation)
        layout.addWidget(matrix_calc_button)
        
        # Add stretching space
        layout.addStretch()
    
    def create_base_conversion_tab(self, parent):
        """Create the base conversion tab."""
        layout = QVBoxLayout(parent)
        
        # Number input
        layout.addWidget(QLabel("Number:"))
        self.base_input = QLineEdit()
        self.base_input.setPlaceholderText("Enter a number")
        layout.addWidget(self.base_input)
        
        # From base selection
        layout.addWidget(QLabel("From Base:"))
        self.from_base_combo = QComboBox()
        self.from_base_combo.addItems(["2 (Binary)", "8 (Octal)", "10 (Decimal)", "16 (Hexadecimal)"])
        self.from_base_combo.setCurrentIndex(2)  # Default to decimal
        layout.addWidget(self.from_base_combo)
        
        # To base selection
        layout.addWidget(QLabel("To Base:"))
        self.to_base_combo = QComboBox()
        self.to_base_combo.addItems(["2 (Binary)", "8 (Octal)", "10 (Decimal)", "16 (Hexadecimal)"])
        self.to_base_combo.setCurrentIndex(0)  # Default to binary
        layout.addWidget(self.to_base_combo)
        
        # Convert button
        convert_button = QPushButton("Convert")
        convert_button.clicked.connect(self.perform_base_conversion)
        layout.addWidget(convert_button)
        
        # Add stretching space
        layout.addStretch()
    
    def create_statistics_tab(self, parent):
        """Create the statistics tab."""
        layout = QVBoxLayout(parent)
        
        # Data input
        layout.addWidget(QLabel("Data (comma or space separated):"))
        self.stats_input = QTextEdit()
        self.stats_input.setPlaceholderText("Enter data points (e.g., 1, 2, 3, 4, 5)")
        layout.addWidget(self.stats_input)
        
        # Operation selection
        layout.addWidget(QLabel("Operation:"))
        self.stats_op_combo = QComboBox()
        self.stats_op_combo.addItems([
            "Mean", "Median", "Mode", "Standard Deviation", "Variance"
        ])
        layout.addWidget(self.stats_op_combo)
        
        # Calculate button
        stats_calc_button = QPushButton("Calculate")
        stats_calc_button.clicked.connect(self.perform_stats_operation)
        layout.addWidget(stats_calc_button)
        
        # Add stretching space
        layout.addStretch()
    
    def create_unit_conversion_tab(self, parent):
        """Create the unit conversion tab."""
        layout = QVBoxLayout(parent)
        
        # Value input
        layout.addWidget(QLabel("Value:"))
        self.unit_value_input = QLineEdit()
        self.unit_value_input.setPlaceholderText("Enter a value")
        layout.addWidget(self.unit_value_input)
        
        # Unit type selection
        layout.addWidget(QLabel("Unit Type:"))
        self.unit_type_combo = QComboBox()
        self.unit_type_combo.addItems([
            "Length", "Mass", "Volume", "Time", "Temperature", "Area", "Energy", "Pressure"
        ])
        self.unit_type_combo.currentIndexChanged.connect(self.update_unit_combos)
        layout.addWidget(self.unit_type_combo)
        
        # From unit selection
        layout.addWidget(QLabel("From Unit:"))
        self.from_unit_combo = QComboBox()
        layout.addWidget(self.from_unit_combo)
        
        # To unit selection
        layout.addWidget(QLabel("To Unit:"))
        self.to_unit_combo = QComboBox()
        layout.addWidget(self.to_unit_combo)
        
        # Initialize unit combos
        self.update_unit_combos(0)
        
        # Convert button
        unit_convert_button = QPushButton("Convert")
        unit_convert_button.clicked.connect(self.perform_unit_conversion)
        layout.addWidget(unit_convert_button)
        
        # Add stretching space
        layout.addStretch() 

    # Event handlers and utility functions
    def evaluate_expression(self):
        """Evaluate the expression in the input field."""
        expression = self.input_field.text().strip()
        if not expression:
            return
        
        try:
            # Use a thread to avoid UI freezing for complex calculations
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage("Calculating...")
            
            # Start calculation in a separate thread
            calc_thread = CalculationThread(self.calculator, expression)
            calc_thread.result_ready.connect(self.display_result)
            self.active_threads.append(calc_thread)  # Track the thread
            calc_thread.finished.connect(lambda: self.thread_finished(calc_thread))
            calc_thread.start()
        except Exception as e:
            self.display_result(None, str(e))
    
    def display_result(self, result, error=None):
        """Display the calculation result with enhanced popup functionality."""
        # Always clear previous results at the very start
        self.results_display.clear()
        self.steps_display.clear()
        status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
        
        # Get current theme colors
        colors = DARK_COLORS if self.is_dark_mode else LIGHT_COLORS
        
        if error:
            self.results_display.setTextColor(QColor(colors["error"]))
            self.results_display.append(f"❌ Error: {error}")
            self.results_display.setTextColor(QColor(colors["text"]))
            if status_bar:
                status_bar.showMessage("Error in calculation")
            
            # Clear current calculation data on error
            self.current_calculation = {
                'expression': '',
                'result': '',
                'steps': '',
                'type': 'Error'
            }
        else:
            # Get expression from appropriate input field
            expression = ""
            calculation_type = "General"
            
            # Determine source of calculation
            current_tab_index = self.tabs.currentIndex()
            if current_tab_index == 2:  # Natural Language tab
                expression = self.nl_input.toPlainText().strip()
                calculation_type = "Natural Language AI"
            else:
                expression = self.input_field.text().strip()
                if current_tab_index == 1:
                    calculation_type = "Scientific"
                else:
                    calculation_type = "Basic"
            
            # Add to history (SAUNet 4.0 feature)
            if expression:
                if isinstance(result, tuple):
                    value, steps = result
                    self.add_to_history(expression, self._format_result_for_display(value))
                    
                    # Update current calculation data
                    self.current_calculation = {
                        'expression': expression,
                        'result': self._format_result_for_display(value),
                        'steps': steps if steps else "No detailed steps available.",
                        'type': calculation_type
                    }
                else:
                    self.add_to_history(expression, self._format_result_for_display(result))
                    
                    # Update current calculation data
                    self.current_calculation = {
                        'expression': expression,
                        'result': self._format_result_for_display(result),
                        'steps': "Basic calculation - no detailed steps.",
                        'type': calculation_type
                    }
            
            # Format result based on type
            if isinstance(result, tuple):
                value, steps = result
                # Format the result based on its type
                result_str = self._format_result_for_display(value)
                
                # Create clickable result with enhanced formatting
                self.results_display.setHtml(f"""
                <div style="font-family: {FONT_FAMILY}; padding: 10px;">
                    <h3 style="color: {colors['highlight']}; margin: 0; padding: 5px 0;">
                        🎯 Calculation Result
                    </h3>
                    <div style="background: {colors['secondary']}; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        <p style="font-size: 16px; font-weight: bold; color: {colors['text']}; margin: 0;">
                            📝 Expression: <span style="color: {colors['highlight']};">{expression}</span>
                        </p>
                        <p style="font-size: 18px; font-weight: bold; color: {colors['text']}; margin: 10px 0 0 0;">
                            ✅ Result: <span style="color: {colors['success']};">{result_str}</span>
                        </p>
                    </div>
                    <div style="text-align: center; margin: 15px 0;">
                        <a href="show_detailed_solution" style="
                            background: {colors['highlight']}; 
                            color: white; 
                            padding: 12px 24px; 
                            border-radius: 8px; 
                            text-decoration: none; 
                            font-weight: bold;
                            display: inline-block;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                        ">
                            🔬 View Detailed Solution & Steps
                        </a>
                    </div>
                    <p style="color: {colors['text']}; font-size: 12px; text-align: center; margin: 10px 0;">
                        💡 Click the button above to see step-by-step solution, mathematical methods, and analysis
                    </p>
                </div>
                """)
                
                # Display steps if available in the steps panel
                if steps:
                    self.steps_display.setTextColor(QColor(colors["text"]))
                    self.steps_display.setFontFamily("Consolas")
                    self.steps_display.setText(steps)
            else:
                result_str = self._format_result_for_display(result)
                
                # Create clickable result for simple calculations
                self.results_display.setHtml(f"""
                <div style="font-family: {FONT_FAMILY}; padding: 10px;">
                    <h3 style="color: {colors['highlight']}; margin: 0; padding: 5px 0;">
                        🧮 Quick Calculation
                    </h3>
                    <div style="background: {colors['secondary']}; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        <p style="font-size: 16px; font-weight: bold; color: {colors['text']}; margin: 0;">
                            📝 Expression: <span style="color: {colors['highlight']};">{expression}</span>
                        </p>
                        <p style="font-size: 18px; font-weight: bold; color: {colors['text']}; margin: 10px 0 0 0;">
                            ✅ Result: <span style="color: {colors['success']};">{result_str}</span>
                        </p>
                    </div>
                    <div style="text-align: center; margin: 15px 0;">
                        <a href="show_detailed_solution" style="
                            background: {colors['highlight']}; 
                            color: white; 
                            padding: 12px 24px; 
                            border-radius: 8px; 
                            text-decoration: none; 
                            font-weight: bold;
                            display: inline-block;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                        ">
                            🔍 View Solution Details
                        </a>
                    </div>
                    <p style="color: {colors['text']}; font-size: 12px; text-align: center; margin: 10px 0;">
                        💡 Click to see mathematical analysis and computation details
                    </p>
                </div>
                """)
            
            if status_bar:
                status_bar.showMessage("Calculation completed - Click result for detailed solution")
                
        # Optionally speak the result if voice is available and result is successful
        if VOICE_AVAILABLE and self.voice_controller and not error:
            if isinstance(result, tuple):
                result_for_speech = str(result[0]) if result else None
            else:
                result_for_speech = str(result)
            
            # Only speak for short results to avoid long speech
            if result_for_speech and len(result_for_speech) < 50:
                self.speak_result(result_for_speech)
                
        # Scroll to the bottom to show the latest result
        self.results_display.moveCursor(QTextCursor.End)
        if hasattr(self.steps_display, 'moveCursor'):
            self.steps_display.moveCursor(QTextCursor.End)
    
    def _format_result_for_display(self, result):
        """Format the result for display based on its type."""
        if result is None:
            return "No result"
        
        # For sympy Matrix objects, format them nicely
        if hasattr(result, "__class__") and result.__class__.__name__ == "Matrix":
            formatted = str(result).replace("Matrix", "")
            return formatted
        
        # For lists, check if they contain matrices or other special types
        if isinstance(result, list):
            if all(hasattr(item, "__class__") and item.__class__.__name__ == "Matrix" for item in result):
                return "\n\n".join([str(matrix).replace("Matrix", "") for matrix in result])
            else:
                return str(result)
        
        return str(result)
    
    def clear_input(self):
        """Clear the input field."""
        self.input_field.clear()
        status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
        if status_bar:
            status_bar.showMessage("Input cleared")
    
    def clear_all(self):
        """Clear all fields."""
        self.input_field.clear()
        self.results_display.clear()
        self.steps_display.clear()
        self.nl_input.clear()
        status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
        if status_bar:
            status_bar.showMessage("All fields cleared")
    
    def backspace(self):
        """Delete the last character from the input field."""
        current_text = self.input_field.text()
        self.input_field.setText(current_text[:-1])
    
    def toggle_sign(self):
        """Toggle the sign of the number in the input field."""
        current_text = self.input_field.text()
        if current_text.startswith("-"):
            self.input_field.setText(current_text[1:])
        else:
            self.input_field.setText("-" + current_text)
    
    def append_to_input(self, text):
        """Append text to the input field."""
        current_text = self.input_field.text()
        cursor_pos = self.input_field.cursorPosition()
        
        # Insert at cursor position
        new_text = current_text[:cursor_pos] + text + current_text[cursor_pos:]
        self.input_field.setText(new_text)
        
        # Move cursor after inserted text
        self.input_field.setCursorPosition(cursor_pos + len(text))
    
    def append_function(self, function_name):
        """Append a function to the input field."""
        # Map function names to their actual representation
        function_map = {
            'sin': 'sin(',
            'cos': 'cos(',
            'tan': 'tan(',
            'asin': 'asin(',
            'acos': 'acos(',
            'atan': 'atan(',
            'sinh': 'sinh(',
            'cosh': 'cosh(',
            'tanh': 'tanh(',
            'log': 'log(',
            'ln': 'ln('
        }
        
        # Get the actual function string
        function_str = function_map.get(function_name, function_name + '(')
        
        # Append to input
        self.append_to_input(function_str)
    
    def reciprocal(self):
        """Calculate the reciprocal of the current value."""
        current_text = self.input_field.text()
        if current_text:
            self.input_field.setText(f"1/({current_text})")
        else:
            self.input_field.setText("1/")
    
    def memory_clear(self):
        """Clear the memory."""
        self.calculator.memory = 0
        status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
        if status_bar:
            status_bar.showMessage("Memory cleared")
    
    def memory_recall(self):
        """Recall the value from memory."""
        self.input_field.setText(str(getattr(self.calculator, 'memory', 0)))
        status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
        if status_bar:
            status_bar.showMessage("Memory recalled")
    
    def memory_add(self):
        """Add the current value to memory."""
        try:
            current_value = float(self.input_field.text())
            if hasattr(self.calculator, 'memory'):
                self.calculator.memory = int(self.calculator.memory) + int(current_value)
            else:
                self.calculator.memory = int(current_value)
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage(f"Added to memory: {current_value}")
        except ValueError:
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage("Invalid value for memory operation")
    
    def memory_subtract(self):
        """Subtract the current value from memory."""
        try:
            current_value = float(self.input_field.text())
            if hasattr(self.calculator, 'memory'):
                self.calculator.memory = int(self.calculator.memory) - int(current_value)
            else:
                self.calculator.memory = -int(current_value)
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage(f"Subtracted from memory: {current_value}")
        except ValueError:
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage("Invalid value for memory operation")
    
    def process_nl_query(self):
        """Process the natural language query with Advanced NLU."""
        query = self.nl_input.toPlainText().strip()
        if not query:
            return
        
        # Clear previous results
        self.results_display.clear()
        self.steps_display.clear()
        
        # Display the query in the results area
        colors = DARK_COLORS if self.is_dark_mode else LIGHT_COLORS
        self.results_display.setTextColor(QColor(colors["highlight"]))
        self.results_display.append(f"🧠 AI Query: {query}")
        self.results_display.setTextColor(QColor(colors["text"]))
        
        try:
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage("🤖 Processing with AI NLU...")
            
            # Use Advanced NLU if available
            if self.advanced_nlu:
                self._process_with_advanced_nlu(query)
            else:
                # Fallback to original processing
                nl_thread = NLProcessThread(self.calculator, query)
                nl_thread.result_ready.connect(self.display_result)
                self.active_threads.append(nl_thread)
                nl_thread.finished.connect(lambda: self.thread_finished(nl_thread))
                nl_thread.start()
                
        except Exception as e:
            self.display_result(None, str(e))
    
    def _process_with_advanced_nlu(self, query: str):
        """Process query using Advanced NLU system"""
        if not self.advanced_nlu:
            self.results_display.append("❌ Advanced NLU not available")
            return
            
        try:
            # Understand the query with AI
            understanding = self.advanced_nlu.understand_query(query)
            
            # Generate SymPy code
            sympy_code = self.advanced_nlu.generate_sympy_code(understanding)
            
            # Display understanding
            self.results_display.append(f"🎯 Detected Operation: {understanding['operation']}")
            self.results_display.append(f"📝 Expression: {understanding.get('expression', 'N/A')}")
            self.results_display.append(f"🔢 Variable: {understanding.get('variable', 'N/A')}")
            self.results_display.append(f"🎲 Confidence: {understanding['confidence']:.1%}")
            self.results_display.append(f"⚡ Generated Code: {sympy_code}")
            self.results_display.append("")
            
            # Execute the mathematical operation
            result = self._execute_nlu_operation(understanding, sympy_code)
            
            # Display result
            if result:
                self.results_display.append(f"✅ Result: {result}")
                # Add to history
                self.add_to_history(query, str(result))
            else:
                self.results_display.append("❌ Could not compute result")
            
            status_bar = self.statusBar()
            if status_bar:
                status_bar.showMessage("🎉 AI NLU processing completed!")
                
        except Exception as e:
            self.results_display.append(f"❌ Error in NLU processing: {str(e)}")
            status_bar = self.statusBar()
            if status_bar:
                status_bar.showMessage("❌ NLU processing failed")
    
    def _execute_nlu_operation(self, understanding: Dict, sympy_code: str) -> str:
        """Execute the mathematical operation based on NLU understanding"""
        try:
            operation = understanding.get('operation', 'unknown')
            expression = understanding.get('expression', '')
            variable = understanding.get('variable', 'x')
            
            if not expression:
                return "No expression to compute"
            
            # Execute using SymPy
            result = None
            steps = []
            
            if operation == 'derivative':
                expr = sp.sympify(expression)
                result = sp.diff(expr, variable)
                steps.append(f"🧮 Differentiating {expression} with respect to {variable}")
                steps.append(f"📊 d/d{variable}({expression}) = {result}")
            
            elif operation == 'integral':
                expr = sp.sympify(expression)
                bounds = understanding.get('bounds')
                if bounds:
                    result = sp.integrate(expr, (variable, bounds[0], bounds[1]))
                    steps.append(f"🧮 Integrating {expression} from {bounds[0]} to {bounds[1]}")
                else:
                    result = sp.integrate(expr, variable)
                    steps.append(f"🧮 Integrating {expression} with respect to {variable}")
                steps.append(f"📊 ∫({expression}) d{variable} = {result}")
            
            elif operation == 'solve':
                if '=' in expression:
                    # It's an equation
                    left, right = expression.split('=', 1)
                    eq = sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip()))
                    result = sp.solve(eq, variable)
                else:
                    # Assume equation = 0
                    result = sp.solve(sp.sympify(expression), variable)
                steps.append(f"🧮 Solving for {variable}")
                steps.append(f"📊 Solution(s): {result}")
            
            elif operation == 'limit':
                expr = sp.sympify(expression)
                limit_point = understanding.get('limit_point', '0')
                result = sp.limit(expr, variable, limit_point)
                steps.append(f"🧮 Finding limit of {expression} as {variable} → {limit_point}")
                steps.append(f"📊 lim({expression}) = {result}")
            
            elif operation in ['simplify', 'expand', 'factor']:
                expr = sp.sympify(expression)
                if operation == 'simplify':
                    result = sp.simplify(expr)
                elif operation == 'expand':
                    result = sp.expand(expr)
                elif operation == 'factor':
                    result = sp.factor(expr)
                steps.append(f"🧮 {operation.title()}ing {expression}")
                steps.append(f"📊 Result: {result}")
            
            elif operation in ['evaluate', 'calculate']:
                expr = sp.sympify(expression)
                result = sp.simplify(expr)
                numerical_result = float(result) if result.is_number else result
                steps.append(f"🧮 Calculating {expression}")
                steps.append(f"📊 Result: {numerical_result}")
                result = numerical_result
            
            # Display steps
            if steps:
                self.steps_display.setText('\n'.join(steps))
            
            return str(result) if result is not None else "No result"
            
        except Exception as e:
            return f"Computation error: {str(e)}"
    
    def show_calculus_dialog(self, operation_type):
        """Show dialog for calculus operations."""
        dialog = CalculusDialog(self, operation_type)
        if dialog.exec_():
            pass  # Dialog handles the calculation and result display
    
    def on_ocr_finished(self, text):
        """Handle the OCR text extraction result."""
        # Clear previous results
        self.results_display.clear()
        self.steps_display.clear()
        
        # Check if the OCR process encountered an error
        if text is None or text.startswith("Error"):
            self.results_display.setText(str(text))
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage("OCR error occurred")
            return
        
        # Show the OCR result and let the user edit it
        edited, ok = self.show_ocr_edit_dialog(text)
        
        if not ok or not edited.strip():
            # User cancelled or provided empty text
            self.results_display.append(f"OCR: {text}")
            self.results_display.append("Processing cancelled by user.")
            return
        
        # Update the input field with the edited expression
        self.input_field.setText(edited)
        self.results_display.append(f"OCR (edited): {edited}")
        
        try:
            # Determine if this is an equation (contains =) or an expression
            if '=' in edited:
                # It's an equation, use the solver directly
                result, steps = self.solver.solve_equation(edited)
                
                # Display the result
                if result:
                    self.results_display.append(f"Solutions: {result}")
                else:
                    self.results_display.append("No solutions found.")
            else:
                # It's an expression, use the calculator's process_text_command
                result = self.calculator.process_text_command(edited)
                
                if isinstance(result, tuple) and len(result) == 2:
                    value, steps = result
                    # Display the computed result
                    self.results_display.append(f"Result: {value}")
                    # Display steps if available
                    if steps:
                        self.steps_display.setText(str(steps))
                else:
                    # If result is not a tuple with steps, just display the result
                    self.results_display.append(f"Result: {result}")
            
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage("OCR text processed successfully")
                
        except Exception as e:
            self.results_display.append(f"Error processing: {str(e)}")
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage("Error processing OCR result")

    def show_ocr_dialog(self):
        """Show the dialog for OCR image upload with ML enhancement options."""
        if not OCR_AVAILABLE:
            QMessageBox.warning(self, "OCR Not Available", 
                               "OCR functionality requires EasyOCR and PyTesseract libraries.\n\n"
                               "To install them, run:\n"
                               "pip install easyocr pytesseract Pillow\n\n"
                               "For Tesseract, also install the Tesseract OCR engine from:\n"
                               "https://github.com/tesseract-ocr/tesseract")
            return
        
        # Show enhancement level selection dialog
        enhancement_level = self.show_enhancement_dialog()
        if enhancement_level is None:
            return  # User cancelled
        
        try:
            options = QFileDialog.Options()
            if self.is_dark_mode:
                # Force non-native dialog in dark mode for better styling
                options |= QFileDialog.DontUseNativeDialog
                
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Select Image with Math Expression", "", 
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", 
                options=options
            )
            
            if file_name:
                # Show a processing message
                status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
                if status_bar:
                    status_bar.showMessage(f"Processing image with {enhancement_level} enhancement... This might take a moment.")
                
                # Process the image to get OCR text with ML enhancement
                try:
                    ocr_text, cleaned, simplified, steps = solve_from_image(file_name, enhancement_level)
                    # Pass the OCR result to the handler
                    self.on_ocr_finished(ocr_text)
                    if steps:
                        self.steps_display.setText(str(steps))
                except Exception as e:
                    QMessageBox.warning(self, "OCR Error", f"Error processing image: {str(e)}")
                    if status_bar:
                        status_bar.showMessage("OCR processing failed")
        except Exception as e:
            QMessageBox.warning(self, "OCR Error", f"Error opening image: {str(e)}")
    
    def show_enhancement_dialog(self):
        """Show dialog to select image enhancement level"""
        dialog = QDialog(self)
        dialog.setWindowTitle("🔬 ML Image Enhancement Level")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("🧠 Choose ML Enhancement Level")
        title.setFont(QFont(FONT_FAMILY, FONT_SIZE + 2, QFont.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Select the level of AI-powered image enhancement for better OCR accuracy:")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Enhancement options
        enhancement_group = QGroupBox("Enhancement Options")
        enhancement_layout = QVBoxLayout(enhancement_group)
        
        self.enhancement_radio_light = QRadioButton("🔹 Light Enhancement")
        self.enhancement_radio_medium = QRadioButton("🔸 Medium Enhancement (Recommended)")
        self.enhancement_radio_aggressive = QRadioButton("🔺 Aggressive Enhancement")
        
        # Set medium as default
        self.enhancement_radio_medium.setChecked(True)
        
        # Add descriptions
        light_desc = QLabel("   • Basic cleanup and noise reduction\n   • Fast processing\n   • Good for clear images")
        light_desc.setStyleSheet("color: gray; margin-left: 20px;")
        
        medium_desc = QLabel("   • ML denoising + background removal\n   • Balanced speed and quality\n   • Best for most mathematical images")
        medium_desc.setStyleSheet("color: gray; margin-left: 20px;")
        
        aggressive_desc = QLabel("   • Full ML pipeline with super-resolution\n   • Slower but highest quality\n   • Best for poor quality or small images")
        aggressive_desc.setStyleSheet("color: gray; margin-left: 20px;")
        
        enhancement_layout.addWidget(self.enhancement_radio_light)
        enhancement_layout.addWidget(light_desc)
        enhancement_layout.addWidget(self.enhancement_radio_medium)
        enhancement_layout.addWidget(medium_desc)
        enhancement_layout.addWidget(self.enhancement_radio_aggressive)
        enhancement_layout.addWidget(aggressive_desc)
        
        layout.addWidget(enhancement_group)
        
        # ML availability info
        try:
            from advanced_image_preprocessing import ADVANCED_PREPROCESSING_AVAILABLE
            if ADVANCED_PREPROCESSING_AVAILABLE:
                ml_status = QLabel("✅ ML preprocessing models available")
                ml_status.setStyleSheet("color: green; font-weight: bold;")
            else:
                ml_status = QLabel("⚠️ ML models not available - will use traditional preprocessing")
                ml_status.setStyleSheet("color: orange; font-weight: bold;")
        except ImportError:
            ml_status = QLabel("⚠️ ML models not available - will use traditional preprocessing")
            ml_status.setStyleSheet("color: orange; font-weight: bold;")
        
        layout.addWidget(ml_status)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Process Image")
        cancel_button = QPushButton("Cancel")
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)
        
        # Connect buttons
        result = {'enhancement': None}
        
        def accept():
            if self.enhancement_radio_light.isChecked():
                result['enhancement'] = 'light'
            elif self.enhancement_radio_medium.isChecked():
                result['enhancement'] = 'medium'
            elif self.enhancement_radio_aggressive.isChecked():
                result['enhancement'] = 'aggressive'
            dialog.accept()
        
        def reject():
            result['enhancement'] = None
            dialog.reject()
        
        ok_button.clicked.connect(accept)
        cancel_button.clicked.connect(reject)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            return result['enhancement']
        else:
            return None
    
    def show_ocr_edit_dialog(self, ocr_text):
        """Show a dialog to let the user edit the OCR result before parsing, with a helpful hint."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit OCR Result")
        layout = QVBoxLayout(dialog)
        label = QLabel("Please review and edit the recognized text below to a valid math expression.\n\nFor example, you can write:\n((4 + 3*x) + (5 - 4*x + 2*x**2)) - ((3*x**2 - 5*x) + (-x**2 + 2*x + 5))\n\nUse * for multiplication and ** for powers.")
        label.setWordWrap(True)
        layout.addWidget(label)
        text_edit = QTextEdit()
        text_edit.setText(ocr_text)
        layout.addWidget(text_edit)
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)
        result = {'ok': False}
        def accept():
            result['ok'] = True
            dialog.accept()
        def reject():
            result['ok'] = False
            dialog.reject()
        ok_button.clicked.connect(accept)
        cancel_button.clicked.connect(reject)
        dialog.exec_()
        return text_edit.toPlainText(), result['ok']
    
    def toggle_voice_recognition(self):
        """Toggle voice recognition on/off - ChatGPT-like interaction"""
        if not VOICE_AVAILABLE or not self.voice_controller:
            QMessageBox.information(self, "Voice Recognition Setup", 
                              "🎤 Voice recognition is ready to use!\n\n"
                              "Dependencies installed:\n"
                              "✅ SpeechRecognition\n"
                              "✅ PyAudio\n"
                              "✅ pyttsx3\n\n"
                              "Your microphone should now work with the calculator.")
            return
        
        if self.voice_button.isChecked():
            # Start voice recognition with ChatGPT-like experience
            success = self.voice_controller.start_voice_recognition()
            if success:
                self.voice_button.setText("🔴 Listening...")
                self.voice_button.setStyleSheet("""
                    background-color: #ff4444; 
                    color: white; 
                    font-weight: bold;
                    border-radius: 8px;
                    padding: 6px;
                """)
                status_bar = self.statusBar()
                if status_bar:
                    status_bar.showMessage("🎤 Listening for your mathematical question - speak naturally!")
                
                # Provide voice feedback
                self.voice_controller.speak_result("Ready to listen. What's your mathematical question?")
            else:
                self.voice_button.setChecked(False)
                QMessageBox.warning(self, "Microphone Access", 
                                  "🎤 Could not access microphone.\n\n"
                                  "Please check:\n"
                                  "• Microphone is connected and working\n"
                                  "• Python has microphone permissions\n"
                                  "• No other apps are using the microphone")
        else:
            # Stop voice recognition
            self.voice_controller.stop_voice_recognition()
            self.voice_button.setText("🎤 Voice")
            self.voice_button.setStyleSheet("")  # Reset style
            status_bar = self.statusBar()
            if status_bar:
                status_bar.showMessage("Voice recognition stopped")
            
            # Provide voice feedback
            self.voice_controller.speak_result("Voice recognition stopped")
    
    def process_voice_input(self, recognized_text):
        """Process recognized voice input"""
        if not recognized_text:
            return
        
        # Update UI to show voice input was processed
        self.voice_button.setChecked(False)
        self.voice_button.setText("🎤 Voice")
        self.voice_button.setStyleSheet("")
        
        # Set the recognized text in the appropriate input field
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 2:  # Natural Language tab
            if hasattr(self, 'nl_input'):
                self.nl_input.setPlainText(recognized_text)
                status_bar = self.statusBar()
                if status_bar:
                    status_bar.showMessage(f"🎤 Voice input: {recognized_text}")
                # Automatically process the query
                QTimer.singleShot(500, self.process_nl_query)  # Small delay for UI update
        else:
            # For other tabs, use main input field
            self.input_field.setText(recognized_text)
            status_bar = self.statusBar()
            if status_bar:
                status_bar.showMessage(f"🎤 Voice input: {recognized_text}")
            # Automatically evaluate if it looks like a complete expression
            if any(op in recognized_text for op in ['=', '+', '-', '*', '/', 'solve', 'calculate']):
                QTimer.singleShot(500, self.evaluate_expression)
    
    def speak_result(self, result_text):
        """Speak the calculation result using text-to-speech"""
        if VOICE_AVAILABLE and self.voice_controller:
            # Clean the result for speaking
            speech_text = self._prepare_text_for_speech(result_text)
            self.voice_controller.speak_result(speech_text)
    
    def _prepare_text_for_speech(self, text):
        """Prepare text for speech synthesis by cleaning mathematical notation"""
        if not text:
            return "No result"
        
        # Remove excessive formatting and convert to spoken form
        speech_text = str(text)
        
        # Replace mathematical symbols with spoken equivalents
        replacements = {
            '+': ' plus ',
            '-': ' minus ',
            '*': ' times ',
            '/': ' divided by ',
            '=': ' equals ',
            '**': ' to the power of ',
            'sqrt': ' square root of ',
            'sin': ' sine of ',
            'cos': ' cosine of ',
            'tan': ' tangent of ',
            'pi': ' pi ',
            'e': ' e ',
            '(': ' open parenthesis ',
            ')': ' close parenthesis ',
        }
        
        for symbol, spoken in replacements.items():
            speech_text = speech_text.replace(symbol, spoken)
        
        # Clean up multiple spaces
        speech_text = ' '.join(speech_text.split())
        
        # Limit length for speech
        if len(speech_text) > 200:
            speech_text = speech_text[:200] + "... and more"
        
        return f"The result is: {speech_text}"
    
    def show_voice_help(self):
        """Show voice recognition help dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("🎤 Voice Recognition Help - SAUNet 4.0")
        dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("🎤 Voice Recognition Guide")
        title.setFont(QFont(FONT_FAMILY, FONT_SIZE + 4, QFont.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Instructions
        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setHtml("""
        <h3>🗣️ How to Use Voice Recognition:</h3>
        <ol>
        <li><b>Click the 🎤 Voice button</b> to start listening</li>
        <li><b>Wait for "Ready to listen"</b> voice confirmation</li>
        <li><b>Speak clearly</b> into your microphone</li>
        <li><b>Wait for processing</b> - results will appear automatically</li>
        </ol>
        
        <h3>📝 Example Voice Commands:</h3>
        <ul>
        <li><b>"Solve two x plus three equals seven"</b></li>
        <li><b>"What is the square root of sixteen"</b></li>
        <li><b>"Calculate five times seven plus two"</b></li>
        <li><b>"Differentiate x squared with respect to x"</b></li>
        <li><b>"Integrate x cubed from zero to five"</b></li>
        <li><b>"Find the limit of sin x over x as x approaches zero"</b></li>
        <li><b>"Simplify two x plus three x minus five"</b></li>
        </ul>
        
        <h3>💡 Voice Tips:</h3>
        <ul>
        <li><b>Speak naturally</b> - no need to spell out symbols</li>
        <li><b>Use words</b>: "plus" for +, "minus" for -, "times" for ×</li>
        <li><b>Say "squared"</b> for ², "cubed" for ³, "to the power of" for ^</li>
        <li><b>Use "x"</b> for variables, "pi" for π, "e" for Euler's number</li>
        <li><b>Be clear</b> about equations: "two x plus three equals seven"</li>
        </ul>
        
        <h3>🔧 Troubleshooting:</h3>
        <ul>
        <li><b>Not hearing you?</b> Check microphone permissions and volume</li>
        <li><b>Wrong recognition?</b> Try speaking more slowly and clearly</li>
        <li><b>No response?</b> Ensure no other apps are using the microphone</li>
        <li><b>Background noise?</b> Try using the voice feature in a quieter environment</li>
        </ul>
        
        <h3>🎯 Features:</h3>
        <ul>
        <li><b>Multi-engine recognition</b> for better accuracy</li>
        <li><b>Automatic mathematical processing</b> with AI understanding</li>
        <li><b>Voice feedback</b> confirms what was heard</li>
        <li><b>Smart tab switching</b> to appropriate calculator mode</li>
        </ul>
        """)
        layout.addWidget(instructions)
        
        # Close button
        close_button = QPushButton("Got it! 👍")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.exec_()
    
    def solve_step_by_step(self, force_detailed=False):
        """Solve mathematical problem with intelligent step-by-step breakdown."""
        if not INTELLIGENT_SOLVER_AVAILABLE:
            self.step_solution_display.setHtml("""
                <div style='color: #e74c3c; padding: 20px;'>
                <h3>❌ Intelligent Solver Not Available</h3>
                <p>The intelligent math solver module is not installed.</p>
                <p>Install with: <code>pip install sympy</code></p>
                <p>This is a preview of what the interface would look like.</p>
                </div>
            """)
            self.step_status_label.setText("❌ Intelligent solver module not available")
            return
        
        problem = self.step_problem_input.text().strip()
        if not problem:
            self.step_status_label.setText("⚠️ Please enter a mathematical problem")
            return
        
        self.step_status_label.setText("🔄 Analyzing problem with AI...")
        QApplication.processEvents()
        
        try:
            result = self.intelligent_solver.solve_step_by_step(problem, force_detailed)
            self.display_step_solution(result)
            
            # Add to calculation history
            if result.get('success'):
                if result.get('is_simple'):
                    answer = str(result['direct_answer'])
                elif result.get('is_repetitive'):
                    answer = result['quick_answer']
                else:
                    answer = result['solution']['final_answer']
                
                self.add_to_history(problem, answer)
        
        except Exception as e:
            self.step_solution_display.setHtml(f"""
                <div style='color: #e74c3c; padding: 20px;'>
                <h3>❌ Error</h3>
                <p>Failed to solve: {str(e)}</p>
                <p>Please check your problem format and try again.</p>
                </div>
            """)
            self.step_status_label.setText("❌ Error occurred during solving")
    
    def display_step_solution(self, result):
        """Display the intelligent step-by-step solution."""
        if not result.get('success'):
            self.step_solution_display.setHtml("""
                <div style='color: #e74c3c; padding: 20px;'>
                <h3>❌ Unable to Solve</h3>
                <p>Please check your problem format and try again.</p>
                <p>Try examples like: 'solve x + 5 = 10' or 'what is 25% of 80'</p>
                </div>
            """)
            self.step_status_label.setText("❌ Unable to solve the problem")
            return
        
        html = "<div style='font-family: Arial; font-size: 12px; line-height: 1.4;'>"
        
        # Handle different response types
        if result.get('is_simple'):
            html += f"""
                <div style='background: #d4edda; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #28a745;'>
                <h3 style='color: #155724; margin: 0 0 10px 0;'>🎯 Quick Answer</h3>
                <p style='font-size: 18px; font-weight: bold; margin: 10px 0; color: #155724;'>{result['direct_answer']}</p>
                <p style='margin: 0; color: #155724;'>{result['offer_steps']}</p>
                </div>
            """
            self.step_status_label.setText("✅ Quick calculation completed!")
        
        elif result.get('is_repetitive'):
            html += f"""
                <div style='background: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #ffc107;'>
                <h3 style='color: #856404; margin: 0 0 10px 0;'>🔄 Smart Memory Response</h3>
                <p style='margin: 10px 0; color: #856404;'>{result['message']}</p>
                <div style='background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0;'>
                <p style='font-weight: bold; margin: 0; color: #495057;'>Quick Answer: {result['quick_answer']}</p>
                </div>
                </div>
            """
            self.step_status_label.setText("🔄 Found in memory - providing smart response")
        
        else:
            # Detailed solution
            solution = result['solution']
            
            # Problem analysis header
            html += f"""
                <div style='background: #cce5ff; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #007bff;'>
                <h3 style='color: #004085; margin: 0 0 10px 0;'>📋 Problem Analysis</h3>
                <table style='width: 100%; color: #004085;'>
                <tr><td><strong>Problem:</strong></td><td>{solution['problem']}</td></tr>
                <tr><td><strong>Domain:</strong></td><td>{solution['domain'].title()}</td></tr>
                <tr><td><strong>Difficulty:</strong></td><td>{solution['difficulty']}</td></tr>
                <tr><td><strong>Key Concepts:</strong></td><td>{', '.join(solution['key_concepts'])}</td></tr>
                </table>
                </div>
            """
            
            # Step-by-step solution
            html += """
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #6c757d;'>
                <h3 style='color: #495057; margin: 0 0 15px 0;'>🔧 Step-by-Step Solution</h3>
            """
            
            steps = solution['steps']
            step_count = 0
            for step_text in steps:
                if '🎯' in step_text or '🧮' in step_text or '📊' in step_text:
                    # Header step
                    html += f"""
                        <div style='background: #e9ecef; padding: 8px; border-radius: 4px; margin: 8px 0;'>
                        <h4 style='margin: 0; color: #495057;'>{step_text}</h4>
                        </div>
                    """
                elif step_text.startswith('===') or step_text == '':
                    # Skip separators
                    continue
                elif any(emoji in step_text for emoji in ['📝', '🔧', '⚡', '✅']):
                    # Main steps
                    step_count += 1
                    html += f"""
                        <div style='background: white; padding: 10px; border-left: 3px solid #007bff; margin: 8px 0; border-radius: 4px;'>
                        <p style='margin: 0; color: #212529; font-weight: 500;'>{step_text}</p>
                        </div>
                    """
                elif step_text.strip().startswith('→') or step_text.strip().startswith('•'):
                    # Sub-steps
                    html += f"""
                        <div style='background: #f1f3f4; padding: 8px; margin: 4px 0 4px 20px; border-radius: 4px; border-left: 2px solid #28a745;'>
                        <p style='margin: 0; color: #495057; font-style: italic;'>{step_text.strip()}</p>
                        </div>
                    """
                else:
                    # Other text
                    html += f"""
                        <div style='padding: 4px 0; margin: 4px 0;'>
                        <p style='margin: 0; color: #6c757d;'>{step_text}</p>
                        </div>
                    """
            
            html += "</div>"
            
            # Final answer
            html += f"""
                <div style='background: #d4edda; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #28a745;'>
                <h3 style='color: #155724; margin: 0 0 10px 0;'>🎉 Final Answer</h3>
                <p style='font-size: 18px; font-weight: bold; margin: 0; color: #155724;'>{solution['final_answer']}</p>
                </div>
            """
            
            self.step_status_label.setText(f"✅ Problem solved with detailed steps!")
        
        html += "</div>"
        self.step_solution_display.setHtml(html)
    
    def load_step_example(self, example):
        """Load an example problem into the step solver."""
        self.step_problem_input.setText(example)
        self.solve_step_by_step()
    
    def clear_step_solution(self):
        """Clear the step-by-step solution display."""
        self.step_solution_display.clear()
        self.step_problem_input.clear()
        self.step_status_label.setText("Ready to solve mathematical problems with detailed explanations!")

    
    def perform_matrix_operation(self):
        """Perform the selected matrix operation."""
        matrix_a_text = self.matrix_a_input.toPlainText().strip()
        matrix_b_text = self.matrix_b_input.toPlainText().strip()
        operation = self.matrix_op_combo.currentText().lower()
        if not matrix_a_text:
            QMessageBox.warning(self, "Input Error", "Matrix A is required")
            return
        # Check if matrix B is needed
        if operation in ["multiply", "add", "subtract"] and not matrix_b_text:
            QMessageBox.warning(self, "Input Error", "Matrix B is required for this operation")
            return
        try:
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage("Calculating...")
            # Map operation name to operation code
            op_map = {
                "determinant": "determinant",
                "inverse": "inverse",
                "transpose": "transpose",
                "multiply": "multiply",
                "add": "add",
                "subtract": "subtract",
                "eigenvalues": "eigenvalues"
            }
            # Start calculation in a separate thread
            matrix_thread = MatrixThread(self.calculator, matrix_a_text, matrix_b_text, op_map[operation.lower()])
            matrix_thread.result_ready.connect(self.display_result)
            self.active_threads.append(matrix_thread)  # Track the thread
            matrix_thread.finished.connect(lambda: self.thread_finished(matrix_thread))
            matrix_thread.start()
        except Exception as e:
            self.display_result(None, str(e))
    
    def perform_base_conversion(self):
        """Perform base conversion."""
        number = self.base_input.text().strip()
        if not number:
            QMessageBox.warning(self, "Input Error", "Please enter a number")
            return
        
        from_base_text = self.from_base_combo.currentText()
        to_base_text = self.to_base_combo.currentText()
        from_base_match = re.search(r'\d+', from_base_text)
        to_base_match = re.search(r'\d+', to_base_text)
        if not from_base_match or not to_base_match:
            QMessageBox.warning(self, "Input Error", "Invalid base selection")
            return
        from_base = int(from_base_match.group())
        to_base = int(to_base_match.group())
        try:
            result, steps = self.calculator.convert_base(number, from_base, to_base)
            self.display_result((result, steps))
        except Exception as e:
            self.display_result(None, str(e))
    
    def perform_stats_operation(self):
        """Perform statistical operation."""
        data_text = self.stats_input.toPlainText().strip()
        if not data_text:
            QMessageBox.warning(self, "Input Error", "Please enter data points")
            return
        
        operation = self.stats_op_combo.currentText().lower()
        
        try:
            # Map operation name to operation code
            op_map = {
                "mean": "mean",
                "median": "median",
                "mode": "mode",
                "standard deviation": "stdev",
                "variance": "variance"
            }
            
            # Perform statistical analysis
            result, steps = self.calculator.statistical_analysis(data_text, op_map[operation])
            self.display_result((result, steps))
        except Exception as e:
            self.display_result(None, str(e))
    
    def update_unit_combos(self, index):
        """Update the unit combo boxes based on the selected unit type."""
        unit_type = self.unit_type_combo.currentText().lower()
        
        # Clear current items
        self.from_unit_combo.clear()
        self.to_unit_combo.clear()
        
        # Add units based on the selected type
        if unit_type == "length":
            units = ["mm", "cm", "m", "km", "in", "ft", "yd", "mi"]
        elif unit_type == "mass":
            units = ["mg", "g", "kg", "oz", "lb", "ton"]
        elif unit_type == "volume":
            units = ["ml", "l", "m3", "gal", "qt"]
        elif unit_type == "time":
            units = ["s", "min", "hr", "day"]
        elif unit_type == "temperature":
            units = ["C", "F", "K"]
        elif unit_type == "area":
            units = ["mm2", "cm2", "m2", "km2", "in2", "ft2", "acre"]
        elif unit_type == "energy":
            units = ["J", "cal", "kcal", "kWh"]
        elif unit_type == "pressure":
            units = ["Pa", "kPa", "atm", "bar", "psi"]
        else:
            units = []
        
        # Add the units to the combo boxes
        self.from_unit_combo.addItems(units)
        self.to_unit_combo.addItems(units)
        
        # Set different default units
        if len(units) > 1:
            self.to_unit_combo.setCurrentIndex(1)
    
    def perform_unit_conversion(self):
        """Perform unit conversion."""
        value_text = self.unit_value_input.text().strip()
        if not value_text:
            QMessageBox.warning(self, "Input Error", "Please enter a value")
            return
        
        try:
            value = float(value_text)
            from_unit = self.from_unit_combo.currentText()
            to_unit = self.to_unit_combo.currentText()
            
            # Perform conversion
            result, steps = self.calculator.unit_conversion(value, from_unit, to_unit)
            self.display_result((result, steps))
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage("Unit conversion completed")
        except ValueError:
            self.display_result(None, "Invalid input value. Please enter a number.")
        except Exception as e:
            self.display_result(None, str(e))
            status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
            if status_bar:
                status_bar.showMessage("Unit conversion failed")
    
    def export_results(self):
        """Export results to a text file."""
        try:
            # Get the file name from a save dialog
            options = QFileDialog.Options()
            if self.is_dark_mode:
                # Force native dialog in dark mode for better compatibility
                options |= QFileDialog.DontUseNativeDialog
                
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "", 
                "Text Files (*.txt);;All Files (*)", 
                options=options
            )
            
            if file_name:
                if not file_name.lower().endswith('.txt'):
                    file_name += '.txt'
                    
                with open(file_name, 'w') as file:
                    results_text = self.results_display.toPlainText()
                    steps_text = self.steps_display.toPlainText()
                    
                    # Combine the results and steps
                    file.write("=== CALCULATION RESULTS ===\n\n")
                    file.write(f"{results_text}\n\n")
                    
                    if steps_text:
                        file.write("=== STEP-BY-STEP SOLUTION ===\n\n")
                        file.write(f"{steps_text}\n")
                
                status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
                if status_bar:
                    status_bar.showMessage(f"Results exported to {file_name}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def copy_result(self):
        """Copy the result to clipboard."""
        clipboard = QApplication.clipboard() if hasattr(QApplication, 'clipboard') else None
        if clipboard:
            clipboard.setText(self.results_display.toPlainText())
        status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
        if status_bar:
            status_bar.showMessage("Result copied to clipboard")
    
    def paste_expression(self):
        """Paste text from clipboard to input field."""
        clipboard = QApplication.clipboard() if hasattr(QApplication, 'clipboard') else None
        if clipboard:
            clipboard_text = clipboard.text()
            self.input_field.setText(clipboard_text)
        status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
        if status_bar:
            status_bar.showMessage("Text pasted from clipboard")
    
    def toggle_theme(self):
        """Toggle between light and dark mode."""
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()
        mode = "dark" if self.is_dark_mode else "light"
        status_bar = self.statusBar() if hasattr(self, 'statusBar') else None
        if status_bar:
            status_bar.showMessage(f"Switched to {mode} mode")
    
    def apply_theme(self):
        """Apply the current theme to the application."""
        colors = DARK_COLORS if self.is_dark_mode else LIGHT_COLORS
        palette = QPalette()
        
        # Set colors for all palette roles
        palette.setColor(QPalette.Window, QColor(colors["background"]))
        palette.setColor(QPalette.WindowText, QColor(colors["text"]))
        palette.setColor(QPalette.Base, QColor(colors["background"]))
        palette.setColor(QPalette.AlternateBase, QColor(colors["secondary"]))
        palette.setColor(QPalette.ToolTipBase, QColor(colors["background"]))
        palette.setColor(QPalette.ToolTipText, QColor(colors["text"]))
        palette.setColor(QPalette.Text, QColor(colors["text"]))
        palette.setColor(QPalette.Button, QColor(colors["button"]))
        palette.setColor(QPalette.ButtonText, QColor(colors["button_text"]))
        palette.setColor(QPalette.Link, QColor(colors["highlight"]))
        palette.setColor(QPalette.Highlight, QColor(colors["highlight"]))
        palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
        
        # Make sure disabled text is still readable in dark mode
        if self.is_dark_mode:
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#AAAAAA"))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#AAAAAA"))
            # Use darker backgrounds for disabled elements
            palette.setColor(QPalette.Disabled, QPalette.Button, QColor("#1A1A2A"))
            palette.setColor(QPalette.Disabled, QPalette.Base, QColor("#1A1A2A"))
            palette.setColor(QPalette.Disabled, QPalette.Window, QColor("#1A1A2A"))
        
        QApplication.setPalette(palette)
        
        # Set stylesheet for the entire application to handle widgets not covered by QPalette
        if self.is_dark_mode:
            # No need to set global stylesheet - we'll style individual widgets
            pass
        else:
            # No need to clear global stylesheet
            pass
        
        # Explicitly set text colors on the display widgets
        self.results_display.setTextColor(QColor(colors["text"]))
        self.steps_display.setTextColor(QColor(colors["text"]))
        
        mono_font = QFont("Roboto Mono", FONT_SIZE - 1) if "Roboto Mono" in QFontDatabase().families() else QFont("Courier New", FONT_SIZE - 1)
        self.steps_display.setFont(mono_font)
        self._update_ui_for_theme()
    
    def _update_ui_for_theme(self):
        """Update UI elements like buttons for the current theme."""
        # Update button styling
        colors = DARK_COLORS if self.is_dark_mode else LIGHT_COLORS
        
        # ---- Button styling ----
        button_style = f"""
            QPushButton {{
                background-color: {colors["button"]};
                color: {colors["button_text"]};
                border: 1px solid {colors["highlight"]};
                border-radius: 4px;
                padding: 4px;
            }}
            QPushButton:hover {{
                background-color: {colors["highlight"]};
                color: white;
            }}
        """
        
        # Apply to all buttons
        for child in self.findChildren(QPushButton):
            child.setStyleSheet(button_style)
        
        # ---- Text edit styling ----
        text_edit_style = f"""
            QTextEdit {{
                background-color: {colors["background"]};
                color: {colors["text"]};
                border: 1px solid {colors["secondary"]};
                border-radius: 4px;
                selection-background-color: {colors["highlight"]};
                selection-color: white;
            }}
        """
        
        self.results_display.setStyleSheet(text_edit_style)
        self.steps_display.setStyleSheet(text_edit_style)
        
        # ---- Line edit styling ----
        line_edit_style = f"""
            QLineEdit {{
                background-color: {colors["background"]};
                color: {colors["text"]};
                border: 1px solid {colors["secondary"]};
                border-radius: 4px;
                padding: 2px 4px;
                selection-background-color: {colors["highlight"]};
                selection-color: white;
            }}
        """
        
        self.input_field.setStyleSheet(line_edit_style)
        
        # Ensure NL input has proper styling too
        if hasattr(self, 'nl_input'):
            self.nl_input.setStyleSheet(line_edit_style)
        
        # ---- ComboBox styling ----
        combo_style = f"""
            QComboBox {{
                background-color: {colors["background"]};
                color: {colors["text"]};
                border: 1px solid {colors["secondary"]};
                border-radius: 4px;
                padding: 2px 4px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left: 1px solid {colors["secondary"]};
            }}
            QComboBox::down-arrow {{
                width: 8px;
                height: 8px;
                background-color: {colors["text"]};
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors["background"]};
                color: {colors["text"]};
                selection-background-color: {colors["highlight"]};
                selection-color: white;
                border: 1px solid {colors["secondary"]};
            }}
        """
        
        for combo in self.findChildren(QComboBox):
            combo.setStyleSheet(combo_style)
            
        # ---- Enhanced Tab Widget styling with modern animations ----
        tab_style = f"""
            QTabWidget::pane {{
                border: 2px solid {colors["secondary"]};
                border-radius: 12px;
                background-color: {colors["background"]};
                margin-top: -1px;
                padding: 4px;
            }}
            QTabBar::tab {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 {colors["button"]}, stop: 1 {colors["secondary"]});
                color: {colors["text"]};
                border: 2px solid {colors["secondary"]};
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 12px 20px;
                margin-right: 3px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
                max-width: 160px;
            }}
            QTabBar::tab:selected {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 {colors["highlight"]}, stop: 1 {colors["highlight"]});
                color: white;
                border-color: {colors["highlight"]};
                border-bottom: 2px solid {colors["background"]};
                margin-bottom: -2px;
                font-weight: 700;
            }}
            QTabBar::tab:hover:!selected {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 {colors["secondary"]}, stop: 1 {colors["button"]});
                border-color: {colors["highlight"]};
                color: {colors["highlight"]};
                font-weight: 600;
            }}
            QTabBar::tab:disabled {{
                background-color: {colors["button"]};
                color: #888888;
                border-color: #666666;
            }}
        """
        
        for tab_widget in self.findChildren(QTabWidget):
            tab_widget.setStyleSheet(tab_style)
            
        # ---- Label styling ----
        label_style = f"""
            QLabel {{
                color: {colors["text"]};
            }}
        """
        
        for label in self.findChildren(QLabel):
            label.setStyleSheet(label_style)
            
        # ---- GroupBox styling ----
        group_box_style = f"""
            QGroupBox {{
                border: 1px solid {colors["secondary"]};
                border-radius: 4px;
                margin-top: 10px;
                font-weight: bold;
                color: {colors["text"]};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                color: {colors["text"]};
            }}
        """
        
        for group_box in self.findChildren(QGroupBox):
            group_box.setStyleSheet(group_box_style)
            
        # ---- RadioButton and CheckBox styling ----
        radio_check_style = f"""
            QRadioButton, QCheckBox {{
                color: {colors["text"]};
            }}
            QRadioButton::indicator::unchecked {{
                border: 1px solid {colors["secondary"]};
                border-radius: 6px;
                background-color: {colors["background"]};
            }}
            QRadioButton::indicator::checked {{
                border: 1px solid {colors["highlight"]};
                border-radius: 6px;
                background-color: {colors["highlight"]};
                width: 10px;
                height: 10px;
            }}
            QCheckBox::indicator::unchecked {{
                border: 1px solid {colors["secondary"]};
                background-color: {colors["background"]};
            }}
            QCheckBox::indicator::checked {{
                border: 1px solid {colors["highlight"]};
                background-color: {colors["highlight"]};
            }}
        """
        
        for radio in self.findChildren(QRadioButton):
            radio.setStyleSheet(radio_check_style)
            
        for checkbox in self.findChildren(QCheckBox):
            checkbox.setStyleSheet(radio_check_style)
            
        # ---- Dialog styling ----
        # This will be applied to any dialogs created after this point
        dialog_style = f"""
            QDialog {{
                background-color: {colors["background"]};
                color: {colors["text"]};
            }}
            QMessageBox {{
                background-color: {colors["background"]};
                color: {colors["text"]};
            }}
        """
        # Apply the dialog style to all existing dialogs
        for dialog in self.findChildren(QDialog):
            dialog.setStyleSheet(dialog_style)
        
        # Style the status bar
        status_bar = self.statusBar()
        if status_bar:
            status_bar.setStyleSheet(f"""
                QStatusBar {{
                    background-color: {colors["background"]};
                    color: {colors["text"]};
                    border-top: 1px solid {colors["secondary"]};
                }}
            """)
            
        # Add styles for menus and other global elements
        menu_style = f"""
            QMenu {{
                background-color: {colors["background"]};
                color: {colors["text"]};
                border: 1px solid {colors["secondary"]};
            }}
            QMenu::item:selected {{
                background-color: {colors["highlight"]};
                color: white;
            }}
        """
        
        for menu in self.findChildren(QMenu):
            menu.setStyleSheet(menu_style)
            
        # Add styles for headers and tables
        header_style = f"""
            QHeaderView::section {{
                background-color: {colors["button"]};
                color: {colors["text"]};
                border: 1px solid {colors["secondary"]};
            }}
            QTableView {{
                background-color: {colors["background"]};
                color: {colors["text"]};
                gridline-color: {colors["secondary"]};
                selection-background-color: {colors["highlight"]};
                selection-color: white;
            }}
        """
        
        for header in self.findChildren(QHeaderView):
            header.setStyleSheet(header_style)
    
    def show_about(self):
        """Show the about dialog."""
        nlu_status = "✅ Advanced AI NLU with Transformers" if ADVANCED_NLU_AVAILABLE else "⚠️ Basic NLU"
        message = (
            "🚀 SAUNet 4.0 Advanced Scientific Calculator\n\n"
            "🔥 NEW in v4.0: AI-Powered Mathematical Understanding!\n\n"
            "🧠 AI Features:\n"
            f"{nlu_status}\n"
            "✅ ChatGPT-like natural language processing\n"
            "✅ Intelligent mathematical problem analysis\n"
            "✅ FLAN-T5 transformer model integration\n"
            "✅ spaCy linguistic analysis engine\n"
            "✅ Multi-modal AI reasoning (text + images)\n\n"
            "💪 Mathematical Capabilities:\n"
            "✅ Advanced calculus (derivatives, integrals, limits)\n"
            "✅ Equation solving with step-by-step solutions\n"
            "✅ Matrix operations & linear algebra\n"
            "✅ Statistical analysis & data processing\n"
            "✅ Unit conversions across multiple systems\n"
            "✅ AI-powered OCR for math from images\n"
            "✅ Symbolic mathematics & expression manipulation\n\n"
            "🎨 Enhanced UI:\n"
            "✅ Smooth tab transitions & auto-clear\n"
            "✅ Calculation history with smart navigation\n"
            "✅ Professional dark/light themes\n"
            "✅ Export results & copy functionality\n\n"
            "🛠️ AI Stack: Transformers, spaCy, PyTorch, FLAN-T5\n"
            "🔧 Math Stack: PyQt5, SymPy, NumPy, SciPy, EasyOCR\n"
            "⚡ Optimized for both CPU and GPU acceleration\n\n"
            "© 2024 SAUNet Technologies - Version 4.0.0"
        )
        QMessageBox.about(self, "About SAUNet 4.0", message)

    def closeEvent(self, event):
        """Handle the window close event to clean up resources."""
        # Wait for all threads to finish
        for thread in self.active_threads:
            if thread.isRunning():
                thread.terminate()
                thread.wait()
        
        # Accept the close event
        event.accept()

    def thread_finished(self, thread):
        """Remove thread from active threads list when finished."""
        if thread in self.active_threads:
            self.active_threads.remove(thread)


class CalculationThread(QThread):
    """Thread for performing calculations in the background."""
    result_ready = pyqtSignal(object, object)
    
    def __init__(self, calculator, expression):
        super().__init__()
        self.calculator = calculator
        self.expression = expression
        # Set the daemon property to avoid thread destruction issues
        self.setTerminationEnabled(True)
    
    def run(self):
        try:
            result, steps = self.calculator.evaluate(self.expression)
            self.result_ready.emit(result, steps)
        except Exception as e:
            self.result_ready.emit(None, str(e))


class NLProcessThread(QThread):
    """Thread for processing natural language queries."""
    result_ready = pyqtSignal(object, object)
    
    def __init__(self, calculator, query):
        super().__init__()
        self.calculator = calculator
        self.query = query
        # Set the daemon property to avoid thread destruction issues
        self.setTerminationEnabled(True)
    
    def run(self):
        try:
            result = self.calculator.process_text_command(self.query)
            self.result_ready.emit(result, None)
        except Exception as e:
            self.result_ready.emit(None, str(e))


class OCRThread(QThread):
    """Thread for OCR processing."""
    result_ready = pyqtSignal(str)
    
    def __init__(self, ocr, image_path):
        super().__init__()
        self.ocr = ocr
        self.image_path = image_path
        # Set the daemon property to avoid thread destruction issues
        self.setTerminationEnabled(True)
    
    def run(self):
        try:
            ocr_text, cleaned, simplified, steps = solve_from_image(self.image_path)
            self.result_ready.emit(steps)
        except Exception as e:
            self.result_ready.emit(f"Error: {str(e)}")


class MatrixThread(QThread):
    """Thread for matrix operations."""
    result_ready = pyqtSignal(object, object)
    
    def __init__(self, calculator, matrix_a, matrix_b, operation):
        super().__init__()
        self.calculator = calculator
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.operation = operation
        # Set the daemon property to avoid thread destruction issues
        self.setTerminationEnabled(True)
    
    def run(self):
        try:
            result, steps = self.calculator.matrix_operation(self.matrix_a, self.matrix_b, self.operation)
            self.result_ready.emit(result, steps)
        except Exception as e:
            self.result_ready.emit(None, str(e))


class CalculusDialog(QDialog):
    """Dialog for calculus operations."""
    
    def __init__(self, parent, operation_type):
        super().__init__(parent)
        self.parent = parent
        self.operation_type = operation_type
        self.solver = parent.solver
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        if self.operation_type == "derivative":
            self.setWindowTitle("Differentiate")
            title = "Find the derivative"
        elif self.operation_type == "integral":
            self.setWindowTitle("Integrate")
            title = "Find the integral"
        elif self.operation_type == "limit":
            self.setWindowTitle("Limit")
            title = "Find the limit"
        elif self.operation_type == "solve":
            self.setWindowTitle("Solve Equation")
            title = "Solve the equation"
        else:
            self.setWindowTitle("Calculus")
            title = "Operation"
        
        layout = QVBoxLayout(self)
        
        # Title
        layout.addWidget(QLabel(title))
        
        # Expression input
        layout.addWidget(QLabel("Expression:"))
        self.expression_input = QLineEdit()
        self.expression_input.setPlaceholderText("Enter mathematical expression...")
        layout.addWidget(self.expression_input)
        
        # Variable input
        layout.addWidget(QLabel("Variable:"))
        self.variable_input = QLineEdit()
        self.variable_input.setText("x")
        layout.addWidget(self.variable_input)
        
        # Additional inputs based on operation type
        if self.operation_type == "derivative":
            # Order input
            layout.addWidget(QLabel("Order:"))
            self.order_input = QLineEdit()
            self.order_input.setText("1")
            layout.addWidget(self.order_input)
        
        elif self.operation_type == "integral":
            # Definite integral checkbox
            self.definite_check = QCheckBox("Definite Integral")
            self.definite_check.stateChanged.connect(self.toggle_limits)
            layout.addWidget(self.definite_check)
            
            # Limits group (initially hidden)
            self.limits_group = QGroupBox("Integration Limits")
            self.limits_group.setVisible(False)
            limits_layout = QVBoxLayout(self.limits_group)
            
            # Lower limit
            limits_layout.addWidget(QLabel("Lower Limit:"))
            self.lower_input = QLineEdit()
            limits_layout.addWidget(self.lower_input)
            
            # Upper limit
            limits_layout.addWidget(QLabel("Upper Limit:"))
            self.upper_input = QLineEdit()
            limits_layout.addWidget(self.upper_input)
            
            layout.addWidget(self.limits_group)
        
        elif self.operation_type == "limit":
            # Point input
            layout.addWidget(QLabel("Point:"))
            self.point_input = QLineEdit()
            self.point_input.setText("0")
            layout.addWidget(self.point_input)
            
            # Direction radio buttons
            layout.addWidget(QLabel("Direction:"))
            
            direction_group = QGroupBox()
            direction_layout = QHBoxLayout(direction_group)
            
            self.both_radio = QRadioButton("Both")
            self.left_radio = QRadioButton("Left")
            self.right_radio = QRadioButton("Right")
            
            self.both_radio.setChecked(True)
            
            direction_layout.addWidget(self.both_radio)
            direction_layout.addWidget(self.left_radio)
            direction_layout.addWidget(self.right_radio)
            
            layout.addWidget(direction_group)
        
        # Buttons
        button_box = QHBoxLayout()
        
        calculate_button = QPushButton("Calculate")
        calculate_button.clicked.connect(self.calculate)
        button_box.addWidget(calculate_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(cancel_button)
        
        layout.addLayout(button_box)
    
    def toggle_limits(self, state):
        """Toggle the visibility of integration limits."""
        self.limits_group.setVisible(state == Qt.CheckState.Checked)
    
    def calculate(self):
        """Calculate and display the result."""
        expression = self.expression_input.text().strip()
        variable = self.variable_input.text().strip() or "x"
        
        if not expression:
            QMessageBox.warning(self, "Input Error", "Please enter an expression")
            return
        
        try:
            if self.operation_type == "derivative":
                order = int(self.order_input.text())
                # Use the solver's differentiate method directly
                result, steps = self.solver.differentiate(expression, variable, order)
            
            elif self.operation_type == "integral":
                if self.definite_check.isChecked():
                    lower = self.lower_input.text().strip()
                    upper = self.upper_input.text().strip()
                    if not lower or not upper:
                        QMessageBox.warning(self, "Input Error", "Please enter both limits")
                        return
                    
                    # Use the solver's integrate method directly with limits
                    result, steps = self.solver.integrate(expression, variable, lower, upper)
                else:
                    # Use the solver's integrate method directly without limits
                    result, steps = self.solver.integrate(expression, variable)
            
            elif self.operation_type == "limit":
                point = self.point_input.text().strip()
                
                if self.left_radio.isChecked():
                    direction = "-"
                elif self.right_radio.isChecked():
                    direction = "+"
                else:
                    direction = "+-"
                
                result, steps = self.solver.limit_step_by_step(expression, variable, point, direction)
            
            elif self.operation_type == "solve":
                result, steps = self.solver.solve_equation(expression, variable)
            
            # Display the result in the main window
            self.parent.display_result((result, steps))
            self.accept()
        
        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", str(e)) 


class SolutionPopupDialog(QDialog):
    """Enhanced popup dialog for displaying detailed solutions and step-by-step explanations."""
    
    def __init__(self, parent, expression, result, steps, calculation_type="General"):
        super().__init__(parent)
        self.expression = expression
        self.result = result
        self.steps = steps
        self.calculation_type = calculation_type
        self.init_ui()
    
    def init_ui(self):
        """Initialize the solution popup UI."""
        self.setWindowTitle("🔬 Detailed Solution - SAUNet 4.0")
        self.setMinimumSize(700, 600)
        self.resize(900, 700)
        
        # Apply theme colors
        colors = DARK_COLORS if hasattr(self.parent(), 'is_dark_mode') and self.parent().is_dark_mode else LIGHT_COLORS
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header section
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Box)
        header_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {colors['highlight']}, stop:1 {colors['secondary']});
                border: 2px solid {colors['highlight']};
                border-radius: 10px;
                padding: 15px;
            }}
        """)
        header_layout = QVBoxLayout(header_frame)
        
        # Title
        title = QLabel(f"🧮 {self.calculation_type} Solution")
        title.setFont(QFont(FONT_FAMILY, FONT_SIZE + 4, QFont.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white; font-weight: bold;")
        header_layout.addWidget(title)
        
        # Expression
        expr_label = QLabel(f"📝 Expression: {self.expression}")
        expr_label.setFont(QFont(FONT_FAMILY, FONT_SIZE + 1))
        expr_label.setWordWrap(True)
        expr_label.setStyleSheet("color: white; background: rgba(255,255,255,0.1); padding: 8px; border-radius: 5px;")
        header_layout.addWidget(expr_label)
        
        # Result
        result_label = QLabel(f"✅ Result: {self.result}")
        result_label.setFont(QFont(FONT_FAMILY, FONT_SIZE + 2, QFont.Bold))
        result_label.setWordWrap(True)
        result_label.setStyleSheet("color: white; background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;")
        header_layout.addWidget(result_label)
        
        main_layout.addWidget(header_frame)
        
        # Tabs for different views
        content_tabs = QTabWidget()
        
        # Step-by-step tab
        steps_tab = QWidget()
        steps_layout = QVBoxLayout(steps_tab)
        
        steps_label = QLabel("🔍 Step-by-Step Solution:")
        steps_label.setFont(QFont(FONT_FAMILY, FONT_SIZE + 1, QFont.Bold))
        steps_layout.addWidget(steps_label)
        
        # Enhanced scrollable text area for steps
        self.steps_display = QTextBrowser()
        self.steps_display.setFont(QFont("Consolas", FONT_SIZE))
        self.steps_display.setPlainText(self.steps if self.steps else "No detailed steps available.")
        self.steps_display.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {colors['background']};
                color: {colors['text']};
                border: 2px solid {colors['secondary']};
                border-radius: 8px;
                padding: 15px;
                line-height: 1.4;
            }}
            QScrollBar:vertical {{
                background: {colors['button']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {colors['highlight']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {colors['secondary']};
            }}
        """)
        steps_layout.addWidget(self.steps_display)
        
        content_tabs.addTab(steps_tab, "📖 Steps")
        
        # Formula/Method tab
        formula_tab = QWidget()
        formula_layout = QVBoxLayout(formula_tab)
        
        formula_label = QLabel("📐 Mathematical Method:")
        formula_label.setFont(QFont(FONT_FAMILY, FONT_SIZE + 1, QFont.Bold))
        formula_layout.addWidget(formula_label)
        
        formula_display = QTextBrowser()
        formula_display.setFont(QFont("Consolas", FONT_SIZE))
        formula_info = self._generate_formula_info()
        formula_display.setPlainText(formula_info)
        formula_display.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {colors['background']};
                color: {colors['text']};
                border: 2px solid {colors['secondary']};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        formula_layout.addWidget(formula_display)
        
        content_tabs.addTab(formula_tab, "🧪 Method")
        
        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        
        analysis_label = QLabel("🔬 Solution Analysis:")
        analysis_label.setFont(QFont(FONT_FAMILY, FONT_SIZE + 1, QFont.Bold))
        analysis_layout.addWidget(analysis_label)
        
        analysis_display = QTextBrowser()
        analysis_display.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        analysis_info = self._generate_analysis_info()
        analysis_display.setPlainText(analysis_info)
        analysis_display.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {colors['background']};
                color: {colors['text']};
                border: 2px solid {colors['secondary']};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        analysis_layout.addWidget(analysis_display)
        
        content_tabs.addTab(analysis_tab, "📊 Analysis")
        
        main_layout.addWidget(content_tabs)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Copy result button
        copy_btn = QPushButton("📋 Copy Result")
        copy_btn.clicked.connect(self.copy_result)
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['highlight']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {colors['secondary']};
            }}
        """)
        button_layout.addWidget(copy_btn)
        
        # Copy steps button
        copy_steps_btn = QPushButton("📖 Copy Steps")
        copy_steps_btn.clicked.connect(self.copy_steps)
        copy_steps_btn.setStyleSheet(copy_btn.styleSheet())
        button_layout.addWidget(copy_steps_btn)
        
        # Save solution button
        save_btn = QPushButton("💾 Save Solution")
        save_btn.clicked.connect(self.save_solution)
        save_btn.setStyleSheet(copy_btn.styleSheet())
        button_layout.addWidget(save_btn)
        
        button_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("❌ Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['error']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #d32f2f;
            }}
        """)
        button_layout.addWidget(close_btn)
        
        main_layout.addWidget(QWidget())  # Spacer
        main_layout.addLayout(button_layout)
    
    def _generate_formula_info(self):
        """Generate information about the mathematical method used."""
        method_info = []
        method_info.append("📐 Mathematical Method Information")
        method_info.append("=" * 50)
        method_info.append("")
        
        if "derivative" in self.calculation_type.lower():
            method_info.extend([
                "🧮 Differentiation Rules Applied:",
                "• Power Rule: d/dx(x^n) = n·x^(n-1)",
                "• Chain Rule: d/dx(f(g(x))) = f'(g(x))·g'(x)",
                "• Product Rule: d/dx(uv) = u'v + uv'",
                "• Quotient Rule: d/dx(u/v) = (u'v - uv')/v²"
            ])
        elif "integral" in self.calculation_type.lower():
            method_info.extend([
                "🧮 Integration Techniques:",
                "• Basic Integration: ∫x^n dx = x^(n+1)/(n+1) + C",
                "• Substitution: ∫f(g(x))g'(x) dx = ∫f(u) du",
                "• Integration by Parts: ∫u dv = uv - ∫v du",
                "• Trigonometric Integration"
            ])
        elif "solve" in self.calculation_type.lower():
            method_info.extend([
                "🧮 Equation Solving Methods:",
                "• Algebraic Manipulation",
                "• Factoring Techniques",
                "• Quadratic Formula: x = (-b ± √(b²-4ac))/2a",
                "• Numerical Methods (if applicable)"
            ])
        else:
            method_info.extend([
                "🧮 General Mathematical Operations:",
                "• Arithmetic Operations",
                "• Algebraic Simplification",
                "• SymPy Computer Algebra System",
                "• Exact Symbolic Computation"
            ])
        
        method_info.extend([
            "",
            "🔧 Computational Tools:",
            "• SymPy Library for symbolic mathematics",
            "• Python numerical computation",
            "• SAUNet 4.0 AI-powered analysis"
        ])
        
        return "\n".join(method_info)
    
    def _generate_analysis_info(self):
        """Generate analysis information about the solution."""
        analysis_info = []
        analysis_info.append("🔬 Solution Analysis")
        analysis_info.append("=" * 40)
        analysis_info.append("")
        
        analysis_info.extend([
            f"📝 Input Expression: {self.expression}",
            f"✅ Final Result: {self.result}",
            f"🎯 Problem Type: {self.calculation_type}",
            "",
            "🧠 AI Analysis Features:",
            "• Automatic problem type detection",
            "• Step-by-step solution generation",
            "• Error checking and validation",
            "• Multiple solution methods when applicable",
            "",
            "📊 Solution Properties:"
        ])
        
        # Add specific analysis based on result type
        try:
            if isinstance(self.result, (int, float)):
                analysis_info.extend([
                    f"• Result Type: Numerical ({type(self.result).__name__})",
                    f"• Value: {self.result}",
                    f"• Is Integer: {isinstance(self.result, int) or float(self.result).is_integer()}"
                ])
            elif isinstance(self.result, str):
                analysis_info.extend([
                    f"• Result Type: Symbolic Expression",
                    f"• Length: {len(self.result)} characters",
                    f"• Contains Variables: {'x' in self.result or 'y' in self.result}"
                ])
        except:
            analysis_info.append("• Result Type: Complex mathematical object")
        
        analysis_info.extend([
            "",
            "⚡ Computation Details:",
            "• Powered by SAUNet 4.0 AI Engine",
            "• Real-time mathematical analysis",
            "• Enhanced with ML preprocessing",
            "• ChatGPT-like intelligent problem solving"
        ])
        
        return "\n".join(analysis_info)
    
    def copy_result(self):
        """Copy the result to clipboard."""
        try:
            clipboard = QApplication.clipboard()
            clipboard.setText(str(self.result))
            if hasattr(self.parent(), 'statusBar') and self.parent().statusBar():
                self.parent().statusBar().showMessage("Result copied to clipboard!", 2000)
        except Exception as e:
            QMessageBox.information(self, "Copy", f"Result copied: {self.result}")
    
    def copy_steps(self):
        """Copy the step-by-step solution to clipboard."""
        try:
            clipboard = QApplication.clipboard()
            full_solution = f"Expression: {self.expression}\nResult: {self.result}\n\nSteps:\n{self.steps}"
            clipboard.setText(full_solution)
            if hasattr(self.parent(), 'statusBar') and self.parent().statusBar():
                self.parent().statusBar().showMessage("Solution steps copied to clipboard!", 2000)
        except Exception as e:
            QMessageBox.information(self, "Copy", f"Steps copied to clipboard")
    
    def save_solution(self):
        """Save the complete solution to a file."""
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Solution", f"SAUNet_Solution_{self.calculation_type}.txt", 
                "Text Files (*.txt);;All Files (*)", options=options
            )
            
            if file_name:
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write("🧮 SAUNet 4.0 - Detailed Mathematical Solution\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Problem Type: {self.calculation_type}\n")
                    f.write(f"Expression: {self.expression}\n")
                    f.write(f"Result: {self.result}\n\n")
                    f.write("Step-by-Step Solution:\n")
                    f.write("-" * 30 + "\n")
                    f.write(self.steps if self.steps else "No detailed steps available.")
                    f.write(f"\n\nGenerated by SAUNet 4.0 Advanced Scientific Calculator")
                
                if hasattr(self.parent(), 'statusBar') and self.parent().statusBar():
                    self.parent().statusBar().showMessage(f"Solution saved to {file_name}", 3000)
                else:
                    QMessageBox.information(self, "Save", f"Solution saved to {file_name}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not save solution: {str(e)}")


class EnhancedTextEdit(QTextEdit):
    """Enhanced text edit with improved scrolling and wheel support."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Enhanced scroll settings
        self.verticalScrollBar().setStyleSheet("""
            QScrollBar:vertical {
                background: #f0f0f0;
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #2196F3;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {{
                background: #1976D2;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
    
    def wheelEvent(self, event: QWheelEvent):
        """Enhanced wheel event handling for smooth scrolling."""
        # Get the scroll delta
        delta = event.angleDelta().y()
        
        # Smooth scrolling with smaller steps
        scroll_lines = 3  # Number of lines to scroll per wheel step
        
        if delta > 0:
            # Scroll up
            for _ in range(scroll_lines):
                self.verticalScrollBar().setValue(
                    self.verticalScrollBar().value() - 1
                )
        else:
            # Scroll down
            for _ in range(scroll_lines):
                self.verticalScrollBar().setValue(
                    self.verticalScrollBar().value() + 1
                )
        
        event.accept()


class ScrollableLineEdit(QLineEdit):
    """Enhanced QLineEdit with mouse wheel scroll support for horizontal navigation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for horizontal scrolling and value increment/decrement."""
        # Check if Ctrl is pressed for value modification
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Get current cursor position
            cursor_pos = self.cursorPosition()
            text = self.text()
            
            # Try to find a number at cursor position
            if cursor_pos < len(text) and text[cursor_pos].isdigit():
                # Find the complete number
                start = cursor_pos
                end = cursor_pos
                
                # Find start of number
                while start > 0 and (text[start - 1].isdigit() or text[start - 1] == '.'):
                    start -= 1
                
                # Find end of number
                while end < len(text) and (text[end].isdigit() or text[end] == '.'):
                    end += 1
                
                try:
                    # Extract and modify the number
                    number_str = text[start:end]
                    current_number = float(number_str)
                    
                    # Increment or decrement based on wheel direction
                    delta = 1 if event.angleDelta().y() > 0 else -1
                    new_number = current_number + delta
                    
                    # Replace the number in the text
                    new_text = text[:start] + str(int(new_number) if new_number.is_integer() else new_number) + text[end:]
                    self.setText(new_text)
                    self.setCursorPosition(cursor_pos)
                    
                    event.accept()
                    return
                except ValueError:
                    pass
        
        # Regular horizontal scrolling
        if event.angleDelta().y() != 0:
            # Scroll horizontally based on wheel direction
            scroll_amount = 3  # Characters to scroll
            current_pos = self.cursorPosition()
            
            if event.angleDelta().y() > 0:  # Scroll up (move cursor left)
                new_pos = max(0, current_pos - scroll_amount)
            else:  # Scroll down (move cursor right)
                new_pos = min(len(self.text()), current_pos + scroll_amount)
            
            self.setCursorPosition(new_pos)
            event.accept()
        else:
            super().wheelEvent(event)


class AdvancedTextEdit(QTextEdit):
    """Enhanced QTextEdit with better scrolling and navigation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        
    def wheelEvent(self, event: QWheelEvent):
        """Enhanced wheel event with smooth scrolling."""
        # Check for Ctrl+Wheel for zoom
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            font = self.font()
            current_size = font.pointSize()
            
            if event.angleDelta().y() > 0:  # Zoom in
                new_size = min(24, current_size + 1)
            else:  # Zoom out
                new_size = max(8, current_size - 1)
            
            font.setPointSize(new_size)
            self.setFont(font)
            event.accept()
        else:
            # Use default scrolling behavior
            super().wheelEvent(event)


class ResultDetailDialog(QDialog):
    """Enhanced popup dialog for displaying detailed calculation results and steps."""
    
    def __init__(self, parent, expression, result, steps=None, solution_type="Calculation"):
        super().__init__(parent)
        self.expression = expression
        self.result = result
        self.steps = steps
        self.solution_type = solution_type
        
        self.setWindowTitle(f"📊 {solution_type} Details - SAUNet 4.0")
        self.setMinimumSize(700, 500)
        self.resize(800, 600)
        
        # Set window flags for better appearance
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowSystemMenuHint | Qt.WindowType.WindowMinMaxButtonsHint)
        
        self.init_ui()
        self.apply_theme(getattr(parent, 'is_dark_mode', False))
    
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Header section
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        header_layout = QVBoxLayout(header_frame)
        
        # Title
        title = QLabel(f"📊 {self.solution_type} Results")
        title.setFont(QFont(FONT_FAMILY, FONT_SIZE + 4, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        
        # Expression
        if self.expression:
            expr_label = QLabel("📝 Problem:")
            expr_label.setFont(QFont(FONT_FAMILY, FONT_SIZE + 1, QFont.Weight.Bold))
            header_layout.addWidget(expr_label)
            
            expr_text = QLabel(str(self.expression))
            expr_text.setFont(QFont("Courier", FONT_SIZE + 1))
            expr_text.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px; border: 1px solid #ccc;")
            expr_text.setWordWrap(True)
            expr_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            header_layout.addWidget(expr_text)
        
        # Result
        result_label = QLabel("✅ Result:")
        result_label.setFont(QFont(FONT_FAMILY, FONT_SIZE + 1, QFont.Weight.Bold))
        header_layout.addWidget(result_label)
        
        result_text = QLabel(str(self.result))
        result_text.setFont(QFont("Courier", FONT_SIZE + 2, QFont.Weight.Bold))
        result_text.setStyleSheet("padding: 15px; background-color: #e8f5e8; border-radius: 5px; border: 2px solid #4CAF50; color: #2E7D32;")
        result_text.setWordWrap(True)
        result_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_layout.addWidget(result_text)
        
        layout.addWidget(header_frame)
        
        # Steps section (if available)
        if self.steps:
            steps_label = QLabel("📋 Step-by-Step Solution:")
            steps_label.setFont(QFont(FONT_FAMILY, FONT_SIZE + 1, QFont.Weight.Bold))
            layout.addWidget(steps_label)
            
            # Scrollable steps area
            self.steps_area = AdvancedTextEdit()
            self.steps_area.setReadOnly(True)
            self.steps_area.setFont(QFont("Courier", FONT_SIZE))
            self.steps_area.setText(str(self.steps))
            self.steps_area.setMinimumHeight(200)
            layout.addWidget(self.steps_area)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Copy result button
        copy_result_btn = QPushButton("📋 Copy Result")
        copy_result_btn.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        copy_result_btn.clicked.connect(self.copy_result)
        button_layout.addWidget(copy_result_btn)
        
        # Copy all button
        copy_all_btn = QPushButton("📄 Copy All")
        copy_all_btn.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        copy_all_btn.clicked.connect(self.copy_all)
        button_layout.addWidget(copy_all_btn)
        
        # Export button
        export_btn = QPushButton("💾 Export")
        export_btn.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        export_btn.clicked.connect(self.export_solution)
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("✖️ Close")
        close_btn.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        close_btn.clicked.connect(self.close)
        close_btn.setDefault(True)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def apply_theme(self, is_dark_mode):
        """Apply light or dark theme to the dialog."""
        if is_dark_mode:
            self.setStyleSheet("""
                QDialog {
                    background-color: #1E1E2E;
                    color: #FFFFFF;
                }
                QLabel {
                    color: #FFFFFF;
                }
                QTextEdit {
                    background-color: #313244;
                    color: #FFFFFF;
                    border: 1px solid #45475A;
                    border-radius: 5px;
                }
                QPushButton {
                    background-color: #313244;
                    color: #FFFFFF;
                    border: 1px solid #45475A;
                    border-radius: 5px;
                    padding: 8px;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #45475A;
                }
                QPushButton:pressed {
                    background-color: #585B70;
                }
                QFrame {
                    background-color: #313244;
                    border: 1px solid #45475A;
                    border-radius: 8px;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background-color: #FFFFFF;
                    color: #212121;
                }
                QTextEdit {
                    background-color: #FFFFFF;
                    color: #212121;
                    border: 1px solid #CCCCCC;
                    border-radius: 5px;
                }
                QPushButton {
                    background-color: #F5F5F5;
                    color: #212121;
                    border: 1px solid #CCCCCC;
                    border-radius: 5px;
                    padding: 8px;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #E0E0E0;
                }
                QPushButton:pressed {
                    background-color: #D0D0D0;
                }
                QFrame {
                    background-color: #F8F9FA;
                    border: 1px solid #DEE2E6;
                    border-radius: 8px;
                }
            """)
    
    def copy_result(self):
        """Copy just the result to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(str(self.result))
        
        # Show temporary message
        btn = self.sender()
        original_text = btn.text()
        btn.setText("✅ Copied!")
        QTimer.singleShot(1500, lambda: btn.setText(original_text))
    
    def copy_all(self):
        """Copy expression, result, and steps to clipboard."""
        content = []
        if self.expression:
            content.append(f"Problem: {self.expression}")
        content.append(f"Result: {self.result}")
        if self.steps:
            content.append(f"Steps:\n{self.steps}")
        
        clipboard = QApplication.clipboard()
        clipboard.setText("\n\n".join(content))
        
        # Show temporary message
        btn = self.sender()
        original_text = btn.text()
        btn.setText("✅ Copied!")
        QTimer.singleShot(1500, lambda: btn.setText(original_text))
    
    def export_solution(self):
        """Export the solution to a text file."""
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Export Solution", f"solution_{int(QTimer().timerId())}.txt",
                "Text Files (*.txt);;All Files (*)", options=options
            )
            
            if file_name:
                content = []
                content.append("SAUNet 4.0 - Mathematical Solution")
                content.append("=" * 40)
                if self.expression:
                    content.append(f"Problem: {self.expression}")
                content.append(f"Result: {self.result}")
                if self.steps:
                    content.append("\nStep-by-Step Solution:")
                    content.append(str(self.steps))
                content.append("\n" + "=" * 40)
                content.append("Generated by SAUNet 4.0 Advanced Scientific Calculator")
                
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write("\n".join(content))
                
                # Show success message
                btn = self.sender()
                original_text = btn.text()
                btn.setText("✅ Exported!")
                QTimer.singleShot(2000, lambda: btn.setText(original_text))
                
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export solution:\n{str(e)}")