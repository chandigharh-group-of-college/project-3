#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import traceback
import threading
import atexit
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QFontDatabase
import cv2
from sympy import sympify, simplify, symbols
import re

from calculator import Calculator
from ui import CalculatorUI
from solver import EquationSolver

# Global variables to track threads
active_threads = []

def excepthook(exc_type, exc_value, exc_tb):
    """Custom exception hook for unhandled exceptions."""
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"Error: {tb}")
    QMessageBox.critical(None, "Error", f"An unexpected error occurred:\n{exc_value}")

def cleanup_threads():
    """Clean up any active threads when the application exits."""
    for thread in active_threads:
        if thread.is_alive():
            print(f"Cleaning up thread: {thread.name}")
            # Note: We don't forcefully terminate threads as it can lead to resource leaks
            # Instead, we rely on daemon=True for background threads

def main():
    """Main function to start the SAUNet 4.0 calculator application."""
    print("🚀 Starting SAUNet 4.0 Advanced Scientific Calculator...")
    print("   Version: 4.0.0")
    print("   Features: OCR, History, Smooth Transitions, AI-Powered Math")
    print("   Loading...")
    
    # Set up exception handling for uncaught exceptions
    sys.excepthook = excepthook
    
    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("SAUNet 4.0")
    app.setApplicationVersion("4.0.0")
    
    # Load custom fonts (fallback to system fonts if not available)
    for font_family in ["Roboto Mono", "Fira Code"]:
        try:
            QFontDatabase.addApplicationFont(f"fonts/{font_family}.ttf")
        except:
            # Skip if font file doesn't exist
            pass
    
    try:
        # Create calculator and solver instances
        calculator = Calculator()
        solver = EquationSolver()
        
        # Create the UI
        ui = CalculatorUI(calculator, solver)
        
        # Register the cleanup function to be called when application exits
        atexit.register(cleanup_threads)
        
        # Show the UI
        ui.show()
        print("✅ SAUNet 4.0 successfully loaded!")
        print("   Enjoy your enhanced calculation experience! 🧮✨")
        
        # Run the application event loop
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 