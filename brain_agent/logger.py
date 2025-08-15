#!/usr/bin/env python3
"""
Logger for brain-agent - Simple standalone logger using BRAIN_AGENT_DEBUG env variable.

Environment variable:
- BRAIN_AGENT_DEBUG: Set to "1" to enable debug prints to console. Set to "0" to disable.
  Default is "1" (enabled).

Logs are always written to file, debug prints are conditional.
Error logs are renamed with _ERROR suffix.
"""

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

def log_and_print(message: str, level: str = "INFO", error_details: str = None):
    """
    Log a message to file and conditionally print to console.
    
    Args:
        message: The main log message
        level: Log level (INFO, DEBUG, ERROR, etc.)
        error_details: Additional error details (for exceptions)
    """
    # Create log directory if it doesn't exist
    log_dir = Path.home() / '.brain_agent' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Check debug flag (same as global exception handler)
    debug_enabled = os.getenv("BRAIN_AGENT_DEBUG", "1") == "1"
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'brain_agent_{timestamp}.log'
    
    # Track if any errors were logged
    has_errors = False
    
    # Format log entry
    log_entry = f"[{datetime.now().isoformat()}] [{level}] {message}"
    if error_details:
        log_entry += f"\nError Details: {error_details}"
        has_errors = True
    
    # Write to log file
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
            f.flush()
    except Exception as e:
        # Fallback to stderr if file logging fails
        print(f"Failed to write to log file: {e}", file=sys.stderr)
    
    # Conditionally print to console
    if debug_enabled:
        print(f"[BRAIN-AGENT] {message}")
        if error_details:
            print(f"[BRAIN-AGENT] Error Details: {error_details}")
    
    # Rename log file if errors occurred
    if has_errors:
        error_log_file = log_dir / f'brain_agent_{timestamp}_ERROR.log'
        try:
            log_file.rename(error_log_file)
        except Exception:
            pass  # If rename fails, keep original name

def log_exception(exception: Exception, context: str = ""):
    """
    Log an exception with full traceback.
    
    Args:
        exception: The exception object
        context: Additional context about where the exception occurred
    """
    error_details = traceback.format_exc()
    message = f"Exception in {context}: {str(exception)}" if context else f"Exception: {str(exception)}"
    log_and_print(message, level="ERROR", error_details=error_details)

def debug_print(message: str):
    """
    Print a debug message (only when debug is enabled).
    
    Args:
        message: Debug message to print
    """
    log_and_print(message, level="DEBUG")

def info_print(message: str):
    """
    Print an info message.
    
    Args:
        message: Info message to print
    """
    log_and_print(message, level="INFO")

def error_print(message: str, error_details: str = None):
    """
    Print an error message.
    
    Args:
        message: Error message to print
        error_details: Additional error details
    """
    log_and_print(message, level="ERROR", error_details=error_details)