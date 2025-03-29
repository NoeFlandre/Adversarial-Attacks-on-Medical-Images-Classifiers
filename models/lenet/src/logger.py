import logging
import os
import datetime

def setup_logger(name='lenet', log_type=None, level=logging.INFO):
    """
    Set up a logger that writes to both console and a file.
    
    Args:
        name (str): Name of the logger
        log_type (str): Type of log (e.g., 'training_evaluation', 'adversarial_attacks', 'visualize_attacks')
        level (int): Logging level
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Set default log directory to be within the current module's directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_base_dir = os.path.join(current_dir, 'logs')
    
    # Create logs base directory if it doesn't exist
    if not os.path.exists(logs_base_dir):
        os.makedirs(logs_base_dir, exist_ok=True)
    
    # Create a unique log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use log_type to determine subfolder, default to name if not provided
    log_subfolder = log_type if log_type else name
    log_dir = os.path.join(logs_base_dir, f"{log_subfolder}.log")
    
    # Create log subfolder if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Create the log file path
    log_file = os.path.join(log_dir, f"lenet_{timestamp}.log")
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create file handler with 'a' mode for appending
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger