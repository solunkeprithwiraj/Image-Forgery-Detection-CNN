import logging

def setup_logger():
    """Configure and return a logger for the API"""
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("image_forgery_api")
    return logger

# Create a global logger instance
logger = setup_logger()