import logging
import os

def get_logger(logger_name, log_filename="saf-relax"):
  """
  Creates or retrieves a logger that writes to a specific file.
  """
  # 1. Create a logger with the name of the calling module
  logger = logging.getLogger(logger_name)
  
  # 2. Prevent the logger from doubling up lines if called multiple times
  if not logger.handlers:
    logger.setLevel(logging.INFO)
    
    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # 3. Define your format
    formatter = logging.Formatter(
      fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      datefmt="%Y/%m/%d %H:%M:%S %p"
    )
    
    # 4. Create and attach the File Handler
    file_handler = logging.FileHandler(
      filename=f"logs/{log_filename}.log", 
      encoding='utf-8', 
      mode='a' # 'a' for append, 'w' wipes the file every time the script runs
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 5. Optional: Also print to the terminal console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent logs from bubbling up to the root logger
    logger.propagate = False

  return logger

def get_logger_old(log_filename="saf-relax"):
  logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  logging_date_format = "%Y/%m/%d %H:%M:%S %p"
  logger = logging.getLogger(__name__)
  logging.basicConfig(
    handlers = [
      logging.FileHandler(
        filename = f"logs/{log_filename}.log", 
        encoding = 'utf-8', 
        mode = 'w',
        delay = True
      )
    ],
    format = logging_format,
    datefmt = logging_date_format,
    level = logging.INFO
  )
  return logger