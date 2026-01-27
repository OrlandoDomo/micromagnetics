import logging

def get_logger(log_filename="saf-relax"):
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