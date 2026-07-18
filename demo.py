# from src.logger import logger
# logger.info("checking logger")
from src.exception import CustomException
import sys
try:
    num = 10/0
except Exception as e:
    raise CustomException(e,sys)
