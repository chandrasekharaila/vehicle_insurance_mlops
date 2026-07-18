from types import ModuleType
from src.logger import logger
def error_message_detail(error_message, error_details: ModuleType) -> str:
    """
    Extracts explicit file name and line number details from the active traceback.
    """
    _, _, exc_tb = error_details.exc_info()
 
    filename = exc_tb.tb_frame.f_code.co_filename
    linenum = exc_tb.tb_lineno

    msg = str(error_message)

    error_report = (
        f"Error occurred in file - {filename}\n"
        f"Error occurred in linenum - {linenum}\n"
        f"Error message - {msg}"
    )
    logger.error(error_report)
    return error_report


class CustomException(Exception):
    def __init__(self, error_message, error_details: ModuleType):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self) -> str:
        return self.error_message