from src.exception import CustomException
import sys
import yaml


def read_yaml_file(filepath:str):
    try:
        with open(filepath, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)