
# import sys

# try:
#     a = 1/0
# except Exception as e:
#     print(e)
# print(sys.exc_info())

# from src.logger.logging import logging

# logging.info("this is my testing")
# logging.info(" this my second testing")
# logging.info(" this my second testing")

# logging.info(" this my second testing")
# logging.info(" this my second testing")
# logging.info(" this my second testing")


import mlflow

def calculator(a, b, operation=None):
    if operation == "add":
        return (a+b)
    if operation == "sub":
        return (a-b)
    if operation == "mul":
        return (a*b)
    if operation == "div":
        return (a/b)

if __name__ == "__main__":
    a, b = 109, 3142
    oper = "div"
    with mlflow.start_run():
        result = calculator(a, b, oper)
        mlflow.log_param("a", a)
        mlflow.log_param("b", b)
        mlflow.log_param("operations", oper)
        print(f"Result for {oper} is {result}")
        mlflow.log_param("result", result)
