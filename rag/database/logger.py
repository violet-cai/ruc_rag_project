# logger.py
import logging

# 配置数据库相关的 logger
db_logger = logging.getLogger("database")
db_logger.setLevel(logging.INFO)

# 创建控制台输出的 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建文件输出的 Handler
file_handler = logging.FileHandler("baike_spider.log")
file_handler.setLevel(logging.ERROR)

# 定义日志输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将 Handler 添加到 logger
db_logger.addHandler(console_handler)
db_logger.addHandler(file_handler)