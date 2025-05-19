import os
import logging
import datetime
import torch.distributed as dist

def setup_logger(log_dir):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    logger = logging.getLogger(f"Process_{rank}")
    logger.setLevel(logging.INFO)

    # 清理现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 禁用全局日志器的所有处理器
    logging.getLogger().handlers.clear()

    # 获取当前时间并格式化为字符串
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"LOG_training_rank_{rank}_{current_time}.txt"  # 将时间戳加入文件名
    log_dir = log_dir
    log_path = os.path.join(log_dir, log_filename)

    # 创建一个输出到 .txt 文件的 handler
    file_handler = logging.FileHandler(log_path, mode='a')  # 使用 'a' 来追加写入
    #file_formatter = logging.Formatter(f"[Rank {rank}] %(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_formatter = logging.Formatter(
        f"[Rank {rank}] %(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)  # 使用相同的格式化器
    logger.addHandler(file_handler)  # 将文件 handler 添加到 logger 中

    return logger