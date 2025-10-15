def log_title(message: str):
    """
    打印格式化的标题
    
    参数:
        message: 要显示的标题消息
    """
    total_length = 80
    message_length = len(message)
    padding = max(0, total_length - message_length - 4)  # 4 for the "=="
    left_padding = '=' * (padding // 2)
    right_padding = '=' * ((padding + 1) // 2)
    padded_message = f"{left_padding} {message} {right_padding}"
    
    # Python中使用colorama或termcolor来实现彩色输出
    # 这里提供基础版本和彩色版本
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)
        print(f"{Style.BRIGHT}{Fore.CYAN}{padded_message}{Style.RESET_ALL}")
    except ImportError:
        # 如果没有安装colorama，使用ANSI转义码
        print(f"\033[1;36m{padded_message}\033[0m")
