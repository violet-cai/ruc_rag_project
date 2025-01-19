import subprocess
import time
import requests
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from rag.config.config import Config

config = Config()

SERVER_SCRIPT = config["server_script"]
CLIENT_SCRIPT = config["client_script"]
SERVER_URL = config["server_url"]


def start_server():
    """
    启动服务端
    """
    print("启动服务端...")
    server_process = subprocess.Popen(
        ["python", SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return server_process

def wait_for_server():
    """
    等待服务端启动
    """
    print("等待服务端启动...")
    while True:
        try:
            response = requests.get(f"{SERVER_URL}/status")
            if response.status_code == 200:
                status = response.json()
                print(f"服务端已启动！状态：{status['status']}, 运行时间：{status['uptime']:.2f} 秒")
                break
        except requests.exceptions.ConnectionError:
            # 如果连接失败，等待 1 秒后重试
            time.sleep(1)
            continue

def main():
    # 启动服务端
    server_process = start_server()
    wait_for_server()

    # 启动客户端
    print("启动客户端...")
    try:
        subprocess.run(["python", CLIENT_SCRIPT], check=True)
    except subprocess.CalledProcessError as e:
        print(f"客户端脚本运行失败：{e}")
    except KeyboardInterrupt:
        print("\n用户中断，退出程序...")
    finally:
        # 关闭服务端
        print("关闭服务端...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()
    # 当前示例的输入
    # query = "如果在准备材料的过程中遇到问题，应该联系谁呢？"