import socket
import json
import time

def start_server(host='0.0.0.0', port=50007):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"Server listening on {host}:{port}")

    conn, addr = server.accept()
    print(f"Connected by {addr}")

    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break

            decoded = json.loads(data.decode('utf-8'))
            print(f"Received from Unity: {decoded}")

            # 模拟长时间处理
            for progress in range(0, 101, 20):
                message = {
                    "status": "progress",
                    "progress": progress / 100.0
                }
                conn.sendall((json.dumps(message) + "\n").encode('utf-8'))  # 注意加换行分隔！

                time.sleep(1)  # 模拟耗时处理

            # 最后发送最终结果
            result = {
                "status": "success",
                "result": f"Processed param={decoded['param']}"
            }
            conn.sendall((json.dumps(result) + "\n").encode('utf-8'))

    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        conn.close()
        server.close()

if __name__ == "__main__":
    start_server()
