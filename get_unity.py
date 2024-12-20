import socket
import json
import base64

HOST = "192.168.1.51"  # local host
PORT = 8965

# 創建 socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print(f"server listen on {HOST}:{PORT} ...")

while True:
    try:
        # 接受連線
        client_socket, addr = server_socket.accept()
        print(f"connection from {addr} ...")

        # 接收資料
        data = client_socket.recv(10 * 1024 * 1024)  # 允許最大 10 MB 資料
        if not data:
            print("No data")
            client_socket.close()
            continue

        # 解析 JSON 資料
        try:
            json_data = json.loads(data.decode("utf-8"))
            base64_image = json_data.get("image")
            filename = json_data.get("filename", "output.jpg")

            if base64_image:
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(base64_image))
                print(f" {filename}")

            else:
                print("No image in JSON")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"JSON decode error: {e}")

        # 關閉連線
        client_socket.close()
    except KeyboardInterrupt:
        print("Server closing...")
        break

server_socket.close()
