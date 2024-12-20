import argparse
import main
import socket
import threading
import datetime
import os
import json
import base64
from PIL import Image
from io import BytesIO
import requests

HOST = "192.168.1.51"
PORT = 8964

directory = "history"
if not os.path.exists(directory):
    os.makedirs(directory)

line_notify_token = "2OwGrjbkvJKx9jCCO7Nwr8DVP1OBWayyfjHFdcBn5d9"


def save_text_to_file(text, filename):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"[TEXT SAVED] {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save text: {e}")


def read_report(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            print("File content as string:")
            print(content)
            return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def send_line_notify(token, message, image_path_1=None, image_path_2=None):
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}

    # Send images first
    if image_path_1:
        with open(image_path_1, "rb") as image_file_1:
            files = {"imageFile": image_file_1}
            response = requests.post(
                url, headers=headers, data={"message": "Image 1"}, files=files
            )
            if response.status_code == 200:
                print("[IMAGE 1 SENT] Successfully sent image 1.")
            else:
                print(
                    f"[ERROR] Failed to send image 1: {response.status_code}, {response.text}"
                )

    if image_path_2:
        with open(image_path_2, "rb") as image_file_2:
            files = {"imageFile": image_file_2}
            response = requests.post(
                url, headers=headers, data={"message": "Image 2"}, files=files
            )
            if response.status_code == 200:
                print("[IMAGE 2 SENT] Successfully sent image 2.")
            else:
                print(
                    f"[ERROR] Failed to send image 2: {response.status_code}, {response.text}"
                )

    # Split the message into chunks if necessary
    max_length = 800
    chunks = [message[i : i + max_length] for i in range(0, len(message), max_length)]

    for index, chunk in enumerate(chunks):
        data = {"message": chunk}
        response = requests.post(url, headers=headers, data=data)

        if response.status_code == 200:
            print(f"LINE Notify part {index+1} sent successfully!")
        else:
            print(
                f"Failed to send part {index+1}: {response.status_code}, {response.text}"
            )


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    while connected:
        try:
            length_data = conn.recv(4)
            if not length_data:
                print(f"Connection closed by {addr}")
                break

            data_length = int.from_bytes(length_data, "big")
            received_data = b""
            while len(received_data) < data_length:
                packet = conn.recv(data_length - len(received_data))
                if not packet:
                    raise ConnectionResetError("Connection closed unexpectedly")
                received_data += packet
            json_data = received_data.decode("utf-8")
            data = json.loads(json_data)
            text = data["text"]
            image_data = data["image"]

            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

            if not os.path.exists(f"{directory}/{formatted_time}"):
                os.makedirs(f"{directory}/{formatted_time}")

            text_path = f"{directory}/{formatted_time}/text.txt"
            text = f"{formatted_time}\n" + text
            save_text_to_file(text, text_path)

            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            path_name = f"{directory}/{formatted_time}/image1.png"
            image.save(path_name)

            print(text)

            last_char = text.strip()[-1] if text.strip() else ""
            if last_char == "A":
                image_path_2 = "unity_A.png"
            else:
                image_path_2 = "unity_B.png"

            main.run(text, "rag", formatted_time)  # "rag", "plain" or "exp".
            # TODO: The flag above doesn't work, fix this parser.
            report_path = f"{directory}/{formatted_time}/report.txt"
            report = read_report(report_path)
            notify = f"{text}\n{report}"
            send_line_notify(
                line_notify_token,
                notify,
                image_path_1=f"{directory}/{formatted_time}/image1.png",
                image_path_2=image_path_2,
            )
            print("send to client")
            conn.send("complete".encode("utf-8"))
        except:
            connected = False
            print(f"[{addr}] disconnect")
    conn.close()


def start():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[LISTENING] Server is listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()
            print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] server is starting...")

start()
