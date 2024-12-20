import argparse
import main
import socket
import threading
import datetime
import os
import json
import base64

HOST = "192.168.1.51"  # (localhost)
PORT = 8964

directory = "history"
if not os.path.exists(directory):
    os.makedirs(directory)


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    while connected:
        try:
            data = conn.recv(1024)
            re_list = json.loads(data.decode("utf-8"))
            print(f"[{addr}] {re_list[0]}")
            print(f"[{addr}] {re_list[1]}")
            # main.run(data1, "rag")  # "rag", "plain" or "exp".
            # TODO: Add the flag to the parser
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
