import socket

def receive_udp_message(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    print(f"UDP server listening on {ip}:{port}")
    data = ''
    while True:
        data, addr = sock.recvfrom(1024)
        if data.decode().strip('"') == "exit":
            print("Received exit command")
            sock.close()
            break
        print(f'Message received: {data.decode()} ')


if __name__ == "__main__":
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    receive_udp_message(UDP_IP, UDP_PORT)
