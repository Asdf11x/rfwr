"""
netowrking.py: Contains the networking methods for main.py
"""

import socket


class Network:

    def __init__(self):
        # General settings
        self.IP = "127.0.0.1"

        # UDP settings
        self.UDP_PORT_SEND = 5005
        self.UDP_PORT_REC = 5006

    def send(self, message):
        """
        Send passed data
        :param message: data to be send via UDP
        """
        message = str(message).encode()
        sock = socket.socket(socket.AF_INET,  # Internet
                             socket.SOCK_DGRAM)  # UDP
        sock.sendto(message, (self.IP, self.UDP_PORT_SEND))

    def receive(self):
        """
        Receive data
        :return: received data from the lwpr algorithm
        """
        sock = socket.socket(socket.AF_INET,  # Internet
                             socket.SOCK_DGRAM)  # UDP
        sock.bind((self.IP, self.UDP_PORT_REC))

        while True:
            data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
            return data

