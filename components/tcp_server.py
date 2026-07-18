# components/tcp_server.py
import socket
import threading
import logging
from queue import Queue
from datetime import datetime
from time import perf_counter
from config import HOST, PORT, MESSAGE_DELIMITER, ACK_MESSAGE

logger = logging.getLogger(__name__)

class TCPServer:
    """
    Handles TCP connections and receives raw data from the client.
    """
    def __init__(self, output_queue: Queue):
        self.host = HOST
        self.port = PORT
        self.output_queue = output_queue
        self.server_socket = None
        self.running = False
        self.client_thread = None

    def start(self):
        """Starts the TCP server in a new thread."""
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1) # Only one client at a time
        logger.debug(f"TCP Server listening on {self.host}:{self.port}")
        
        server_thread = threading.Thread(target=self._accept_clients, daemon=True)
        server_thread.start()

    def stop(self):
        """Stops the TCP server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logger.debug("TCP Server stopped.")

    def _accept_clients(self):
        """Main loop for accepting client connections."""
        while self.running:
            conn, addr = self.server_socket.accept()
            logger.debug(f"Client connected from {addr}")
            # If a client thread is already running, wait for it to finish
            if self.client_thread and self.client_thread.is_alive():
                logger.debug("New client connected, but previous one is still active. Waiting.")
                self.client_thread.join()

            self.client_thread = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
            self.client_thread.start()

    def _handle_client(self, conn: socket.socket, addr):
        """Handles a single client connection."""
        logger.debug(f"Handling client {addr}")
        buffer = b""
        conn.settimeout(30.0)

        while self.running:
            data = conn.recv(4096 * 10)
            if not data:
                logger.debug(f"Client {addr} disconnected.")
                break
            
            buffer += data
            while MESSAGE_DELIMITER in buffer:
                message_data, _, buffer = buffer.partition(MESSAGE_DELIMITER)
                self.output_queue.put(
                    {
                        "payload": message_data,
                        "ingress_wall": datetime.now(),
                        "ingress_perf": perf_counter(),
                    }
                )
                
                conn.sendall(ACK_MESSAGE)

        logger.debug(f"Closing connection with {addr}")
        conn.close()
        # Signal to the main controller that the session is over
        self.output_queue.put(None) # Sentinel value
