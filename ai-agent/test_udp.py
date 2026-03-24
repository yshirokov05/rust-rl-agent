import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', 5000))
sock.settimeout(10.0)

print("Listening for UDP vision packets on 127.0.0.1:5000...")
try:
    data, addr = sock.recvfrom(65536)
    print(f"Received {len(data)} bytes from {addr}")
    parsed = json.loads(data.decode('utf-8'))
    print(f"Keys: {list(parsed.keys())}")
    if 'SemanticMapBase64' in parsed:
        print(f"SemanticMapBase64 length: {len(parsed['SemanticMapBase64'])}")
except socket.timeout:
    print("TIMEOUT: No data received in 10 seconds.")
except Exception as e:
    print(f"ERROR: {e}")
finally:
    sock.close()
