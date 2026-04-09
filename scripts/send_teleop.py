import sys, socket, json
if len(sys.argv) < 2:
    print("Usage: send_teleop.py f,t,lock,grab")
    sys.exit(1)
parts = sys.argv[1].split(',')
arr = [float(p) for p in parts]
msg = json.dumps({"a": arr})
with socket.create_connection(('127.0.0.1', 5001), timeout=1) as s:
    s.sendall(msg.encode('utf-8'))
print(f"sent: {msg}")
