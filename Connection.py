from socket import *

HOST = "192.168.1.134" #local host
PORT = 8000 #open port 7000 for connection
s = socket(AF_INET, SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1) #how many connections can it receive at one time
conn, addr = s.accept() #accept the connection
print("Connected by: " , addr) #print the address of the person connected
while True:
    data = conn.recv(1024) #how many bytes of data will the server receive
    print("Received: ", repr(data))
    reply = input("Reply: ") #server's reply to the client
    conn.sendall(reply)
conn.close()