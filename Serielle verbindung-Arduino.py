import tkinter as tk
from tkinter import ttk
import serial
import sys
min = 100
tro =min
ser = serial.Serial("COM4",baudrate=115200, timeout=.1)

def send():


    #print(tro)
    ser.write(bytes(str(tro),'utf-8'))
    root.after(900,send)

def sup0(val):
    # Add semicolon after the value
    msg0 = "1" +val+";"
    print(msg0)
    # Encode the message
    encodedMessage = msg0.encode()
    # Send the encoded message
    ser.write(encodedMessage)
def sup1(val):
    # Add semicolon after the value
    msg1 ="2" + val+";"
    print(msg1)
    # Encode the message
    encodedMessage = msg1.encode()
    # Send the encoded message
    ser.write(encodedMessage)
def sup2(val):
    # Add semicolon after the value
    msg1 ="3" + val+";"
    print(msg1)
    # Encode the message
    encodedMessage = msg1.encode()
    # Send the encoded message
    ser.write(encodedMessage)
def sup3(val):
    # Add semicolon after the value
    msg1 ="4" + val+";"
    print(msg1)
    # Encode the message
    encodedMessage = msg1.encode()
    # Send the encoded message
    ser.write(encodedMessage)
def sup4(val):
    # Add semicolon after the value
    msg1 ="5" + val+";"
    print(msg1)
    # Encode the message
    encodedMessage = msg1.encode()
    # Send the encoded message
    ser.write(encodedMessage)


def move_servo2(pos):
    global tro
    tro = pos



root = tk.Tk()
sc = tk.Scale(root,command=sup0, from_=100, to=270,orient=tk.HORIZONTAL, length=400, label='Klaue')
sc.pack(anchor=tk.CENTER)
sc2 = tk.Scale(root,command=sup1, from_=100, to=650,orient=tk.HORIZONTAL, length=400, label='Arm_1')
#sc2.set(250)
sc2.pack(anchor=tk.CENTER)
sc3 = tk.Scale(root,command=sup2, from_=100, to=670,orient=tk.HORIZONTAL, length=400, label='Arm_2')
sc3.pack(anchor=tk.CENTER)
sc4 = tk.Scale(root,command=sup3, from_=100, to=670,orient=tk.HORIZONTAL, length=400, label='Arm_3')
sc4.pack(anchor=tk.CENTER)
sc5 = tk.Scale(root,command=sup4, from_=100, to=670,orient=tk.HORIZONTAL, length=400, label='Base')
sc5.pack(anchor=tk.CENTER)


root.mainloop()
