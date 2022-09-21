
import pyfirmata
from tkinter import *


def move_servo11(angle):
    pin11.write(angle)
def move_servo10(angle):
    pin10.write(angle)
def move_servo9(angle):
    pin9.write(angle)

def move_servo6(angle):
    pin8.write(angle)



def main():
    global pin11
    global pin10
    global pin9
    global pin8


    board = pyfirmata.Arduino('COM4')

    iter8 = pyfirmata.util.Iterator(board)
    iter8.start()

    pin11 = board.get_pin('d:11:s')
    pin10 = board.get_pin('d:10:s')
    pin9 = board.get_pin('d:9:s')
    pin8 = board.get_pin('d:6:s')

    root = Tk()
    scale = Scale(root, command=move_servo11, from_=0,to=600,
                  orient=HORIZONTAL, length=400, label='Klaue')
    scale.pack(anchor=CENTER)
    scale.set(90)


    scale2 = Scale(root, command=move_servo10, from_=0, to=600,
                      orient=HORIZONTAL, length=400, label='Arm_1')
    scale2.pack(anchor=CENTER)
    scale2.set(90)

    scale3 = Scale(root, command=move_servo9, from_=0, to=600,
                   orient=HORIZONTAL, length=400, label='Arm_3')
    scale3.pack(anchor=CENTER)
    scale3.set(90)

    scale4 = Scale(root, command=move_servo6, from_=0, to=600,
                   orient=HORIZONTAL, length=400, label='Arm_2')
    scale4.pack(anchor=CENTER)
    scale4.set(90)












    root.mainloop()


main()