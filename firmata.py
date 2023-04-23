
import pyfirmata
from tkinter import *


def move_servo(angle):
    pin9.write(angle)


def main():
    global pin9

    board = pyfirmata.Arduino('COM4')

    iter8 = pyfirmata.util.Iterator(board)
    iter8.start()

    pin9 = board.get_pin('d:3:s')

    root = Tk()
    scale = Scale(root, command=move_servo, from_=30,to=105,
                  orient=HORIZONTAL, length=400, label='Angle')
    scale.pack(anchor=CENTER)

    root.mainloop()


main()