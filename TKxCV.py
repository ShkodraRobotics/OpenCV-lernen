import cv2 as cv
import numpy as np
from PIL import Image , ImageTk
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.geometry("500x500")

label = tk.Label(root)
label.grid(column=0,row=0)

cap = cv.VideoCapture(2)

def frame_():

    vision = cv.cvtColor(cap.read()[1],cv.COLOR_RGB2BGR)

    img = Image.fromarray(vision)

    img_tk =ImageTk.PhotoImage(image=img)

    label.imgtk = img_tk
    label.configure(image=img_tk)

label.after(1,frame_)

frame_()
root.mainloop()























