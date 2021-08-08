from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog

root = Tk()
root.title("ATI Interfaz")

commands = ["Abrir Imagen", "Guardar Imagen"]

clicked = StringVar()
clicked.set(commands[0])

# Open file window
def open_file_window():
    global entry
    root.filename = filedialog.askopenfilename(initialdir="./photos", title="Select an Image" , filetypes=(("raw", "*.RAW"), ("pgm", "*.pgm"), ("ppm", "*.ppm"), ("jpg", "*.jpg"), ("png", "*.png")))
    entry.delete(0, END)
    entry.insert(0, root.filename)


entry = Entry(root, width=40)
entry.grid(row=1, column=0, columnspan=2)
button = Button(root, text=clicked.get())
button.grid(row=1,column=3)
search_button = Button(root, text= "Search", command=open_file_window)
search_button.grid(row=1, column=2)

# Show command
def show():
    global options
    global entry
    global button
    global search_button

    command = clicked.get()
    
    button.grid_forget()
    button = Button(root, text=clicked.get())
    button.grid(row=1,column=3)

    

options = OptionMenu(root, clicked, *commands)
options.grid(row=0, column=0, columnspan=3, sticky="ew")
select_button = Button(root, text="Select", command=show)
select_button.grid(row=0, column=3, sticky="ew")

root.mainloop()

