# from tkinter import *
# from PIL import ImageTk, Image
# from tkinter import filedialog
# from image_utils import *

# root = Tk()
# root.title("ATI")

# commands = ["Open Image", "Open RAW Image", "Save Image","Get Pixel", "Modify Pixel", "Create Circle", "Create Square"]

# clicked = StringVar()
# clicked.set(commands[0])

# # Open file window
# def open_file_window(self):
#     filename = filedialog.askopenfilename(initialdir="./photos", title="Select an Image" , filetypes=(("raw", "*.RAW"), ("pgm", "*.pgm"), ("ppm", "*.ppm"), ("jpg", "*.jpg"), ("png", "*.png")))
#     img = load(filename)
    


# # Open directory window
# def open_directory_window():
#     global entry2
#     root.filename = filedialog.asksaveasfilename(initialdir="./photos", title="Select an Directory")
#     entry2.delete(0, END)
#     entry2.insert(0, root.filename)


# # Open Image in new window
# def open_image():
#     global entry
#     global img
#     filename = entry.get()
#     img = load(filename, 290, 207)
#     img_tk = ImageTk.PhotoImage(Image.fromarray(img, "RGB"))

#     top = Toplevel()
#     label = Label(top, image=img_tk).pack()


# # Save image
# def save_image():
#     global entry
#     global entry2
#     global img

#     filename = entry.get()
#     dir = entry2.get()
#     # img = load(filename=filename)
#     img = ImageTk.PhotoImage(load(filename=filename))
#     save_filename = entry2.get()
#     save(img, save_filename)


# #Create an image with a circle
# def circle_image():
#     global img

#     top = Toplevel()
#     img = ImageTk.PhotoImage(bin_circle())
#     label = Label(top, image=img).pack()


# #Create an image with a square
# def square_image():
#     global img

#     top = Toplevel()
#     img = ImageTk.PhotoImage(bin_rectangle())
#     label = Label(top, image=img).pack()


# entry = Entry(root, width=40)
# entry.grid(row=1, column=0, columnspan=2)

# entry2 = Entry(root, width=40)

# button = Button(root, text=clicked.get(), command=open_image)
# button.grid(row=1,column=3)
# search_button = Button(root, text= "Search", command=open_file_window)
# search_button.grid(row=1, column=2)

# search_button2 = Button(root, text= "Search")


# # Show command
# def show():
#     global options
#     global entry
#     global button
#     global search_button
#     global entry2
#     global search_button2

#     command = clicked.get()
    
#     button.grid_forget()
#     search_button.grid_forget()
#     search_button2.grid_forget()
#     entry2.grid_forget()

#     if command == commands[0]:
#         button = Button(root, text=clicked.get(), command=open_image)
#         button.grid(row=1,column=3)

#         search_button = Button(root, text= "Search", command=open_file_window)
#         search_button.grid(row=1, column=2)

#     elif command == commands[1]:
#         button = Button(root, text=clicked.get(), command=save_image)
#         button.grid(row=1,column=3)

#         search_button = Button(root, text= "Search Image", command=open_file_window)
#         search_button.grid(row=1, column=2)

#         entry2 = Entry(root, width=40)
#         entry2.grid(row=2, column=0, columnspan=2)

#         search_button2 = Button(root, text= "Search Directory", command=open_directory_window)
#         search_button2.grid(row=2, column=2)
    
#     elif command == commands[2]:
#         button = Button(root, text=clicked.get(), command=circle_image)
#         button.grid(row=1, column=0, columnspan=4)

#     elif command == commands[3]:
#         button = Button(root, text=clicked.get(), command=square_image)
#         button.grid(row=1, column=0, columnspan=4)
     
#     else:
#         button = Button(root, text=clicked.get())
#         button.grid(row=1,column=3)

#         search_button = Button(root, text= "Search", command=open_file_window)
#         search_button.grid(row=1, column=2)

  

# options = OptionMenu(root, clicked, *commands)
# options.grid(row=0, column=0, columnspan=3, sticky="ew")
# select_button = Button(root, text="Select", command=show)
# select_button.grid(row=0, column=3, sticky="ew")

# root.mainloop()

