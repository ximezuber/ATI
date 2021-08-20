from matplotlib.pyplot import title
from secondWindow import secondWindow
from tkinter import * 
from image_utils import *
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


class mainWindow:

    def __init__(self):
        self.root = Tk()
        self.root.title("ATI")

        # Menu 
        self.menubar = Menu(self.root)
        self.image_menu = Menu(self.menubar, tearoff=0)

        # Image
        self.image_menu.add_command(label="Load", command=self.open_file_window)
        self.image_menu.add_command(label="Save", command=self.save_image_window)
        self.image_menu.add_separator()
        self.image_menu.add_command(label="Create Circle Image", command=self.create_circle)
        self.image_menu.add_command(label="Create Square Image", command=self.create_square)
        self.image_menu.add_separator()
        self.image_menu.add_command(label="Exit", command=self.ask_quit)

        self.menubar.add_cascade(label="Image", menu=self.image_menu)

        # Edit
        self.edit_menu = Menu(self.menubar, tearoff=0)
        self.edit_menu.add_command(label="Get Pixel", command=self.get_pixel_value)
        self.edit_menu.add_command(label="Modify Pixel", command=self.modify_pixel)
        self.edit_menu.add_command(label="Copy into Other", command=self.copy_img_into_other)   
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Add")    
        self.edit_menu.add_command(label="Substract")
        self.edit_menu.add_command(label="Product") 

        self.menubar.add_cascade(label="Edit", menu=self.edit_menu)

        # Advance
        self.advance_menu = Menu(self.menubar, tearoff=0)
        self.advance_menu.add_command(label="turn to HSV", command=self.show_hist)
        self.advance_menu.add_command(label="Pixel Region Info")

        self.menubar.add_cascade(label="Advance", menu=self.advance_menu)

        self.frame = Frame(
            self.root, 
            height=self.root.winfo_screenheight() / 8,
            width=self.root.winfo_screenwidth(),
        )
        self.frame.pack()
        Label(self.frame, text="Welcome to ATI Program").pack()
        self.root.config(menu=self.menubar)

        self.root.pack_propagate(0)

        self.windows = []
        self.result_img = None

        self.root.mainloop()


    # Open file window
    def open_file_window(self):
        # See if windows were closed and erase from list
        self.check_windows()

        window = secondWindow(len(self.windows), self.root)

        self.windows.append(window)

        filename = filedialog.askopenfilename(initialdir="./photos", title="Select an Image" , filetypes=(("raw", "*.RAW"), ("pgm", "*.pgm"), ("ppm", "*.ppm"), ("jpg", "*.jpg"), ("png", "*.png")))
        
        if filename.lower().endswith('.raw'):

            Label(window.frame, text="Enter image width:").grid(row=0, column=0)   
            self.entry1 = Entry(window.frame, width=10)
            self.entry1.grid(row=1, column=0)

            Label(window.frame, text="Enter image height:").grid(row=0, column=1)   
            self.entry2 = Entry(window.frame, width=10)
            self.entry2.grid(row=1, column=1)

            button = Button(window.frame, text="Enter", command= (lambda : self.load_raw_image(filename, window)), padx=20)
            button.grid(row=1, column=2)    
        else:
            window.img = load(filename)

            window.tk_img = ImageTk.PhotoImage(image=Image.fromarray(window.img))
            
            Label(window.frame, image=window.tk_img).pack()


    # Load a RAW Image
    def load_raw_image(self, filename, window):
        window.img = load(filename, int(self.entry1.get()), int(self.entry2.get()))

        window.frame.pack_forget()
        window.frame = Frame(window.top)
        window.frame.pack()

        window.tk_img = ImageTk.PhotoImage(image=Image.fromarray(window.img))
        Label(window.frame, image=window.tk_img).pack()
    

    # Save Image
    def save_image_window(self):
        if self.result_img is None:

            top = Toplevel()
            Label(top, text="No photo to save").grid(row=0, column=0, columnspan=3)
            Button(top, text= "Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            filename = filedialog.asksaveasfilename(initialdir="./photos", title="Save As")
            
            save(self.result_img, filename)

            top = Toplevel()
            Label(top, text="Photo saved successfully").grid(row=0, column=0, columnspan=3)
            Button(top, text= "Done", command=top.destroy, padx=20).grid(row=2, column=1)


    # Create Image with white Circle in the middle
    def create_circle(self):
        self.result_img = np.asarray(bin_circle())

        # See if windows were closed and erase from list
        self.check_windows()

        window = secondWindow(len(self.windows), self.root)
        self.windows.append(window)

        window.img = self.result_img
        window.tk_img = ImageTk.PhotoImage(image=Image.fromarray(window.img))
        Label(window.frame, image=window.tk_img).pack()

    
     # Create Image with white Square in the middle
    def create_square(self): 
        self.result_img = np.asarray(bin_rectangle())
   
        # See if windows were closed and erase from list
        self.check_windows()

        window = secondWindow(len(self.windows), self.root)
        self.windows.append(window)

        window.img = self.result_img
        window.tk_img = ImageTk.PhotoImage(image=Image.fromarray(window.img))
        Label(window.frame, image=window.tk_img).pack()
        

    # Get pixel value
    def get_pixel_value(self):  
        if len(self.windows) == 0:

            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text= "Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            top = Toplevel()
            frame2 = Frame(top)
            frame2.pack()

            if len(self.windows) > 1:
                Label(frame2, text = "Select image").grid(row=0,column=0,columnspan=3)
                
                self.check_windows()

                clicked = StringVar()
                options = self.get_windows_titles()

                clicked.set(options[0])

                op_menu = OptionMenu(frame2, clicked, *options)
                op_menu.grid(row=1, column=0, columnspan=2)

                button = Button(frame2, text="Select", command=lambda: self.get_pixel_coordinates(frame2, top, clicked.get()))
                button.grid(row=1, column=2)
            else:
                self.get_pixel_coordinates(frame2, top, self.windows[0].title)


    def get_pixel_coordinates(self, frame2, top, clicked):
        frame2.pack_forget()
        frame2 = Frame(top)
        frame2.pack()    

        Label(frame2, text = "Enter (x,y) coordinates of pixel").grid(row=0,column=0,columnspan=3)

        Label(frame2, text="Enter x:").grid(row=1, column=0)   
        self.entry1 = Entry(frame2, width=10)
        self.entry1.grid(row=2, column=0)

        Label(frame2, text="Enter y:").grid(row=1, column=1)   
        self.entry2 = Entry(frame2, width=10)
        self.entry2.grid(row=2, column=1)

        window = self.get_clicked_window(clicked)

        button = Button(frame2, text="Enter", command= (lambda : self.show_pixel_value(top, frame2, window)), padx=20)
        button.grid(row=2, column=2)
    
            
    def show_pixel_value(self, top, frame, window):
        x = int(self.entry1.get())
        y = int(self.entry2.get())

        frame.pack_forget()
        frame = Frame(top)
        frame.pack()

        value = get_pixel(window.img, x, y)

        Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(value)).grid(row=0, column=0, columnspan=3)
        Button(frame, text= "Done", command=top.destroy, padx=20).grid(row=2, column=1)


    # Modify pixel
    def modify_pixel(self):
    
        if len(self.windows) == 0:

            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text= "Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            top = Toplevel()
            frame2 = Frame(top)
            frame2.pack()

            if len(self.windows) > 1:
                Label(frame2, text = "Select image").grid(row=0,column=0,columnspan=3)
                
                self.check_windows()

                clicked = StringVar()
                options = self.get_windows_titles()

                clicked.set(options[0])

                op_menu = OptionMenu(frame2, clicked, *options)
                op_menu.grid(row=1, column=0, columnspan=2)

                button = Button(frame2, text="Select", command=lambda: self.get_coordinates_modify(frame2, top, clicked.get()))
                button.grid(row=1, column=2)
            else:
                self.get_coordinates_modify(frame2, top, self.windows[0].title)


    def get_coordinates_modify(self, frame2, top, clicked):
        frame2.pack_forget()
        frame2 = Frame(top)
        frame2.pack()

        Label(frame2, text = "Enter (x,y) coordinates of pixel").grid(row=0,column=0,columnspan=3)

        Label(frame2, text="Enter x:").grid(row=1, column=0)   
        self.entry1 = Entry(frame2, width=10)
        self.entry1.grid(row=2, column=0)

        Label(frame2, text="Enter y:").grid(row=1, column=1)   
        self.entry2 = Entry(frame2, width=10)
        self.entry2.grid(row=2, column=1)

        Label(frame2, text="Enter new value:").grid(row=3, column=0, columnspan=2)   
        self.entry3 = Entry(frame2, width=20)
        self.entry3.grid(row=4, column=0, columnspan=2)

        window = self.get_clicked_window(clicked)

        button = Button(frame2, text="Enter", command= (lambda : self.show_modified_pixel(top, frame2, window)), padx=20)
        button.grid(row=4, column=2)
    

    def show_modified_pixel(self, top, frame, window):
        x = int(self.entry1.get())
        y = int(self.entry2.get())
        new_value = int(self.entry3.get())

        frame.pack_forget()
        frame = Frame(top)
        frame.pack()        

        self.check_windows()

        new_window = secondWindow(len(self.windows), self.root)
        self.windows.append(new_window)

        new_window.copy_img(window.img)

        put_pixel(new_window.img, x, y, new_value)

        Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(new_value)).grid(row=0, column=0, columnspan=3)
        Button(frame, text= "Done", command=lambda: self.show_new_pixel(top, new_window), padx=20).grid(row=2, column=1)


    def show_new_pixel(self, top, window):
        top.destroy()
        
        window.tk_img = ImageTk.PhotoImage(image=Image.fromarray(window.img))
        Label(window.frame, image=window.tk_img).pack()
        

    def copy_img_into_other(self):
        if len(self.windows) < 1:
            top = Toplevel()
            Label(top, text="Not enough images, please load at least one image").grid(row=0, column=0, columnspan=3)
            Button(top, text= "Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            top = Toplevel()
            frame2 = Frame(top)
            frame2.pack()

            Label(frame2, text = "Select images").grid(row=0,column=0,columnspan=3)
            
            self.check_windows()

            clicked1 = StringVar()
            clicked2 = StringVar()
            options = self.get_windows_titles()

            clicked1.set(options[0])
            clicked2.set(options[0])

            Label(frame2, text = "Select image from").grid(row=1,column=0,columnspan=2)
            op_menu1 = OptionMenu(frame2, clicked1, *options)
            op_menu1.grid(row=2, column=0, columnspan=2)

            Label(frame2, text = "Select image to").grid(row=3,column=0,columnspan=2)
            op_menu2 = OptionMenu(frame2, clicked2, *options)
            op_menu2.grid(row=4, column=0, columnspan=2)

            button = Button(frame2, text="Select", command=lambda: self.exchange_image(top, frame2, [clicked1.get(), clicked2.get()]))
            button.grid(row=4, column=2)
            
      
    def exchange_image(self, top, frame2, windows):
        frame2.pack_forget()
        frame2 = Frame(top)
        frame2.pack()

        Label(frame2, text = "Enter (x1,y1) and (x2, y2) coordinates of pixel of from image").grid(row=0,column=0,columnspan=3)

        Label(frame2, text="Enter x1:").grid(row=1, column=0)   
        self.entry1 = Entry(frame2, width=10)
        self.entry1.grid(row=2, column=0)

        Label(frame2, text="Enter y1:").grid(row=1, column=1)   
        self.entry2 = Entry(frame2, width=10)
        self.entry2.grid(row=2, column=1)

        Label(frame2, text="Enter x2:").grid(row=1, column=0)   
        self.entry3 = Entry(frame2, width=10)
        self.entry3.grid(row=3, column=0)

        Label(frame2, text="Enter y2:").grid(row=1, column=1)   
        self.entry4 = Entry(frame2, width=10)
        self.entry4.grid(row=3, column=1)

        Label(frame2, text = "Enter (x,y) coordinates of left corner pixel of to image").grid(row=4,column=0,columnspan=3)

        Label(frame2, text="Enter x:").grid(row=1, column=0)   
        self.entry5 = Entry(frame2, width=10)
        self.entry5.grid(row=5, column=0)

        Label(frame2, text="Enter y:").grid(row=1, column=1)   
        self.entry6 = Entry(frame2, width=10)
        self.entry6.grid(row=5, column=1)

        button = Button(frame2, text="Enter", command= (lambda : self.change_image(top, windows)), padx=20)
        button.grid(row=5, column=2)


    def change_image(self, top, windows):
        x1 = int(self.entry1.get())
        y1 = int(self.entry2.get())
        x2 = int(self.entry3.get())
        y2 = int(self.entry4.get())
        x = int(self.entry5.get())
        y = int(self.entry6.get())

        self.check_windows()
        new_window = secondWindow(len(self.windows), self.root)
        self.windows.append(new_window)

        from_window = self.get_clicked_window(windows[0])
        to_window = self.get_clicked_window(windows[1])

        new_window.img = paste_section(from_window.img, x1, y1, x2, y2, to_window.img, x, y)

        top.destroy()

        new_window.tk_img = ImageTk.PhotoImage(image=Image.fromarray(new_window.img))
        Label(new_window.top, image=new_window.tk_img).pack()


    def show_hist(self):
        if len(self.windows) == 0:
            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text= "Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            top = Toplevel()
            frame2 = Frame(top)
            frame2.pack()

            if len(self.windows) > 1:
                Label(frame2, text = "Select image").grid(row=0,column=0,columnspan=3)
                
                self.check_windows()

                clicked = StringVar()
                options = self.get_windows_titles()

                clicked.set(options[0])

                op_menu = OptionMenu(frame2, clicked, *options)
                op_menu.grid(row=1, column=0, columnspan=2)

                button = Button(frame2, text="Select", command=lambda: self.show_plot_hist(top, clicked.get()))
                button.grid(row=1, column=2)
            else:
                self.show_plot_hist(top, self.windows[0].title)


    def show_plot_hist(self, top, clicked):
        top.destroy()
        window = self.get_clicked_window(clicked)

        plot_hist(window.img)

        
    def ask_quit(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.root.destroy()
        
    
    def check_windows(self):
        for window in self.windows:
            if not Toplevel.winfo_exists(window.top):
                self.windows.remove(window)

        for i in range(0, len(self.windows)):
            window = self.windows[i]
            window.modify_id(i)


    def get_windows_titles(self):
        titles = []
        for window in self.windows:
            titles.append(window.title)
        return titles
    

    def get_clicked_window(self, clicked):
        for window in self.windows:
            if window.title == clicked:
                return window
        

if __name__ == '__main__':
    x=mainWindow()