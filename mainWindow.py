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
        self.edit_menu.add_command(label="Copy into Other")   
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

        # # Drop down menu
        # self.options = OptionMenu(self.root, self.clicked, *self.commands)
        # self.options.grid(row=0, column=0, columnspan=3, sticky="ew")

        # # Button to select option from drop down menu
        # self.select_button = Button(self.root, text="Select", command=self.show)
        # self.select_button.grid(row=0, column=3, sticky="ew")

        # # Entry n1
        # self.entry = Entry(self.root, width=40)
        # self.entry.grid(row=1, column=0, columnspan=2)

        # # Entry n2
        # self.entry2 = Entry(self.root, width=40)

        # # Button to run option
        # self.run_button = Button(self.root, text=self.clicked.get(), command=self.open_image)
        # self.run_button.grid(row=1,column=3)

        # # Button to search entry n1
        # self.search_button = Button(self.root, text= "Search", command=open_file_window)
        # self.search_button.grid(row=1, column=2)

        # # Button to search entry n2
        # self.search_button2 = Button(self.root, text= "Search")
        self.frame = Frame(
            self.root, 
            height=self.root.winfo_screenheight() / 8,
            width=self.root.winfo_screenwidth(),
        )
        self.frame.pack()
        Label(self.frame, text="Welcome to ATI Program").pack()
        self.root.config(menu=self.menubar)

        self.root.pack_propagate(0)

        self.img = None

        self.root.mainloop()


    # Open file window
    def open_file_window(self):
        global tk_img

        self.frame.pack_forget()
        self.frame = Frame(self.root)
        self.frame.pack()

        self.root.pack_propagate(1)

        filename = filedialog.askopenfilename(initialdir="./photos", title="Select an Image" , filetypes=(("raw", "*.RAW"), ("pgm", "*.pgm"), ("ppm", "*.ppm"), ("jpg", "*.jpg"), ("png", "*.png")))
        
        if filename.lower().endswith('.raw'):

            Label(self.frame, text="Enter image width:").grid(row=0, column=0)   
            self.entry1 = Entry(self.frame, width=10)
            self.entry1.grid(row=1, column=0)

            Label(self.frame, text="Enter image height:").grid(row=0, column=1)   
            self.entry2 = Entry(self.frame, width=10)
            self.entry2.grid(row=1, column=1)

            button = Button(self.frame, text="Enter", command= (lambda : self.load_raw_image(filename)), padx=20)
            button.grid(row=1, column=2)
            
        else:
            self.img = load(filename)

            tk_img = ImageTk.PhotoImage(image=Image.fromarray(self.img))
            Label(self.frame, image=tk_img).pack()


    # Load a RAW Image
    def load_raw_image(self, filename):
        global tk_img

        self.frame.pack_forget()
        self.frame = Frame(self.root)
        self.frame.pack()

        self.img = load(filename, int(self.entry1.get()), int(self.entry2.get()))

        self.root.pack_propagate(1)

        tk_img = ImageTk.PhotoImage(image=Image.fromarray(self.img))
        Label(self.frame, image=tk_img).pack()
    

    # Save Image
    def save_image_window(self):
        
        if self.img is None:
            self.frame.pack_forget()
            self.frame = Frame(self.root)
            self.frame.pack()

            Label(self.frame, text="No Image to Save").pack()
        else:
            filename = filedialog.asksaveasfilename(initialdir="./photos", title="Save As")
            
            save(self.img, filename)

            top = Toplevel()
            Label(top, text="File saves successfully").grid(row=0, column=0, columnspan=3)
            Button(top, text= "Done", command=top.destroy, padx=20).grid(row=2, column=1)


    # Create Image with white Circle in the middle
    def create_circle(self):
        global tk_img

        self.frame.pack_forget()
        self.frame = Frame(self.root)
        self.frame.pack()

        self.img = np.asarray(bin_circle())

        self.root.pack_propagate(1)

        tk_img = ImageTk.PhotoImage(image=Image.fromarray(self.img))
        Label(self.frame, image=tk_img).pack()

    
     # Create Image with white Square in the middle
    def create_square(self):
        global tk_img

        self.frame.pack_forget()
        self.frame = Frame(self.root)
        self.frame.pack()

        self.img = np.asarray(bin_rectangle())
        
        self.root.pack_propagate(1)

        tk_img = ImageTk.PhotoImage(image=Image.fromarray(self.img))
        Label(self.frame, image=tk_img).pack()
        

    # Get pixel value
    def get_pixel_value(self):
        
        if self.img is None:
            self.frame.pack_forget()
            self.frame = Frame(self.root)
            self.frame.pack()

            Label(self.frame, text="No Image, please load one").pack()
        else:
            top = Toplevel()
            frame2 = Frame(top)
            frame2.pack()
            Label(frame2, text = "Enter (x,y) coordinates of pixel").grid(row=0,column=0,columnspan=3)

            Label(frame2, text="Enter x:").grid(row=1, column=0)   
            self.entry1 = Entry(frame2, width=10)
            self.entry1.grid(row=2, column=0)

            Label(frame2, text="Enter y:").grid(row=1, column=1)   
            self.entry2 = Entry(frame2, width=10)
            self.entry2.grid(row=2, column=1)

            button = Button(frame2, text="Enter", command= (lambda : self.show_pixel_value(top, frame2)), padx=20)
            button.grid(row=2, column=2)

            
    def show_pixel_value(self, top, frame):
        x = int(self.entry1.get())
        y = int(self.entry2.get())

        frame.pack_forget()
        frame = Frame(top)
        frame.pack()

        value = get_pixel(self.img, x, y)

        Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(value)).grid(row=0, column=0, columnspan=3)
        Button(frame, text= "Done", command=top.destroy, padx=20).grid(row=2, column=1)


    # Modify pixel
    def modify_pixel(self):
    
        if self.img is None:
            self.frame.pack_forget()
            self.frame = Frame(self.root)
            self.frame.pack()

            Label(self.frame, text="No Image, please load one").pack()
        else:
            top = Toplevel()
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

            button = Button(frame2, text="Enter", command= (lambda : self.show_modified_pixel(top, frame2)), padx=20)
            button.grid(row=4, column=2)
    

    def show_modified_pixel(self, top, frame):
            
        x = int(self.entry1.get())
        y = int(self.entry2.get())
        new_value = int(self.entry3.get())

        frame.pack_forget()
        frame = Frame(top)
        frame.pack()

        put_pixel(self.img, x, y, new_value)

        Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(new_value)).grid(row=0, column=0, columnspan=3)
        Button(frame, text= "Done", command=top.destroy, padx=20).grid(row=2, column=1)

        
    def copy_img_into_other(self):
        if self.img is None:
            self.frame.pack_forget()
            self.frame = Frame(self.root)
            self.frame.pack()

            Label(self.frame, text="No Image, please load one").pack()
        else:
            self.img_old = self.img

            global tk_img

        self.frame.pack_forget()
        self.frame = Frame(self.root)
        self.frame.pack()

        self.root.pack_propagate(1)

        filename = filedialog.askopenfilename(initialdir="./photos", title="Select an Image" , filetypes=(("raw", "*.RAW"), ("pgm", "*.pgm"), ("ppm", "*.ppm"), ("jpg", "*.jpg"), ("png", "*.png")))
        
        if filename.lower().endswith('.raw'):

            Label(self.frame, text="Enter image width:").grid(row=0, column=0)   
            self.entry1 = Entry(self.frame, width=10)
            self.entry1.grid(row=1, column=0)

            Label(self.frame, text="Enter image height:").grid(row=0, column=1)   
            self.entry2 = Entry(self.frame, width=10)
            self.entry2.grid(row=1, column=1)

            button = Button(self.frame, text="Enter", command= (lambda : self.exchange_image(self.load_raw_image(filename))), padx=20)
            button.grid(row=1, column=2)
            
        else:
            self.img = load(filename)
            self.exchange_image(self.img)

    def exchange_image(self, img=None):
        global tk_img
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(self.img))
        Label(self.frame, image=tk_img).pack()

        top = Toplevel()
        frame2 = Frame(top)
        frame2.pack()
        Label(frame2, text = "Enter (x1,y1) and (x2, y2) coordinates of pixel").grid(row=0,column=0,columnspan=3)

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

        button = Button(frame2, text="Enter", command= (lambda : self.change_image(top, frame2)), padx=20)
        button.grid(row=3, column=2)

    def change_image(self, top, frame):
        top.destroy()

        x1 = int(self.entry1.get())
        y1 = int(self.entry2.get())
        x2 = int(self.entry3.get())
        y2 = int(self.entry4.get())

        self.img = paste_section(self.img_old, x1, y1, x2, y2, self.img, x1, y1)

        self.frame.pack_forget()
        self.frame = Frame(self.root)
        self.frame.pack()

        self.root.pack_propagate(1)

        tk_img = ImageTk.PhotoImage(image=Image.fromarray(self.img))
        Label(self.frame, image=tk_img).pack()


    def show_hist(self):
        if self.img is None:
            self.frame.pack_forget()
            self.frame = Frame(self.root)
            self.frame.pack()

            Label(self.frame, text="No Image to Save").pack()
        else:
            plot_hist(self.img)

        

    def ask_quit(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.root.destroy()
        
        

if __name__ == '__main__':
    x=mainWindow()