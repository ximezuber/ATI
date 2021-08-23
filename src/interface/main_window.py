from src.interface.image_window import ImageWindow
from tkinter import *
from src.image_utils import *
from numpy import copy
from tkinter import filedialog, messagebox


class MainWindow:

    def __init__(self):
        self.root = Tk()
        self.root.title("ATI")

        # Menu 
        self.menubar = Menu(self.root)
        # Format of option list:
        #   (Label, function on click)
        #   None = separator
        image_menu_options = [('Load', self.open_file_window),
                              ('Save', self.save_image_window),
                              None,
                              ('Create Circle Image', self.create_circle),
                              ('Create Square Image', self.create_square),
                              None,
                              ('Exit', self.ask_quit)]
        edit_menu_options = [('Select', self.select),
                             None,
                             ('Get Pixel', self.get_pixel_value),
                             ('Modify Pixel', self.modify_pixel),
                             None,
                             ('Add', None),  # TODO
                             ('Subtract', None),  # TODO
                             ('Multiply', None)]  # TODO
        advanced_menu_options = [('Turn to HSV', self.show_hist)]

        menu_options = {'Image': image_menu_options,
                        'Edit': edit_menu_options,
                        'Advanced': advanced_menu_options}

        for option in menu_options.keys():
            self._add_to_menu(option, menu_options[option])

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
        self.result_img = {}

        self.root.mainloop()

    # Open file window
    def open_file_window(self):
        filename = filedialog.askopenfilename(initialdir="./photos", title="Select an Image",
                                              filetypes=(("raw", "*.RAW"),
                                                         ("pgm", "*.pgm"),
                                                         ("ppm", "*.ppm"),
                                                         ("jpg", "*.jpg"),
                                                         ("png", "*.png")))
        if not filename:
            return

        w, h = None, None
        if filename.lower().endswith('.raw'):
            w, h = self.ask_width_and_height()
        ImageWindow(self, load(filename, w, h))

    # Save Image
    def save_image_window(self):
        if len(self.result_img) == 0:
            top = Toplevel()
            Label(top, text="No photo to save").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            filename = filedialog.asksaveasfilename(initialdir="./photos", title="Save As")
            if not filename:
                return
            save(img, filename)

            top = Toplevel()
            Label(top, text="Photo saved successfully").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)

    # Create Image with white Circle in the middle
    def create_circle(self):
        window = ImageWindow(self, np.asarray(bin_circle()))
        self.result_img[window.title] = window.img

    # Create Image with white Square in the middle
    def create_square(self):
        window = ImageWindow(self, np.asarray(bin_rectangle()))
        self.result_img[window.title] = window.img

    # Get pixel value
    def get_pixel_value(self):
        if len(self.windows) == 0:
            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            x, y = self.ask_xy('Enter (x,y) coordinates of pixel')
            value = get_pixel(img, x, y)

            window = Toplevel()
            frame = Frame(window)
            frame.pack()
            Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(value)) \
                .grid(row=0, column=0, columnspan=3)

            Button(frame, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)

    # Modify pixel TODO: Support RGB values in colored images
    def modify_pixel(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = copy(self.select_img_from_windows())
            x, y, new_value = self.ask_xy_new('Enter (x,y) coordinates of pixel')
            put_pixel(img, x, y, new_value)
            ImageWindow(self, img)
            info_window = Toplevel()
            frame = Frame(info_window)
            frame.pack()
            Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(new_value))\
                .grid(row=0, column=0, columnspan=3)
            Button(frame, text="Done", command=info_window.destroy, padx=20).grid(row=2, column=1)

    # Exit
    def ask_quit(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.root.destroy()

    # Show RGB histograms
    def show_hist(self):
        if len(self.windows) == 0:
            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            plot_hist(img)

    # Selection mode
    def select(self):
        if len(self.windows) == 0:
            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            image_window = self.select_window()
            image_window.selection_mode()

    # AUXILIARY FUNCTIONS
    def _add_to_menu(self, label, image_menu_options):
        menu = Menu(self.menubar, tearoff=0)
        for option in image_menu_options:
            if option is None:
                menu.add_separator()
            else:
                menu.add_command(label=option[0], command=option[1])
        self.menubar.add_cascade(label=label, menu=menu)

    def select_img_from_windows(self):
        window = self.select_window()
        return window.img

    def select_window(self):
        if len(self.windows) == 1:
            return self.windows[0]
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Select image").grid(row=0, column=0, columnspan=3)

        clicked = StringVar()
        options = self.get_windows_titles()

        clicked.set(options[0])

        op_menu = OptionMenu(frame, clicked, *options)
        op_menu.grid(row=1, column=0, columnspan=2)

        window_name_var = StringVar()
        Button(frame, text="Select", command=(lambda: window_name_var.set(clicked.get()))).grid(row=1, column=2)
        frame.wait_variable(window_name_var)
        window_name = window_name_var.get()
        window.destroy()
        for image_window in self.windows:
            if image_window.title == window_name:
                return image_window

    def get_windows_titles(self):
        titles = []
        for window in self.windows:
            titles.append(window.title)
        return titles

    # User input windows
    @staticmethod
    def ask_xy(window_title):
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text=window_title).grid(row=0, column=0, columnspan=3)

        Label(frame, text="Enter x:").grid(row=1, column=0)
        x_entry = Entry(frame, width=10)
        x_entry.grid(row=2, column=0)

        Label(frame, text="Enter y:").grid(row=1, column=1)
        y_entry = Entry(frame, width=10)
        y_entry.grid(row=2, column=1)

        x_var = IntVar()
        y_var = IntVar()

        button = Button(frame, text="Enter",
                        command=(lambda: (x_var.set(int(x_entry.get())), y_var.set(int(y_entry.get())))),
                        padx=20)
        button.grid(row=2, column=2)

        frame.wait_variable(y_var)
        x, y = x_var.get(), y_var.get()
        window.destroy()
        return x, y

    @staticmethod
    def ask_xy_new(window_title):
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text=window_title).grid(row=0, column=0, columnspan=3)

        Label(frame, text="Enter x:").grid(row=1, column=0)
        x_entry = Entry(frame, width=10)
        x_entry.grid(row=2, column=0)

        Label(frame, text="Enter y:").grid(row=1, column=1)
        y_entry = Entry(frame, width=10)
        y_entry.grid(row=2, column=1)

        Label(frame, text="Enter new value:").grid(row=3, column=0, columnspan=2)
        new_entry = Entry(frame, width=20)
        new_entry.grid(row=4, column=0, columnspan=2)

        x_var = IntVar()
        y_var = IntVar()
        new_var = IntVar()

        button = Button(frame, text="Enter",
                        command=(lambda: (x_var.set(int(x_entry.get())),
                                          y_var.set(int(y_entry.get())),
                                          new_var.set(int(new_entry.get())))),
                        padx=20)
        button.grid(row=4, column=2)

        frame.wait_variable(new_var)
        x, y, new = x_var.get(), y_var.get(), new_var.get()
        window.destroy()
        return x, y, new

    @staticmethod
    def ask_width_and_height():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter image width:").grid(row=0, column=0)
        width_entry = Entry(frame, width=10)
        width_entry.grid(row=1, column=0)

        Label(frame, text="Enter image height:").grid(row=0, column=1)
        height_entry = Entry(frame, width=10)
        height_entry.grid(row=1, column=1)

        width_var = IntVar()
        height_var = IntVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (height_var.set(int(height_entry.get())), width_var.set(int(height_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(width_var)
        w, h = width_var.get(), height_var.get()
        window.destroy()
        return w, h



