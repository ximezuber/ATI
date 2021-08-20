from tkinter import *
from image_utils import *
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from image_window import ImageWindow


class MainWindow:
    def __init__(self):
        self.root = Tk()
        self.root.title("ATI")
        self.windows = {}

        # Menu
        self.menubar = Menu(self.root)

        # Format of option list:
        #   (Label, function on click)
        #   None = separator
        image_menu_options = [('Load', self.open_file_window),
                              None,
                              ('Create Circle Image', self.create_circle),
                              ('Create Square Image', self.create_square),
                              None,
                              ('Exit', self.ask_quit)]

        edit_menu_options = [('Copy into Other', None),
                             None,
                             ('Add', None),
                             ('Subtract', None),
                             ('Product', None)]

        advanced_menu_options = [('Turn to HSV', self.show_hist),
                                 ('Pixel Region Info', None)]

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

        self.img = None

        self.root.mainloop()

    def _add_to_menu(self, label, image_menu_options):
        menu = Menu(self.menubar, tearoff=0)
        for option in image_menu_options:
            if option is None:
                menu.add_separator()
            else:
                menu.add_command(label=option[0], command=option[1])
        self.menubar.add_cascade(label=label, menu=menu)

    # Open file window
    def open_file_window(self):
        filename = filedialog.askopenfilename(initialdir="./photos", title="Select an Image", filetypes=(
            ("raw", "*.RAW"), ("pgm", "*.pgm"), ("ppm", "*.ppm"), ("jpg", "*.jpg"), ("png", "*.png")))
        if not filename:
            return
        if filename.lower().endswith('.raw'):
            new_window = Toplevel()
            frame = Frame(new_window)
            frame.pack()
            new_window.pack_propagate(1)
            Label(frame, text='Open ' + filename).grid(row=0, columnspan=2)

            Label(frame, text="Enter image width:").grid(row=1, column=0)
            width = Entry(frame, width=10)
            width.grid(row=2, column=0)

            Label(frame, text="Enter image height:").grid(row=1, column=1)
            height = Entry(frame, width=10)
            height.grid(row=2, column=1)

            button = Button(frame, text="Enter", command=(lambda: self._on_click_enter_dimensions(
                filename, int(width.get()), int(height.get()), new_window)))
            button.grid(row=3, columnspan=2)

        else:
            image_window = ImageWindow(self)
            image_window.add_image(filename)

    def _on_click_enter_dimensions(self, filename, width, height, dimensions_window):
        image_window = ImageWindow(self)
        image_window.add_image(filename, width, height)
        dimensions_window.destroy()

    # Create Image with white Circle in the middle
    def create_circle(self):
        new_window = ImageWindow(self)
        img = np.asarray(bin_circle())
        new_window.add_image_from_array(img, 'circle')

    # Create Image with white Square in the middle
    def create_square(self):
        new_window = ImageWindow(self)
        img = np.asarray(bin_rectangle())
        new_window.add_image_from_array(img, 'square')

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
    MainWindow()
