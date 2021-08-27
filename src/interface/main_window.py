from src.utils.mask_utils import *
from src.utils.noise_utils import *
from src.interface.image_window import ImageWindow
from tkinter import *
from src.utils.image_utils import *
from numpy import copy
from tkinter import filedialog, messagebox
from src.utils.random_generator import (GaussianGenerator, 
                                        RayleighGenerator, 
                                        ExponentialGenerator)


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
                             ('Add', self.sum_images),  
                             ('Subtract', self.subtract_images),  
                             ('Multiply', self.multiply_images),
                             None, 
                             ('Negative', self.image_negative), 
                             ("Thresholding", self.image_thresholding),
                             ('Power function', self.power_gamma),
                             ('Equalize', self.equalize)]
        advanced_menu_options = [('Show histogram', self.show_hist),
                                 ('Show HSV histogram', self.show_hsv_hist)]
        noise_menu_options = [('Show Generator Histograms', self.show_gen_hist),
                              None,
                              ("Add Gaussian Noise", self.add_gaussian_noise),
                              ("Add Rayleigh Noise", self.add_rayleigh_noise),
                              ("Add Exponential Noise", self.add_exponential_noise),
                              ("Add Salt and Pepper Noise", self.add_salt_peppper_noise)]

        filter_menu_options = [('Mean filter', self.mean_filter),
                               ('Median filter', self.median_filter),
                               ('Gaussian mean filter', self.gaussian_filter),
                               ('Weighted median filter', self.weighted_median_filter),
                               ('Border filter', self.border_filter)]

        menu_options = {'Image': image_menu_options,
                        'Edit': edit_menu_options,
                        'Advanced': advanced_menu_options,
                        "Noise": noise_menu_options,
                        "Filter": filter_menu_options}

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
        self.unsaved_imgs = {}

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
        if len(self.unsaved_imgs) == 0:
            top = Toplevel()
            Label(top, text="No photo to save").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            (title, img) = self.select_save_window()
            filename = filedialog.asksaveasfilename(initialdir="./photos", title="Save As")
            if not filename:
                return
            save(img, filename)

            self.unsaved_imgs.pop(title)

            top = Toplevel()
            Label(top, text="Photo saved successfully").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)

    # Create Image with white Circle in the middle
    def create_circle(self):
        window = ImageWindow(self, np.asarray(bin_circle()))
        self.unsaved_imgs[window.title] = window.img

    # Create Image with white Square in the middle
    def create_square(self):
        window = ImageWindow(self, np.asarray(bin_rectangle()))
        self.unsaved_imgs[window.title] = window.img

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
            window = ImageWindow(self, img)
            info_window = Toplevel()
            frame = Frame(info_window)
            frame.pack()
            Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(new_value))\
                .grid(row=0, column=0, columnspan=3)
            Button(frame, text="Done", command=info_window.destroy, padx=20).grid(row=2, column=1)
            self.unsaved_imgs[window.title] = window.img

    # Add two images
    def sum_images(self): 
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else: 
            windows = self.select_from_to_windows()
            new_img = add(windows[0].img, windows[1].img)
            img_window = ImageWindow(self, new_img)
            self.unsaved_imgs[img_window.title] = img_window.img
    
    # Subtract two images
    def subtract_images(self): 
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else: 
            windows = self.select_from_to_windows()
            new_img = subtract(windows[0].img, windows[1].img)
            img_window = ImageWindow(self, new_img)
            self.unsaved_imgs[img_window.title] = img_window.img

    # Multiply two images
    def multiply_images(self): 
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else: 
            windows = self.select_from_to_windows()
            new_img = multiply(windows[0].img, windows[1].img)
            img_window = ImageWindow(self, new_img)
            self.unsaved_imgs[img_window.title] = img_window.img
            
    # Get image's negaive        
    def image_negative(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = negative(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    
    def image_thresholding(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            new_img = thresholding(threshold, img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def power_gamma(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            gamma = self.ask_for_gamma()
            if (0 < gamma < 2) and gamma != 1:
                new_img = power(img, gamma)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                    title="Error", message="Gamma should be between 0 and 2, and different from 1."
                )

    def equalize(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = equalize(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    # Exit
    def ask_quit(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.root.destroy()

    # Show RGB and gray histograms
    def show_hist(self):
        if len(self.windows) == 0:
            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            plot_hist_rgb(img)

    # Show HSV histograms
    def show_hsv_hist(self):
        if len(self.windows) == 0:
            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            converted = rgb_to_hsv(img)
            plot_hist_hsv(converted)

    # Shows histogram for a random number generator
    def show_gen_hist(self):
        dist = self.ask_distribution()
        if dist == "Gaussian":
            mean, std = self.ask_gaussian_args()
            plot_gen_hist(GaussianGenerator(mean, std, 1000), dist)
        elif dist == "Rayleigh":
            phi = self.ask_rayleigh_args()
            plot_gen_hist(RayleighGenerator(phi, 1000), dist)
        elif dist == "Exponential":
            lam = self.ask_exponential_args()
            plot_gen_hist(ExponentialGenerator(lam, 1000), dist)
    
    # Add Gaussian Noise to image
    def add_gaussian_noise(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else: 
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            if 0.0 <= threshold <= 1.0:
                mean, std = self.ask_gaussian_args()

                size = img.size

                new_img = apply_noise(img, GaussianGenerator(mean, std, size), threshold, True)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                title="Error", message="Threshold should be between 0 and 1."
                )

    # Add Rayleigh Noise to image
    def add_rayleigh_noise(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else: 
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            if 0.0 <= threshold <= 1.0:
                phi = self.ask_rayleigh_args()

                size = img.size

                new_img = apply_noise(img, RayleighGenerator(phi, size), threshold, False)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                title="Error", message="Threshold should be between 0 and 1."
                )

    # Add Exponential Noise to image
    def add_exponential_noise(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else: 
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            if 0.0 <= threshold <= 1.0:
                lam = self.ask_exponential_args()

                size = img.size

                new_img = apply_noise(img, ExponentialGenerator(lam, size), threshold, False)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                title="Error", message="Threshold should be between 0 and 1."
                )

    # Add Salt and Pepper Noise to image
    def add_salt_peppper_noise(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else: 
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            if 0.0 <= threshold <= 1.0:
                new_img = apply_salt_pepper_noise(img, threshold)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                title="Error", message="Threshold should be between 0 and 1."
                )

    def mean_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            mask_size = self.ask_for_mask_size()
            new_img = mean_filter(img, mask_size)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def median_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            mask_size = self.ask_for_mask_size()
            new_img = median_filter(img, mask_size)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def gaussian_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            mask_size, deviation = self.ask_for_mask_size_and_deviation()
            new_img = gaussian_filter(img, mask_size, deviation)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def weighted_median_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = weighted_median_filter(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def border_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            mask_size = self.ask_for_mask_size()
            new_img = border_filter(img, mask_size)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

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

    
    def select_save_window(self):
        if len(self.unsaved_imgs) == 1:
            return list(self.unsaved_imgs.items())[0]
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Select image").grid(row=0, column=0, columnspan=3)

        clicked = StringVar()
        options = list(self.unsaved_imgs.keys())

        clicked.set(options[0])

        op_menu = OptionMenu(frame, clicked, *options)
        op_menu.grid(row=1, column=0, columnspan=2)

        window_name_var = StringVar()
        Button(frame, text="Select", command=(lambda: window_name_var.set(clicked.get()))).grid(row=1, column=2)
        frame.wait_variable(window_name_var)
        window_name = window_name_var.get()
        window.destroy()
        for image_window in self.unsaved_imgs.keys():
            if image_window == window_name:
                return (image_window, self.unsaved_imgs.get(image_window))

    
    def select_from_to_windows(self):
        if len(self.windows) == 1:
            return (self.windows[0], self.windows[0])
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Select image 1").grid(row=0, column=0, columnspan=3)

        clicked_from = StringVar()
        options = self.get_windows_titles()

        clicked_from.set(options[0])

        op_menu_from = OptionMenu(frame, clicked_from, *options)
        op_menu_from.grid(row=1, column=0, columnspan=2)

        Label(frame, text="Select image 2").grid(row=2, column=0, columnspan=3)

        clicked_to = StringVar()
        options = self.get_windows_titles()

        clicked_to.set(options[0])

        op_menu_to = OptionMenu(frame, clicked_to, *options)
        op_menu_to.grid(row=3, column=0, columnspan=2)


        window_name_var_from = StringVar()
        window_name_var_to = StringVar()

        Button(frame, text="Select", command=(lambda: self.set_vars(window_name_var_from, 
                                                                    clicked_from.get(), 
                                                                    window_name_var_to, 
                                                                    clicked_to.get()))).grid(row=3, column=2)

        frame.wait_variable(window_name_var_to)
        window_from_name = window_name_var_from.get()
        window_to_name = window_name_var_to.get()
        window.destroy()

        window_from = None
        window_to = None
        for image_window in self.windows:
            if image_window.title == window_from_name:
                window_from = image_window
            if image_window.title == window_to_name:
                window_to = image_window
        return(window_from, window_to)

    
    def set_vars(self, from_var, from_val, to_var, to_val):
        from_var.set(from_val)
        to_var.set(to_val)


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
                            lambda: (height_var.set(int(height_entry.get())), width_var.set(int(width_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(width_var)
        w, h = width_var.get(), height_var.get()
        window.destroy()
        return w, h

    @staticmethod
    def ask_for_threshold():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text="Enter threshold:").grid(row=0, column=0)
        threshold_entry = Entry(frame, width=10)
        threshold_entry.grid(row=1, column=0)

        threshold_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (threshold_var.set(float(threshold_entry.get())))),
                        padx=20)
        button.grid(row=1, column=1)

        frame.wait_variable(threshold_var)
        threshold = threshold_var.get()
        window.destroy()
        return threshold

    @staticmethod
    def ask_for_gamma():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text="Enter gamma:").grid(row=0, column=0)
        gamma_entry = Entry(frame, width=10)
        gamma_entry.grid(row=1, column=0)

        gamma_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (gamma_var.set(float(gamma_entry.get())))),
                        padx=20)
        button.grid(row=1, column=1)

        frame.wait_variable(gamma_var)
        gamma = gamma_var.get()
        window.destroy()
        return gamma

    @staticmethod
    def ask_gaussian_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter mean:").grid(row=0, column=0)
        mean_entry = Entry(frame, width=10)
        mean_entry.grid(row=1, column=0)

        Label(frame, text="Enter standard deviation:").grid(row=0, column=1)
        std_entry = Entry(frame, width=10)
        std_entry.grid(row=1, column=1)

        mean_var = DoubleVar()
        std_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (mean_var.set(float(mean_entry.get())), std_var.set(float(std_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(std_var)
        m, s = mean_var.get(), std_var.get()
        window.destroy()
        return m, s

    @staticmethod
    def ask_rayleigh_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter phi:").grid(row=0, column=0)
        phi_entry = Entry(frame, width=10)
        phi_entry.grid(row=1, column=0)

        phi_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (phi_var.set(float(phi_entry.get())))),
                        padx=20)
        button.grid(row=1, column=1)

        frame.wait_variable(phi_var)
        phi = phi_var.get()
        window.destroy()
        return phi

    @staticmethod
    def ask_exponential_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter lambda:").grid(row=0, column=0)
        lambda_entry = Entry(frame, width=10)
        lambda_entry.grid(row=1, column=0)

        lambda_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (lambda_var.set(float(lambda_entry.get())))),
                        padx=20)
        button.grid(row=1, column=1)

        frame.wait_variable(lambda_var)
        lam = lambda_var.get()
        window.destroy()
        return lam

    @staticmethod
    def ask_distribution():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Select Distribution").grid(row=0, column=0, columnspan=3)

        clicked = StringVar()
        options = ["Gaussian", "Rayleigh", "Exponential"]

        clicked.set(options[0])

        op_menu = OptionMenu(frame, clicked, *options)
        op_menu.grid(row=1, column=0, columnspan=2)

        dist_var = StringVar()
        Button(frame, text="Select", command=(lambda: dist_var.set(clicked.get()))).grid(row=1, column=2)
        frame.wait_variable(dist_var)
        dist_val = dist_var.get()
        window.destroy()
        return dist_val

    @staticmethod
    def ask_for_mask_size():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter mask side:").grid(row=0, column=0)
        side_entry = Entry(frame, width=10)
        side_entry.grid(row=1, column=0)

        side_var = IntVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (side_var.set(int(side_entry.get())))),
                        padx=20)
        button.grid(row=1, column=1)

        frame.wait_variable(side_var)
        side = side_var.get()
        window.destroy()
        return side

    @staticmethod
    def ask_for_mask_size_and_deviation():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter mask_side:").grid(row=0, column=0)
        side_entry = Entry(frame, width=10)
        side_entry.grid(row=1, column=0)

        Label(frame, text="Enter standard deviation:").grid(row=0, column=1)
        std_entry = Entry(frame, width=10)
        std_entry.grid(row=1, column=1)

        side_var = IntVar()
        std_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (side_var.set(int(side_entry.get())), std_var.set(float(std_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(std_var)
        side, std = side_var.get(), std_var.get()
        window.destroy()
        return side, std
