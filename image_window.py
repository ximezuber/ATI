from tkinter import *
from image_utils import *
from PIL import Image, ImageTk
from tkinter import filedialog


class ImageWindow(Toplevel):
    img = None
    tk_img = None

    def __init__(self, main_window, **kw):
        super().__init__(**kw)
        self.protocol("WM_DELETE_WINDOW", self.destroy_window)
        self.buttons = [('Save', self.save_image),
                        ('Get pixel', self.get_pixel_value),
                        ('Modify pixel', self.modify_pixel),
                        ('Copy into other', self.copy_img_into_other),
                        ('Sum', self.sum_other_image),
                        ('Subtract', self.subtract_other_image),
                        ('Multiply', self.multiply_other_image),
                        ('Show RGB bands', self.show_rgb),
                        ('Show RGB histogram', self.show_hist),
                        ('Show HSV bands', self.show_hsv)
                        ]
        self.main_window = main_window

    def update_image(self):
        self.add_image_from_array(self.img, self.title(), maintain_title=True)

    def copy(self):
        new_window = ImageWindow(self.main_window)
        new_window.add_image_from_array(self.img.copy(), self.title())
        return new_window

    def add_image(self, filename, w=None, h=None):
        self.img = load(filename, w, h)
        self.add_image_from_array(self.img, filename.split('/')[-1])

    def add_image_from_array(self, array, filename, maintain_title=False):
        self.img = array
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(array))
        label = Label(self, image=self.tk_img)
        label.grid(row=0, column=0, columnspan=len(self.buttons))

        for i, button in enumerate(self.buttons):
            new_button = Button(self, text=button[0], command=button[1])
            new_button.grid(row=1, column=i)

        starting_title = filename
        title = starting_title
        if maintain_title:
            self.title(title)
        else:
            i = 0
            while title in self.main_window.windows.keys():
                i += 1
                title = f"{starting_title} ({i})"
            self.title(title)
        self.main_window.windows[title] = self

    def destroy_window(self):
        if self.title() in self.main_window.windows.keys():
            self.main_window.windows.pop(self.title())
        super(ImageWindow, self).destroy()

    def save_image(self):
        # TODO: que no moleste con no poner extension en el filename
        filename = filedialog.asksaveasfilename(initialdir="./photos", title="Save As")

        save(self.img, filename)

        top = Toplevel()
        Label(top, text="File saves successfully").grid(row=0, column=0, columnspan=3)
        Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)

    def get_pixel_value(self):
        top = Toplevel()
        frame = Frame(top)
        frame.pack()
        Label(frame, text="Enter (x,y) coordinates of pixel").grid(row=0, column=0, columnspan=3)

        Label(frame, text="Enter x:").grid(row=1, column=0)
        x_entry = Entry(frame, width=10)
        x_entry.grid(row=2, column=0)

        Label(frame, text="Enter y:").grid(row=1, column=1)
        y_entry = Entry(frame, width=10)
        y_entry.grid(row=2, column=1)

        button = Button(frame, text="Enter", command=(lambda: self._show_pixel_value(top, frame, x_entry, y_entry)),
                        padx=20)
        button.grid(row=2, column=2)

    def modify_pixel(self):
        top = Toplevel()
        frame = Frame(top)
        frame.pack()
        Label(frame, text="Enter (x,y) coordinates of pixel").grid(row=0, column=0, columnspan=3)

        Label(frame, text="Enter x:").grid(row=1, column=0)
        x_entry = Entry(frame, width=10)
        x_entry.grid(row=2, column=0)

        Label(frame, text="Enter y:").grid(row=1, column=1)
        y_entry = Entry(frame, width=10)
        y_entry.grid(row=2, column=1)

        Label(frame, text="Enter new value:").grid(row=3, column=0, columnspan=2)
        new_entry = Entry(frame, width=20)
        new_entry.grid(row=4, column=0, columnspan=2)

        button = Button(frame, text="Enter",
                        command=(lambda: self._show_modified_pixel(top, frame, x_entry, y_entry, new_entry)), padx=20)
        button.grid(row=4, column=2)

    def copy_img_into_other(self):
        self._load_image_and_apply(self._select_copy_params)

    def sum_other_image(self):
        self._load_image_and_apply(self._sum_with_other)

    def subtract_other_image(self):
        self._load_image_and_apply(self._subtract_with_other)

    def multiply_other_image(self):
        self._load_image_and_apply(self._multiply_with_other)

    def show_rgb(self):
        r, g, b = split_rgb(self.img)
        r_img_window = ImageWindow(self.main_window)
        r_img_window.add_image_from_array(r, self.title() + '_R')
        b_img_window = ImageWindow(self.main_window)
        b_img_window.add_image_from_array(b, self.title() + '_B')
        g_img_window = ImageWindow(self.main_window)
        g_img_window.add_image_from_array(g, self.title() + '_G')

    def show_hsv(self):
        hsv_image = rgb_to_hsv(self.img)
        h, s, v = split_hsv(hsv_image)
        h_img_window = ImageWindow(self.main_window)
        h_img_window.add_image_from_array(h, self.title() + '_H')
        s_img_window = ImageWindow(self.main_window)
        s_img_window.add_image_from_array(s, self.title() + '_S')
        v_img_window = ImageWindow(self.main_window)
        v_img_window.add_image_from_array(v, self.title() + '_V')

    def show_hist(self):
        plot_hist(self.img)

    def _sum_with_other(self, img):
        new_img = add(self.img, img)
        new_window = ImageWindow(self.main_window)
        new_window.add_image_from_array(new_img, self.title())

    def _subtract_with_other(self, img):
        new_img = subtract(self.img, img)
        new_window = ImageWindow(self.main_window)
        new_window.add_image_from_array(new_img, self.title())

    def _multiply_with_other(self, img):
        new_img = multiply(self.img, img)
        new_window = ImageWindow(self.main_window)
        new_window.add_image_from_array(new_img, self.title())

    def _show_pixel_value(self, top, frame, x_entry, y_entry):
        x = int(x_entry.get())
        y = int(y_entry.get())

        frame.pack_forget()
        frame = Frame(top)
        frame.pack()

        value = get_pixel(self.img, x, y)

        Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(value)).grid(row=0, column=0,
                                                                                                   columnspan=3)
        Button(frame, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)

    # TODO: check RGB format
    def _show_modified_pixel(self, top, frame, x_entry, y_entry, new_value_entry):
        x = int(x_entry.get())
        y = int(y_entry.get())
        new_value = int(new_value_entry.get())

        frame.pack_forget()
        frame = Frame(top)
        frame.pack()

        new_window = self.copy()
        put_pixel(new_window.img, x, y, new_value)
        new_window.update_image()

        Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(new_value)) \
            .grid(row=0, column=0, columnspan=3)
        Button(frame, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)

    def _select_copy_params(self, img):
        new_window = Toplevel()
        frame = Frame(new_window)
        frame.pack()
        Label(frame, text="Enter (x1,y1) and (x2, y2) coordinates of pixel").grid(row=0, column=0, columnspan=3)

        Label(frame, text="Enter x1:").grid(row=1, column=0)
        x1 = Entry(frame, width=10)
        x1.grid(row=2, column=0)

        Label(frame, text="Enter y1:").grid(row=1, column=1)
        y1 = Entry(frame, width=10)
        y1.grid(row=2, column=1)

        Label(frame, text="Enter x2:").grid(row=3, column=0)
        x2 = Entry(frame, width=10)
        x2.grid(row=4, column=0)

        Label(frame, text="Enter y2:").grid(row=3, column=1)
        y2 = Entry(frame, width=10)
        y2.grid(row=4, column=1)

        Label(frame, text="Enter upper left position to start copying").grid(row=5, column=0, columnspan=2)

        Label(frame, text="Enter x:").grid(row=6, column=0)
        ulx = Entry(frame, width=10)
        ulx.grid(row=7, column=0)

        Label(frame, text="Enter y:").grid(row=6, column=1)
        uly = Entry(frame, width=10)
        uly.grid(row=7, column=1)

        button = Button(frame, text="Enter", command=(lambda: self._make_copy(
            int(x1.get()), int(y1.get()), int(x2.get()), int(y2.get()), int(ulx.get()), int(uly.get()), img,
            new_window)), padx=20)
        button.grid(row=8, columnspan=2)

    def _make_copy(self, x1, y1, x2, y2, ulx, uly, img, new_window):
        new_window.destroy()
        new_img = paste_section(self.img, x1, y1, x2, y2, img, ulx, uly)
        new_img_window = ImageWindow(self.main_window)
        new_img_window.add_image_from_array(new_img, self.title())

    def _destroy_and_apply(self, function, filename, w, h, window):
        img = load(filename, w, h)
        window.destroy()
        function(img)

    def _load_image_and_apply(self, function):
        filename = filedialog.askopenfilename(initialdir="./photos", title="Select an Image", filetypes=(
            ("raw", "*.RAW"), ("pgm", "*.pgm"), ("ppm", "*.ppm"), ("jpg", "*.jpg"), ("png", "*.png")))

        if not filename:
            return

        if filename.lower().endswith('.raw'):
            new_window = Toplevel()
            frame = Frame(new_window)
            frame.pack()
            new_window.pack_propagate(1)
            Label(frame, text="Enter image width:").grid(row=0, column=0)
            width = Entry(frame, width=10)
            width.grid(row=1, column=0)

            Label(frame, text="Enter image height:").grid(row=0, column=1)
            height = Entry(frame, width=10)
            height.grid(row=1, column=1)

            button = Button(frame, text="Enter",
                            command=(lambda: self._destroy_and_apply(function, filename,
                                                                     int(width.get()), int(height.get()),
                                                                     new_window)), padx=20)
            button.grid(row=1, column=2)
        else:
            function(load(filename))
