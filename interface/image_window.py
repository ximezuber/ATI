from tkinter import *

from PIL import ImageTk
from numpy import copy

from image_utils import *
from mouse_selector import Application


class ImageWindow:
    def __init__(self, main_window, img):
        # Internal variables
        self.img = img
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(self.img))
        new_id = len(main_window.windows)
        self.title = "Image " + str(new_id + 1)
        self.main_window = main_window
        self.id = new_id

        # Window layout
        self.top = Toplevel(main_window.root)
        self.top.title(self.title)
        self.frame = Frame(self.top)
        self.frame.pack()
        Label(self.frame, image=self.tk_img).pack()
        self.buttons_frame = Frame(self.top)
        self.buttons_frame.pack()
        self.top.protocol("WM_DELETE_WINDOW", self.destroy_window)

        # In selector mode variables
        self.selector = None
        self.in_select_buttons = [('Get info', self.get_info),
                                  ('Copy into other', self.copy_img_into_other),
                                  ('Crop', self.crop),
                                  ('Done', self.exit_selection_mode)]

        self.main_window.windows.append(self)

    # Exit selection
    def exit_selection_mode(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
        self.selector = None
        label = Label(self.frame, image=self.tk_img)
        label.pack()

        for widget in self.buttons_frame.winfo_children():
            widget.destroy()

    # Pixel quantity and mean
    def get_info(self):
        image = self._get_selection_image()

        new_window = Toplevel()
        frame = Frame(new_window)
        frame.pack()
        pixels, mean = pixels_info(image)
        Label(frame, text=f"Pixels: {pixels}, Mean: {mean}").grid(row=0, column=0, columnspan=3)
        Button(frame, text="Done", command=new_window.destroy, padx=20).grid(row=2, column=1)

    # Crop image
    def crop(self):
        ImageWindow(self.main_window, copy(self._get_selection_image()))

    # Go to selection mode
    def selection_mode(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
        self.selector = Application(self.tk_img, self.frame)
        self.selector.pack()
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()
        max_buttons_per_row = 5
        for i, button in enumerate(self.in_select_buttons):
            new_button = Button(self.buttons_frame, text=button[0], command=button[1])
            new_button.grid(row=(i // max_buttons_per_row), column=i % max_buttons_per_row)

    # Copy image section into other image
    def copy_img_into_other(self):
        img = self.main_window.select_img_from_windows()
        x, y = self.main_window.ask_xy('Enter upper left position to start copying')
        selection = self._get_selection()
        new_img = paste_section(self.img, selection[0], selection[1], selection[2], selection[3], img, x, y)
        ImageWindow(self.main_window, new_img)

    # This executes when closing window
    def destroy_window(self):
        self.main_window.windows.remove(self)
        if self.title in self.main_window.result_img.keys():
            self.main_window.result_img.pop(self.title)
        for i in range(0, len(self.main_window.windows)):
            window = self.main_window.windows[i]
            window.modify_id(i)
        self.top.destroy()

    # AUXILIARY FUNCTIONS

    def modify_id(self, new_id):
        self.id = new_id
        self.title = "Image " + str(new_id + 1)

        self.top.title(self.title)

    def _get_selection(self):
        start, end = self.selector.position_tracker.selection()
        x_start = min(start[0], end[0])
        y_start = min(start[1], end[1])
        x_end = max(start[0], end[0])
        y_end = max(start[1], end[1])
        return x_start, y_start, x_end, y_end

    def _get_selection_image(self):
        x_start, y_start, x_end, y_end = self._get_selection()
        image = self.img[y_start:y_end, x_start:x_end]
        return image
