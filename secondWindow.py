from tkinter import * 
from numpy import copy
from copy import deepcopy

class secondWindow:

    def __init__(self, id, root):
        self.top = Toplevel(root)
        self.frame = Frame(self.top)
        self.frame.pack()
        self.title = "Image " + str(id + 1)

        self.top.title(self.title)
        self.id = id

        self.img = None
        self.tk_img = None
        
    
    def modify_id(self, id):
        self.id = id
        self.title = "Image " + str(id + 1)

        self.top.title(self.title)
    

    def copy_img(self, img):
        self.img = copy(img)
