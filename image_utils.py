from PIL import Image, ImageDraw
import numpy as np


def load(filename, w=None, h=None):
    if filename.lower().endswith('.raw'):
        img = np.fromfile(filename, dtype=np.uint8)
        img = img.reshape((h, w))
    else:
        img = np.array(Image.open(filename))
    return img


def save(image, name):
    im = Image.fromarray(image)
    im.save(name)


def get_pixel(image, x, y):
    return image[x][y]


def put_pixel(image, x, y, pixel):
    image[x][y] = pixel


# ul = upper-left
def paste_section(image_to_cut, x_start, y_start, x_end, y_end, new_image, new_ul_x, new_ul_y):
    image_to_cut = Image.fromarray(image_to_cut)
    new_image = Image.fromarray(new_image)
    section = image_to_cut.crop((x_start, y_start, x_end, y_end))
    new_image.paste(section, (new_ul_x, new_ul_y))
    return np.array(new_image)


def bin_circle(radius=50, img_height=200, img_width=200):
    img = Image.new("1", (img_width, img_height))  # 1 = binary image
    draw = ImageDraw.Draw(img)
    ellipse_shape = (img_width/2-radius, img_height/2-radius, img_width/2+radius, img_height/2+radius)
    draw.ellipse(ellipse_shape, fill=1)
    img.save('circle.jpg', quality=95)  # TODO: check extension


def bin_rectangle(hor_len=150, ver_len=100, img_height=200, img_width=200):
    img = Image.new("1", (img_width, img_height))  # 1 = binary image
    draw = ImageDraw.Draw(img)
    rectangle_shape = ((img_width/2-hor_len/2, img_height/2-ver_len/2), (img_width/2+hor_len/2, img_height/2+ver_len/2))
    draw.rectangle(rectangle_shape, fill=1)
    img.save('rectangle.jpg', quality=95)  # TODO: check extension
