from PIL import Image, ImageDraw


# TODO: test formats (raw, pgm, ppm)

def load(filename, show=False, debug=False):
    img = Image.open(filename)
    if show:
        img.show()
    if debug:
        print(img.format)
        print(img.mode)
    return img


def save(image, name):
    image.save(name)


def get_pixel(image, row, column):
    image = image.load()
    return image[row, column]


# TODO: check what to do in other pixel formats (binary, grey scale, etc)
def put_pixel(image, x, y, r, g, b):
    image = image.load()
    image[x, y] = (r, g, b)


# ul = upper-left
def paste_section(image_to_cut, x_start, y_start, x_end, y_end, new_image, new_ul_x, new_ul_y):
    section = image_to_cut.crop((x_start, y_start, x_end, y_end))
    new_image.paste(section, (new_ul_x, new_ul_y))


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
