from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import csv
import random

def getMask(img):
    # Get mask corresponding to nonzero elements
    arr = np.array(img)
    if len(arr.shape) == 3:
        arr = arr.sum(2)
    mask = arr > 0
    arr[mask] = 255
    return Image.fromarray(arr, "L")

def overlay(front_img, back_img):
    # Extracts the mask from the front img
    mask = getMask(front_img)
    new_img = back_img.copy()
    new_img.paste(front_img, mask=mask)
    return new_img

def draw_ellipse(img, xy, color):
    # Draws a random ellipse in the image of a random size
    ((x0, y0), (x1, y1)) = xy
    x = np.random.random() * (x1 - x0) + x0
    y = np.random.random() * (y1 - y0) + y0
    w = np.random.random() * (x1 - x0)/4
    h = np.random.random() * (y1 - y0)/4
    draw = ImageDraw.Draw(img)
    draw.ellipse(((x - w, y - h), (x + w, y + h)), fill=color)

def distort_entire_img(img, xy):
    w,h = img.size
    draw = ImageDraw.Draw(img)
    num = np.random.randint(5, 10)
    for _ in range(0, num):
        draw_ellipse(img, xy, (0))

def square_front_img(size, mode, xy, color):
    # Creates a front image along with its mask
    img = Image.new(mode=mode, size=size, color=0)
    draw = ImageDraw.Draw(img)
    draw.rectangle(xy, fill=color)
    mask = getMask(img)
    distort_entire_img(img, xy)
    return img, mask

def gen_front_imgs(size, mode, color):
    w,h = size
    wx = w//8
    wy = h//8
    img_list = []
    for cx in np.linspace(w//8, 7*w//8, num=10):
        for cy in np.linspace(w//8, 7*w//8, num=10):
            xy = ((cx - wx, cy - wy), (cx + wx, cy + wy))
            img_list.append(square_front_img(size, mode, xy, color))
    return img_list

def gen_back_imgs(size, mode, color_list):
    img_list = []
    for color in color_list:
        img_list.append(Image.new(mode=mode, size=size, color=color))
    return img_list

def generate_partition_file(img_list, filename):
    random.shuffle(img_list)
    p = 0.8
    total = len(img_list)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for idx, (img_name, label_name) in enumerate(img_list):
            split = 'train' if idx < p*total else 'val'
            writer.writerow([str(img_name), str(label_name), str(split)])

def main():
    size=(256, 256)
    mode = "L"
    color_list = [int(x) for x in np.linspace(50, 220, 10)]
    back_img_list = gen_back_imgs(size, mode, color_list)
    img_count = 1
    all_img_list = []
    dir_name = Path('data/imgs')
    dir_name.mkdir(parents=True, exist_ok=True)
    for back_img, back_color in zip(back_img_list, color_list):
        front_img_list1 = gen_front_imgs(size, mode, back_color + 10)
        front_img_list2 = gen_front_imgs(size, mode, back_color - 10)
        front_img_list1.extend(front_img_list2)
        for front_img, label in front_img_list1:
            new_img = overlay(front_img, back_img)
            img_name = dir_name / "{:06d}.png".format(img_count)
            label_name = dir_name / "{:06d}_L.png".format(img_count)
            all_img_list.append((img_name, label_name))
            new_img.save(img_name)
            label.save(label_name)
            print(img_name, label_name, 'done')
            img_count += 1
    generate_partition_file(all_img_list, 'data/partition.csv')

if __name__ == '__main__':
    main()
            
