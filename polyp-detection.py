import tkinter as tk

from tkinter import filedialog
from tkinter import font as tkFont
from PIL import Image, ImageTk
from tkinter import messagebox

import matplotlib.pyplot as plt
import skimage.io

from keras.models import load_model

from matplotlib.patches import Rectangle
from shapely.geometry import Polygon


def center_window(width=300, height=200):
    # get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # calculate position x and y coordinates
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))

def onOpen():
    global img
    global img_filepath
    global w,h

    filename = filedialog.askopenfilenames(title='select', filetypes=[
                    ("image", ".jpeg"),
                    ("image", ".png"),
                    ("image", ".jpg"),
                    ("image", ".tif"),
                    ("image", ".bmp"),
    ])

    try:
        if filename[0].endswith('.png'):
            img = tk.PhotoImage(file=filename)
            img_filepath = filename[0]
        elif filename[0].endswith('.tif'):
            outfile = 'tmp02.jpg'
            skimg = skimage.io.imread(filename[0], plugin='tifffile')  # load a test image
            skimage.io.imsave(outfile, skimg)

            img = ImageTk.PhotoImage(file=outfile)
            img_filepath = filename[0]
        else:
            img = ImageTk.PhotoImage(file=filename[0])
            img_filepath = filename[0]
        canvas.create_image(w/2, h/2, image=img)
    except:
        messagebox.showinfo("Detect Polyps", "Select a valid image file.")


# processing image
def onDetect():
    global img_output
    global img_filepath
    global canvas
    global model, w, h

    # WINDOW_SIZES = [150]  # using only one size for the sliding window
    WINDOW_SIZES = [150]  # using only one size for the sliding window
    window_sizes = WINDOW_SIZES
    step = 10  # step of sliding on the input image (how to divide the original image)
    # Read the colonoscopy image where we trying to find a polyp:

    if img_filepath.endswith('.tif'):
        skimg = skimage.io.imread(img_filepath, plugin='tifffile')  # load a test image
    else:
        tiff_filepath = 'tmp.tif'
        im = Image.open(img_filepath)
        im2 = im.convert('RGB')  # convert to black and white
        im2.save(tiff_filepath, 'TIFF')

        skimg = skimage.io.imread(tiff_filepath, plugin='tifffile')  # load a test image

    # Slide an window on the image to extract fragments 150 x 150 x 3 to make predictions:
    pred_arr = []

    max_pred = 0.0  # maximum prediction
    max_box = []  # box for the polyp detection
    limit_pred = 0.8
    # Loop window sizes: I will use only 150x150
    for win_size in window_sizes:
        # Loop on both dimensions of the image
        for top in range(0, skimg.shape[0] - win_size + 1, step):
            for left in range(0, skimg.shape[1] - win_size + 1, step):
                # compute the (top, left, bottom, right) of the bounding box
                box = (top, left, top + win_size, left + win_size)

                # crop the original image
                cropped_img = skimg[box[0]:box[2], box[1]:box[3], :]

                # normalize the cropped image (the same processing used for the CNN dataset)
                cropped_img = cropped_img * 1. / 255
                # reshape from (150, 150, 3) to (1, 150, 150, 3) for prediction
                cropped_img = cropped_img.reshape((1, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))

                # make a prediction for only one cropped small image
                preds = model.predict(cropped_img, batch_size=None, verbose=0)
                # print(box[0],box[2],box[1],box[3], preds[0][0])
                if preds[0][0] > max_pred:
                    max_pred = preds[0][0]
                    max_box = box

                if preds[0][0] >= limit_pred:
                    pred_arr.append([preds[0][0], box])
    # img = Image.fromarray(skimg)
    tmp_img = 'tmp.png'
    plt.figure()
    plt.imshow(skimg)
    plt.text(1, -5, 'Best probability: ' + str(max_pred) + ', Rect: ' + str(max_box), fontsize=10)
    currentAxis = plt.gca()
    max_w = max_box[3] - max_box[1]
    max_h = max_box[2] - max_box[0]

    currentAxis.add_patch(Rectangle((max_box[1], max_box[0]), max_w, max_h, linewidth=1, edgecolor='r', facecolor='none'))
    plt.text(max_box[1]+3, max_box[0]+10, 'prob: ' + str(max_pred), fontsize=10, color='red')

    area_to_show = [[max_pred, max_box]]
    # now, check other predicts except for max_pred
    for item in pred_arr:
        pred = item[0]
        box = item[1]
        detected_w = box[3] - box[1]
        detected_h = box[2] - box[0]

        # box[0] - x1
        # box[1] - y1
        # box[2] - x2
        # box[3] - y2
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        polygon = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

        can_be_displayed = True
        for area_item in area_to_show:
            area = area_item[1]
            # Calculate overlapped area of two Rectangle
            x1 = area[0]
            y1 = area[1]
            x2 = area[2]
            y2 = area[3]
            other_polygon = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            intersection = polygon.intersection(other_polygon)
            # print(intersection.area)
            limit_intersection = 150*150*0.2
            if intersection.area > limit_intersection:
                can_be_displayed = False

        if can_be_displayed:
            # add this area to area_to_show
            area_to_show.append([pred, box])
            # show this area
            currentAxis.add_patch(
                Rectangle((box[1], box[0]), detected_w, detected_h, linewidth=1, edgecolor='r', facecolor='none'))
            plt.text(box[1] + 3, box[0] + 10, 'prob: ' + str(pred), fontsize=10, color='red')

    # if detected_w < 150 and detected_h < 150:
    #     currentAxis.add_patch(Rectangle((max_box[1], max_box[0]), detected_w, detected_h, linewidth=1, edgecolor='r', facecolor='none'))
    # else:
    #     currentAxis.add_patch(Rectangle((max_box[1], max_box[0]), 150, 150, linewidth=1, edgecolor='r', facecolor='none'))
    # currentAxis.add_patch(Rectangle((max_box[1], max_box[0]), max_box[3], max_box[2], linewidth=1, edgecolor='r', facecolor='none'))
    # plt.show()
    plt.savefig(tmp_img)

    img_output = tk.PhotoImage(file=tmp_img)
    canvas_output.create_image(w/2, h/2, image=img_output)

root = tk.Tk()
w = 600
h = 500
delta = 10
button_h = 30
button_w = 100
center_window(w+delta*2+w+delta, h+delta*2+button_h+delta)
root.title('Detect Polyps')

canvas = tk.Canvas(root, width = w, height = h, bg='white')
canvas.place(x=delta, y=delta)

canvas_output = tk.Canvas(root, width = w, height = h, bg='white')
canvas_output.place(x=delta*2+w, y=delta)


button_font = tkFont.Font(family='Helvetica', size=13, weight=tkFont.BOLD)

button_openfile = tk.Button(root, text="Open Image", command=onOpen, bd='2', font=button_font)
button_openfile.place(x=delta, y=h+delta*2)

button_detect = tk.Button(root, text="Detect Polyps", command=onDetect, bd='2', font=button_font)
button_detect.place(x=delta + button_w + delta, y=h+delta*2)

img_filepath = ''

model = load_model('saved_models_CNN/model_best2_Conv-Conv-Conv-FC_full.h5')

root.mainloop()