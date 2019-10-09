import cv2
import numpy as np
import os


"""
This code is for making experiment data easily in code.

Click once => Drawing
Click again => Stop Drawing

data_path: Color image path
save_file_name: Just name, don't need filename extension. (will save .bmp format)
"""


data_path = "./examples/ground_truth.jpg"
save_file_name = "example2"
fill_val = np.array([255, 255, 255], np.int)

mouseX, mouseY = 0, 0
drawing = False
R, G, B = 0., 0., 0.


def click_mouse(event, x, y, flags, param):
    global mouseX, mouseY, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            drawing = True
        else:
            drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img[y-5:y+5, x-5:x+5] = [B, G, R]
        mouseX, mouseY = x, y


def trackbar_callback(idx, value):
    fill_val[idx] = value


origin_img = cv2.imread(data_path)
img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
img = np.stack((img, img, img), axis=-1) / 255.
temp_img = np.zeros((img.shape[0], img.shape[1], 3))

cv2.namedWindow("image")
cv2.setMouseCallback('image', click_mouse)
cv2.createTrackbar("R", "image", 0, 255, lambda v: trackbar_callback(2, v))
cv2.createTrackbar("G", "image", 0, 255, lambda v: trackbar_callback(1, v))
cv2.createTrackbar("B", "image", 0, 255, lambda v: trackbar_callback(0, v))

while True:
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    R = cv2.getTrackbarPos("R", 'image') / 255.
    G = cv2.getTrackbarPos("G", 'image') / 255.
    B = cv2.getTrackbarPos("B", 'image') / 255.
    temp_img[:] = [B, G, R]
    vertical_img = np.concatenate((img, temp_img), axis=0)
    cv2.imshow('image', vertical_img)

cv2.destroyAllWindows()
img = img * 255
img_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./examples/{}.bmp".format(save_file_name), img_gray)
cv2.imwrite("./examples/{}_marked.bmp".format(save_file_name), img.astype(np.int))
