import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, acos, degrees
import tkinter as tk

def display(hour, minute):
    canvas = tk.Tk()
    canvas.title("Analog to Digital")
    canvas.geometry("300x250")
    digit = tk.Label(canvas, font=("ds-digital", 65, "bold"), bg="black", fg="blue", bd=80)
    digit.grid(row=0, column=1)

    if minute < 10:
        value = "{}:0{}".format(hour, minute)
    else:
        value = "{}:{}".format(hour, minute)

    digit.config(text=value)
    canvas.mainloop()

def estimate_time_from_needle_info(xs1, ys1, xs2, ys2, xl1, yl1, xl2, yl2):
    xcenter = int(width / 2)
    ycenter = int(height / 2)

    hour1 = abs(xcenter - xs1)
    hour2 = abs(xcenter - xs2)

    if hour1 > hour2:
        xhour = xs1
        yhour = ys1
    else:
        xhour = xs2
        yhour = ys2

    min1 = abs(xcenter - xl1)
    min2 = abs(xcenter - xl2)

    if min1 > min2:
        xmin = xl1
        ymin = yl1
    else:
        xmin = xl2
        ymin = yl2

    l1 = sqrt(((xcenter - xhour) ** 2) + ((ycenter - yhour) ** 2))
    l2 = ycenter
    l3 = sqrt(((xcenter - xhour) ** 2) + ((0 - yhour) ** 2))
    cos_theta_hour = ((l1 ** 2) + (l2 ** 2) - (l3 ** 2)) / (2 * l1 * l2)
    theta_hours_radian = acos(cos_theta_hour)
    theta_hours = math.degrees(theta_hours_radian)

    if xhour > xcenter:
        right = 1
    else:
        right = 0

    if right == 1:
        hour = int(theta_hours / (6 * 5))

    if right == 0:
        hour = 12 - (int(theta_hours / (6 * 5)))

    if hour == 0:
        hour = 12

    l1 = sqrt(((xcenter - xmin) ** 2) + ((ycenter - ymin) ** 2))
    l2 = ycenter
    l3 = sqrt(((xcenter - xmin) ** 2) + ((0 - ymin) ** 2))
    cos_theta_min = ((l1 ** 2) + (l2 ** 2) - (l3 ** 2)) / (2 * l1 * l2)
    theta_min_radian = acos(cos_theta_min)
    theta_min = math.degrees(theta_min_radian)

    if xmin > xcenter:
        right = 1
    else:
        right = 0

    if right == 1:
        minute = int(theta_min / ((6 * 5) / 5))

    if right == 0:
        minute = 60 - (int(theta_min / ((6 * 5) / 5)))
        if xmin == xcenter:
            minutes = 30

    return hour, minute


kernel = np.ones((5, 5), np.uint8)

# img = cv2.imread('./clock/clock1.jpg')
# img = cv2.imread('./clock/clock2.jpg')
# img = cv2.imread('./clock/clock3.jpg')
img = cv2.imread('./clock/clock4.jpg')



gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)

height, width = gray_img.shape
mask = np.zeros((height, width), np.uint8)

edges = cv2.Canny(thresh, 10, 50)

cimg = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

xl1, xl2, yl1, yl2 = 0, 0, 0, 0
xs1, xs2, ys1, ys2 = 0, 0, 0, 0
x, y, w, h = 0, 0, 0, 0

circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1.2, 80)

if circles is not None:
    for i in circles[0, :]:
        i[2] = i[2] + 300
        cv2.circle(mask, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), thickness=-1)
else:
    hour, minute = estimate_time_from_needle_info(xs1, ys1, xs2, ys2, xl1, yl1, xl2, yl2)
    

masked_data = cv2.bitwise_and(img, img, mask=mask)

_, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    (x, y, w, h) = cv2.boundingRect(cnt)
    print(x)
    print(y)
    print(w)
    print(h)

x = x if x else 0
y = y if y else 0
w = w if w else 701 
h = h if h else 698 

crop = masked_data[y + 30: y + h - 30, x + 30: x + w - 30]
i = crop

height, width, channels = i.shape

ret, mask = cv2.threshold(i, 10, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(i, 100, 200)

kernel = np.ones((11, 11), np.uint8)

kernel1 = np.ones((13, 13), np.uint8)

edges = cv2.dilate(edges, kernel, iterations=1)

edges = cv2.erode(edges, kernel1, iterations=1)

minLineLength = 1000
maxLineGap = 10

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength, maxLineGap)

# xl1, xl2, yl1, yl2 = 0, 0, 0, 0
# xs1, xs2, ys1, ys2 = 0, 0, 0, 0

l = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    dx = x2 - x1
    dy = y2 - y1
    hypo = sqrt(dx ** 2 + dy ** 2)
    l.append(hypo)

a = len(l)
l.sort(reverse=True)
m = 0
h = 0

for f in range(a):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        hypo2 = sqrt(dx ** 2 + dy ** 2)
        if hypo2 == l[0]:
            m = hypo2
            xl1 = x1
            xl2 = x2
            yl1 = y1
            yl2 = y2
            cv2.line(crop, (xl1, yl1), (xl2, yl2), (255, 0, 0), 3)
        if m == l[0]:
            if hypo2 == l[f]:
                if sqrt((xl2 - x2) ** 2 + (yl2 - y2) ** 2) > 20 and sqrt((xl1 - x1) ** 2 + (yl1 - y1) ** 2) > 20:
                    xs1 = x1
                    xs2 = x2
                    ys1 = y1
                    ys2 = y2
                    cv2.line(crop, (xs1, ys1), (xs2, ys2), (0, 255, 0), 3)
                    h = 1
                    break
    if h == 1:
        break

xcenter = int(width / 2)
ycenter = int(height / 2)

hour1 = abs(xcenter - xs1)
hour2 = abs(xcenter - xs2)

if hour1 > hour2:
    xhour = xs1
    yhour = ys1
else:
    xhour = xs2
    yhour = ys2

min1 = abs(xcenter - xl1)
min2 = abs(xcenter - xl2)

if min1 > min2:
    xmin = xl1
    ymin = yl1
else:
    xmin = xl2
    ymin = yl2

l1 = sqrt(((xcenter - xhour) ** 2) + ((ycenter - yhour) ** 2))
l2 = ycenter
l3 = sqrt(((xcenter - xhour) ** 2) + ((0 - yhour) ** 2))
cos_theta_hour = ((l1 ** 2) + (l2 ** 2) - (l3 ** 2)) / (2 * l1 * l2)
theta_hours_radian = acos(cos_theta_hour)
theta_hours = math.degrees(theta_hours_radian)

if xhour > xcenter:
    right = 1
else:
    right = 0

if right == 1:
    hour = int(theta_hours / (6 * 5))

if right == 0:
    hour = 12 - (int(theta_hours / (6 * 5)))

if hour == 0:
    hour = 12

l1 = sqrt(((xcenter - xmin) ** 2) + ((ycenter - ymin) ** 2))
l2 = ycenter
l3 = sqrt(((xcenter - xmin) ** 2) + ((0 - ymin) ** 2))
cos_theta_min = ((l1 ** 2) + (l2 ** 2) - (l3 ** 2)) / (2 * l1 * l2)
theta_min_radian = acos(cos_theta_min)
theta_min = math.degrees(theta_min_radian)

if xmin > xcenter:
    right = 1
else:
    right = 0

if right == 1:
    minute = int(theta_min / ((6 * 5) / 5))

if right == 0:
    minute = 60 - (int(theta_min / ((6 * 5) / 5)))
    if xmin == xcenter:
        minutes = 30

display(hour, minute)
