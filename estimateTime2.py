import cv2
import math
import numpy as np
import tkinter as tk

def display(hour, minute, crop):
    canvas = tk.Tk()
    canvas.title("Analog to Digital")
    canvas.geometry("300x250")
    digit = tk.Label(canvas, font=("ds-digital", 65, "bold"), bg="black", fg="blue", bd=80)
    digit.grid(row=0, column=1)

    value = "{}:{:02d}".format(hour, minute)
    digit.config(text=value)
    canvas.mainloop()

    cv2.imshow("Crop", crop)  # crop を表示する
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_theta(xcenter, ycenter, x1, y1, x2, y2):
    l1 = math.sqrt(((xcenter - x1) ** 2) + ((ycenter - y1) ** 2))
    l2 = ycenter
    l3 = math.sqrt(((xcenter - x1) ** 2) + ((0 - y1) ** 2))
    cos_theta = ((l1 ** 2) + (l2 ** 2) - (l3 ** 2)) / (2 * l1 * l2)
    theta_radian = math.acos(cos_theta)
    theta_degrees = math.degrees(theta_radian)
    return theta_degrees, x1 > xcenter

def estimate_time_from_needle_info(xcenter, ycenter, xs1, ys1, xs2, ys2, xl1, yl1, xl2, yl2):
    theta_hours, right_hour = calculate_theta(xcenter, ycenter, xs1, ys1, xs2, ys2)
    theta_min, right_min = calculate_theta(xcenter, ycenter, xl1, yl1, xl2, yl2)

    hour = int(theta_hours / (6 * 5)) if right_hour else 12 - (int(theta_hours / (6 * 5))) if theta_hours > 0 else 12
    minute = int(theta_min / ((6 * 5) / 5)) if right_min else 60 - (int(theta_min / ((6 * 5) / 5))) if theta_min > 0 else 30

    return hour, minute

def process_image(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)

    height, width = gray_img.shape
    mask = np.zeros((height, width), np.uint8)

    edges = cv2.Canny(thresh, 10, 50)
    cimg = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1.2, 80)
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=50, maxRadius=300)


    if circles is not None:
        for i in circles[0, :]:
            i[2] = i[2] + 300
            cv2.circle(mask, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), thickness=-1)
    else:
        hour, minute = estimate_time_from_needle_info(width / 2, height / 2, 0, 0, 0, 0, 0, 0, 0, 0)
        display(hour, minute)
        return

    masked_data = cv2.bitwise_and(img, img, mask=mask)
    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = 0, 0, 0, 0

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

    x = x if x else 0
    y = y if y else 0
    w = w if w else 701
    h = h if h else 698

    crop = masked_data[y + 30: y + h - 30, x + 30: x + w - 30]
    find_needles(crop)

def find_needles(crop):
    height, width, _ = crop.shape
    ret, mask = cv2.threshold(crop, 10, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(crop, 100, 200)


    kernel = np.ones((11, 11), np.uint8)
    kernel1 = np.ones((13, 13), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel1, iterations=1)

    minLineLength = 1000
    maxLineGap = 10

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength, maxLineGap)

    xl1, xl2, yl1, yl2 = 0, 0, 0, 0
    xs1, xs2, ys1, ys2 = 0, 0, 0, 0

    l = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        hypo = math.sqrt(dx ** 2 + dy ** 2)
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
            hypo2 = math.sqrt(dx ** 2 + dy ** 2)
            if hypo2 == l[0]:
                m = hypo2
                xl1 = x1
                xl2 = x2
                yl1 = y1
                yl2 = y2
                cv2.line(crop, (xl1, yl1), (xl2, yl2), (255, 0, 0), 3)
            if m == l[0]:
                if hypo2 == l[f]:
                    if math.sqrt((xl2 - x2) ** 2 + (yl2 - y2) ** 2) > 20 and math.sqrt(
                            (xl1 - x1) ** 2 + (yl1 - y1) ** 2) > 20:
                        xs1 = x1
                        xs2 = x2
                        ys1 = y1
                        ys2 = y2
                        cv2.line(crop, (xs1, ys1), (xs2, ys2), (0, 255, 0), 3)
                        h = 1
                        break
        if h == 1:
            break

    hour, minute = estimate_time_from_needle_info(width / 2, height / 2, xs1, ys1, xs2, ys2, xl1, yl1, xl2, yl2)
    display(hour, minute, crop)

for _ in range(4):
    text = './clock/clock' + str(_ + 1) + '.jpg'
    process_image(text)
    
