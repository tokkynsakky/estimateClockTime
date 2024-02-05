import cv2
import os
import re
import pytesseract
import numpy as np
from datetime import datetime

path = "./clock/"
image_files = [f for f in os.listdir(path) if f.endswith(('.jpg'))]

def convert_to_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    elif len(image.shape) == 2:
        return image
    else:
        raise ValueError("Unsupported image format")


def preprocess_image(image):
    gray_image = convert_to_grayscale(image)
    # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 3)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(thresholded_image, 50, 200)

    return edges


def detect_clock_hand(image):
    lines = cv2.HoughLines(image, 1, np.pi / 180, threshold=80)
    clock_hands = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            angle = np.degrees(theta)
            clock_hands.append((angle, (x1, y1), (x2, y2)))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return clock_hands


def detect_digits(image):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_images = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if 100 < area < 300:
            x, y, w, h = cv2.boundingRect(contour)

            padding_percent = 0.1
            x_padding = int(w * padding_percent)
            y_padding = int(h * padding_percent)

            digit_image = image[max(y - y_padding, 0):min(y + h + y_padding, image.shape[0]),
                               max(x - x_padding, 0):min(x + w + x_padding, image.shape[1])]

            digit_images.append(digit_image)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return digit_images

def extract_digits_text(digit_images):
    extracted_digits = []
    for digit_image in digit_images:
        # '--psm 10' assumes a single character
        text = pytesseract.image_to_string(digit_image, config='--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789')
        cleaned_text = re.sub(r'\D', '', text)
        extracted_digits.append(cleaned_text)
        print(f"Raw Extracted Text: {cleaned_text}")
    return extracted_digits

def estimate_time_from_digits(extracted_digits):
    time_str = ''.join(extracted_digits)
    print(f"Time String: {time_str}")

    try:
        time_object = datetime.strptime(time_str, '%I%M')
        estimated_time = time_object.strftime('%I:%M %p')
        return estimated_time
    except ValueError as e:
        print(f"Error parsing time: {e}")
        return None

def estimate_time_from_clock_hands(clock_hands):
    # 追加: 角度を時、分、秒に分類
    hour_hand = None
    minute_hand = None
    second_hand = None

    for angle, point1, point2 in clock_hands:
        if 60 < angle < 120:  # 時針の角度範囲
            hour_hand = (angle, point1, point2)
        elif 150 < angle < 210:  # 分針の角度範囲
            minute_hand = (angle, point1, point2)
        elif 240 < angle < 300:  # 秒針の角度範囲
            second_hand = (angle, point1, point2)

    # 追加: 各針の位置から時刻を推定
    if hour_hand and minute_hand and second_hand:
        hour = int((hour_hand[0] - 60) / 30) % 12 + 1
        minute = int((minute_hand[0] - 150) / 6) % 60
        second = int((second_hand[0] - 240) / 6) % 60

        estimated_time = f'{hour:02d}:{minute:02d}:{second:02d}'
        return estimated_time
    else:
        return None


def overlay_text(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


for image_file in image_files:
    problematic_image_path = os.path.join(path, image_file)
    problematic_image = cv2.imread(problematic_image_path)
    problematic_gray_image = convert_to_grayscale(problematic_image)
    problematic_edges = preprocess_image(problematic_gray_image)
    clock_hands = detect_clock_hand(problematic_edges)
    digit_images = detect_digits(problematic_edges)
    
    if digit_images is not None and len(digit_images) > 0:
        extract_digits = extract_digits_text(digit_images)
        print(f"Extracted Digits: {extract_digits}")
    
        estimated_time = estimate_time_from_digits(extract_digits)
        if estimated_time:
            print(f"Estimated Time: {estimated_time}")
    
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(problematic_image, f"Estimated Time: {estimated_time}", (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
        else:
            print("Unable to estimate time.")
    
    # cv2.imshow("gray image", problematic_gray_image)
    cv2.imshow("edge image", problematic_edges)
    # cv2.imshow("digit image", digit_images)
    cv2.imshow("Problematic Image", problematic_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
