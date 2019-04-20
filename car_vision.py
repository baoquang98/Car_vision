import cv2
import numpy as np


def make_coord(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * 3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coord(image, left_fit_average)
    right_line = make_coord(image, right_fit_average)
    return np.array([left_line, right_line])


def gradient(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    gradient_img = cv2.Canny(blur_img, 50, 150)
    return gradient_img


def draw_line(image, lines):
    lines_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_img


def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([
        [(450, height), (1100, height), (550, 200)],
        [(0, height), (60, height-200), (550, 200)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# gradient_image = gradient(lane_image)
# cropped_image = region_of_interest(gradient_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# lines_image = draw_line(lane_image, averaged_lines)
# lines_imposed_image = cv2.addWeighted(lane_image, 0.8, lines_image, 1, 1)
# cv2.imshow("result", lines_imposed_image)
# cv2.waitKey(0)


cap = cv2.VideoCapture("test2.mp4")
width_vid = int(cap.get(3))
height_vid = int(cap.get(4))
fps_vid = int(cap.get(5))
out_vid = cv2.VideoWriter("output.avi", 0, fps_vid, (width_vid, height_vid))

while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        gradient_image = gradient(frame)
        cropped_image = region_of_interest(gradient_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=50, maxLineGap=5)
        averaged_lines = average_slope_intercept(frame, lines)
        lines_image = draw_line(frame, averaged_lines)
        lines_imposed_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)
        cv2.imshow("result", lines_imposed_image)
        out_vid.write(lines_imposed_image)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
out_vid.release()
cap.release()
cv2.destroyAllWindows()
