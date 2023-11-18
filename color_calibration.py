# import neccessary libraries
import sys
import time
from datetime import datetime
import cv2
import kociemba
import numpy as np
from scipy import stats
import json

TOLERANCE = 10
# method to concate all the faces in a way so that it can be given to kociemba module


def face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face):
    # solution = [up_face,right_face,front_face,down_face,left_face,back_face]
    solution = np.concatenate(
        (up_face, right_face, front_face, down_face, left_face, back_face))
    # print(solution)
    return solution


# method to detect faces from the cube
def face_detection_in_cube(bgr_image_input):
    # convert  image to gray
    gray = cv2.cvtColor(bgr_image_input, cv2.COLOR_BGR2GRAY)

    # defining kernel for morphological operations using ellipse structure
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('gray',gray)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # gray = cv2.Canny(bgr_image_input,50,100)
    # cv2.imshow('gray',gray)
    # adjusting threshold to get countours easily
    # these needs to be changed based on the lighting condition you have and the environment you are using
    gray = cv2.adaptiveThreshold(
        gray, 5, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 0)
    # cv2.imshow('gray',gray)
    # cv2.imwrite()

    # for finding contours you can also use canny functions that is available in cv2 but for my environment
    # I found gray was working better
    try:
        # get contours from the image after applying morphological operations
        _, contours, hierarchy = cv2.findContours(
            gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    except:
        contours, hierarchy = cv2.findContours(
            gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    i = 0
    contour_id = 0
    # print(len(contours))
    count = 0
    colors_array = []
    for contour in contours:
        # get area of contours , obviously we don't want every contour in our image
        A1 = cv2.contourArea(contour)
        contour_id = contour_id + 1

        if A1 < 3000 and A1 > 1000:
            perimeter = cv2.arcLength(contour, True)

            # after checking the area we will estimate the epsilon structure
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # this is a just in case scenario
            hull = cv2.convexHull(contour)
            if cv2.norm(((perimeter / 4) * (perimeter / 4)) - A1) < 150:
                # if cv2.ma
                count = count + 1

                # get co ordinates of the contours in the cube
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(bgr_image_input, (x, y), (x + w, y + h), (0, 255, 255), 2)
                # cv2.imshow('cutted contour', bgr_image_input[y:y + h, x:x + w])
                val = (50 * y) + (10 * x)

                # get mean color of the contour
                color_array = np.array(
                    cv2.mean(bgr_image_input[y:y + h, x:x + w])).astype(int)

                # below code is to convert bgr color to hsv values so that i can use it in the if conditions
                # even bgr can be used but in my case hsv was giving better results
                blue = color_array[0] / 255
                green = color_array[1] / 255
                red = color_array[2] / 255

                cmax = max(red, blue, green)
                cmin = min(red, blue, green)
                diff = cmax - cmin
                hue = -1
                saturation = -1

                if (cmax == cmin):
                    hue = 0

                elif (cmax == red):
                    hue = (60 * ((green - blue) / diff) + 360) % 360

                elif (cmax == green):
                    hue = (60 * ((blue - red) / diff) + 120) % 360

                elif (cmax == blue):
                    hue = (60 * ((red - green) / diff) + 240) % 360

                if (cmax == 0):
                    saturation = 0
                else:
                    saturation = (diff / cmax) * 100

                value = cmax * 100

                # print(hue,saturation,value)
                # exit()

                color_array[0], color_array[1], color_array[2] = hue, saturation, value

                # print(color_array)
                cv2.drawContours(bgr_image_input, [
                                 contour], 0, (255, 255, 0), 2)
                cv2.drawContours(bgr_image_input, [
                                 approx], 0, (255, 255, 0), 2)
                color_array = np.append(color_array, val)
                color_array = np.append(color_array, x)
                color_array = np.append(color_array, y)
                color_array = np.append(color_array, w)
                color_array = np.append(color_array, h)
                colors_array.append(color_array)
    if len(colors_array) > 0:
        colors_array = np.asarray(colors_array)
        colors_array = colors_array[colors_array[:, 4].argsort()]
    face = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    # Farbintervalle hier anpassen
    # TODO: als KONSTANTEN definieren
    colors = [[], [], []]
    while len(colors[0]) < 20:
        if len(colors_array) == 9:
            for i in range(9):
                for j in range(3):
                    colors[j].append(colors_array[i][j])
            if np.count_nonzero(face) == 9:
                return face, colors
            else:
                return [0, 0], colors
        else:
            return [0, 0, 0], colors


def get_color_domains(video, videoWriter, text="", current_color="white"):
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        # assinging values to face and blob colors based on the face_detection_in_cube method
        face, colors_array = face_detection_in_cube(bgr_image_input)

        bgr_image_input = cv2.putText(
            bgr_image_input, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        if len(colors_array[0]) > 0:
            r_max = int(max(colors_array[0]) + TOLERANCE)
            r_min = int(min(colors_array[0]) - TOLERANCE)
            g_max = int(max(colors_array[1]) + TOLERANCE)
            g_min = int(min(colors_array[1]) - TOLERANCE)
            b_max = int(max(colors_array[2]) + TOLERANCE)
            b_min = int(min(colors_array[2]) - TOLERANCE)
            return {"r": [r_min, r_max], "g": [g_min, g_max], "b": [b_min, b_max]}
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            exit(0)


def wait(text, time, video, videoWriter):
    start_time = datetime.now()
    while True:
        if (datetime.now() - start_time).total_seconds() > time:
            break
        else:
            is_ok, bgr_image_input = video.read()
            if not is_ok:
                exit(1)
            bgr_image_input = cv2.putText(bgr_image_input, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                          2, (0, 0, 255), 3)
            videoWriter.write(bgr_image_input)
            cv2.imshow("Output Image", bgr_image_input)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == 27 or key_pressed == ord('q'):
                exit(0)


# main method
def main():
    white_domain = [0, 0]
    yellow_domain = [0, 0]
    blue_domain = [0, 0]
    green_domain = [0, 0]
    red_domain = [0, 0]
    orange_domain = [0, 0]

    # initialising web cam for recording
    video = cv2.VideoCapture(0)

    # video = cv2.VideoCapture('http://192.168.43.1:8080/video')
    is_ok, bgr_image_input = video.read()
    broke = 0

    if not is_ok:
        print("Cannot read video source")
        sys.exit()

    h1 = bgr_image_input.shape[0]
    w1 = bgr_image_input.shape[1]
    faces = []

    try:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fname = "OUTPUT5.avi"
        fps = 24.0
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, (w1, h1))
    except:
        print("Error: can't create output video: %s" % fname)
        sys.exit()

    while True:
        white_domain = get_color_domains(
            video, videoWriter, text="Show white Face", current_color="white")
        print(white_domain)

        broke = wait("show yellow face", 3, video, videoWriter)
        if broke:
            break
        yellow_domain = get_color_domains(
            video, videoWriter, text="Show yellow Face", current_color="yellow")
        print(yellow_domain)

        broke = wait("show blue face", 3, video, videoWriter)
        if broke:
            break
        blue_domain = get_color_domains(
            video, videoWriter, text="Show blue Face", current_color="blue")
        print(blue_domain)

        broke = wait("show green face", 3, video, videoWriter)
        if broke:
            break
        green_domain = get_color_domains(
            video, videoWriter, text="Show green Face", current_color="green")
        print(green_domain)

        broke = wait("show red face", 3, video, videoWriter)
        if broke:
            break
        red_domain = get_color_domains(
            video, videoWriter, text="Show red Face", current_color="red")
        print(red_domain)

        broke = wait("show orange face", 3, video, videoWriter)
        orange_domain = get_color_domains(
            video, videoWriter, text="Show orange Face", current_color="orange")
        print(orange_domain)
        domains = {"white": white_domain, "yellow": yellow_domain, "blue": blue_domain,
                   "green": green_domain, "red": red_domain, "orange": orange_domain}
        print(domains)
        with open("color_domains.json", "w") as json_file:
            json.dump(domains, json_file)
        exit(0)


if __name__ == "__main__":
    main()
