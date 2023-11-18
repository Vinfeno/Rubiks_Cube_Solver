# import neccessary libraries
import sys
import time
from datetime import datetime
import cv2
import kociemba
import numpy as np
from scipy import stats
import json
import pickle

with open("color_domains.json", "r") as json_file:
    COLOR_DOMAINS = json.load(json_file)

# method to concate all the faces in a way so that it can be given to kociemba module


def face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face):
    solution = np.concatenate(
        (up_face, right_face, front_face, down_face, left_face, back_face))
    solution = [item for sublist in solution for item in sublist]
    print("solution in concat:" + str(solution))
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
    if len(colors_array) == 9:
        print(colors_array)
        for i in range(9):
            # print(colors_array[i])
            # assign values to color_array and faces based on the hsv values
            if (COLOR_DOMAINS["white"]["r"][0] <= colors_array[i][0] <= COLOR_DOMAINS["white"]["r"][1] or colors_array[i][0] > 330) and COLOR_DOMAINS["white"]["g"][0] <= colors_array[i][1] <= COLOR_DOMAINS["white"]["g"][1] and COLOR_DOMAINS["white"]["b"][0] <= colors_array[i][2] <= COLOR_DOMAINS["white"]["b"][1]:
                colors_array[i][3] = 1
                face[i] = 1
                print('white detected')
            elif COLOR_DOMAINS["yellow"]["r"][0] <= colors_array[i][0] <= COLOR_DOMAINS["yellow"]["r"][1] and COLOR_DOMAINS["yellow"]["g"][0] <= colors_array[i][1] <= COLOR_DOMAINS["yellow"]["g"][1] and COLOR_DOMAINS["yellow"]["b"][0] <= colors_array[i][2] <= COLOR_DOMAINS["yellow"]["b"][1]:
                colors_array[i][3] = 2
                face[i] = 2
                print('yellow detected')
            elif COLOR_DOMAINS["blue"]["r"][0] <= colors_array[i][0] <= COLOR_DOMAINS["blue"]["r"][1] and COLOR_DOMAINS["blue"]["g"][0] <= colors_array[i][1] <= COLOR_DOMAINS["blue"]["g"][1] and COLOR_DOMAINS["blue"]["b"][0] <= colors_array[i][2] <= COLOR_DOMAINS["blue"]["b"][1]:
                colors_array[i][3] = 3
                face[i] = 3
                print('blue detected')
            elif COLOR_DOMAINS["green"]["r"][0] <= colors_array[i][0] <= COLOR_DOMAINS["green"]["r"][1] and COLOR_DOMAINS["green"]["g"][0] <= colors_array[i][1] <= COLOR_DOMAINS["green"]["g"][1] and COLOR_DOMAINS["green"]["b"][0] <= colors_array[i][2] <= COLOR_DOMAINS["green"]["b"][1]:
                colors_array[i][3] = 4
                face[i] = 4
                print('green detected')
            elif COLOR_DOMAINS["red"]["r"][0] <= colors_array[i][0] <= COLOR_DOMAINS["red"]["r"][1] and COLOR_DOMAINS["red"]["g"][0] <= colors_array[i][1] <= COLOR_DOMAINS["red"]["g"][1] and COLOR_DOMAINS["red"]["b"][0] <= colors_array[i][2] <= COLOR_DOMAINS["red"]["b"][1]:
                colors_array[i][3] = 5
                face[i] = 5
                print('red detected')
            elif COLOR_DOMAINS["orange"]["r"][0] <= colors_array[i][0] <= COLOR_DOMAINS["orange"]["r"][1] and COLOR_DOMAINS["orange"]["g"][0] <= colors_array[i][1] <= COLOR_DOMAINS["orange"]["g"][1] and COLOR_DOMAINS["orange"]["b"][0] <= colors_array[i][2] <= COLOR_DOMAINS["orange"]["b"][1]:
                colors_array[i][3] = 6
                face[i] = 6
                print('orange detected')
        # print(face)
        if np.count_nonzero(face) == 9:
            # print(face)
            # print (colors_array)
            return face, colors_array
        else:
            return [0, 0], colors_array
    else:
        return [0, 0, 0], colors_array
        # break


def find_face_in_cube(video, videoWriter, uf, rf, ff, df, lf, bf, text=""):
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
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 5:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                # print(final_face)
                uf = np.asarray(uf)
                ff = np.asarray(ff)
                detected_face = np.asarray(detected_face)
                # print(np.array_equal(detected_face, tf))
                # print(np.array_equal(detected_face, ff))
                faces = []
                if np.array_equal(detected_face, uf) == False and np.array_equal(detected_face,
                                                                                 ff) == False and np.array_equal(
                        detected_face, bf) == False and np.array_equal(detected_face, df) == False and np.array_equal(
                        detected_face, lf) == False and np.array_equal(detected_face, rf) == False:
                    return detected_face
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


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
    up_face = [0, 0]
    front_face = [0, 0]
    left_face = [0, 0]
    right_face = [0, 0]
    down_face = [0, 0]
    back_face = [0, 0]

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

    is_ok, bgr_image_input = video.read()
    if not is_ok:
        exit(1)

    front_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                   text="Show Front Face")
    mf = front_face[0, 4]
    print(front_face)
    print(type(front_face))
    print(mf)

    wait("show top face", 3, video, videoWriter)
    up_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                text="Show Top Face")
    mu = up_face[0, 4]
    print(up_face)
    print(mu)

    wait("show down face", 3, video, videoWriter)
    down_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                  text="Show Down Face")

    md = down_face[0, 4]
    print(down_face)
    print(md)

    wait("show right face", 3, video, videoWriter)
    right_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                   text="Show Right Face")
    mr = right_face[0, 4]
    print(right_face)
    print(mr)

    wait("show left face", 3, video, videoWriter)
    left_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                  text="Show Left Face")
    ml = left_face[0, 4]
    print(left_face)
    print(ml)

    wait("show back face", 3, video, videoWriter)
    back_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                  text="Show Back Face")
    mb = back_face[0, 4]
    print(back_face)
    print(mb)

    # append all the faces in the order so that it can be given to kociemba module
    solution = face_concatenation(
        up_face, right_face, front_face, down_face, left_face, back_face)
    print("Solution:")
    print(solution)
    cube_solved = [mu, mu, mu, mu, mu, mu, mu, mu, mu, mr, mr, mr, mr, mr, mr, mr, mr, mr, mf, mf, mf, mf, mf,
                   mf, mf, mf, mf, md, md, md, md, md, md, md, md, md, ml, ml, ml, ml, ml, ml, ml, ml, ml, mb,
                   mb, mb, mb, mb, mb, mb, mb, mb]
    print(cube_solved)
    # if (face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face) == cube_solved).all():
    #     print("CUBE IS SOLVED")
    #     is_ok, bgr_image_input = video.read()
    #     bgr_image_input = cv2.putText(bgr_image_input, "CUBE ALREADY SOLVED", (100, 50),
    #                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    #     videoWriter.write(bgr_image_input)
    #     cv2.imshow("Output Image", bgr_image_input)
    #     key_pressed = cv2.waitKey(1) & 0xFF
    #     if key_pressed == 27 or key_pressed == ord('q'):
    #         break
    #     time.sleep(5)
    #     break

    # assigning respective values to the faces
    ''' F -------> Front face
        R -------> Right face
        B -------> Back face
        L -------> Left face
        U -------> Up face
        D -------> Down face'''
    final_string = ''
    for val in range(len(solution)):
        if solution[val] == mf:
            final_string += 'F'
        elif solution[val] == mr:
            final_string += 'R'
        elif solution[val] == mb:
            final_string += 'B'
        elif solution[val] == ml:
            final_string += 'L'
        elif solution[val] == mu:
            final_string += 'U'
        elif solution[val] == md:
            final_string += 'D'

    print(final_string)
    solved = kociemba.solve(final_string)
    steps = solved.split()
    with open('cuber/solution.txt', 'w') as file:
        for step in steps:
            file.write(step + ",")
    print(steps)
    videoWriter.write(bgr_image_input)
    cv2.imshow("Output Image", bgr_image_input)
    # print(count)
    # print(color_array)
    # print(face)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == 27 or key_pressed == ord('q'):
        exit(0)


if __name__ == "__main__":
    main()
