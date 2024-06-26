import os
import argparse
import cv2
import time
from datetime import datetime as dt
import copy
import numpy as np
import shutil
from lib.key import Key, Mode
import serial


def white_detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([179, 150, 255]) #179, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask


def get_weight(point, image_width, image_height):
    # separate the area
    squears_h = np.linspace(0, image_height, 4)
    squears_w = np.linspace(0, image_width, 10)
    squears_h = np.round(squears_h)
    squears_w = np.round(squears_w)

    # value setting
    weight = 0
    h_weight = [0, 0.4, 1.4]
    w_weight = [-8, -6, -1.5, -1, 0, 1, 1.5, 6, 8]
    h_group = 0
    w_group = 0

    # check where the object is located
    for i in range(len(point)):
        for k in range(len(squears_h) - 1):
            for j in range(len(squears_w) - 1):
                if squears_h[k] <= point[i][1] < squears_h[k+1]:
                    h_group = h_weight[k]
                if squears_w[j] <= point[i][0] < squears_w[j+1]:
                    w_group = w_weight[j]
        weight += w_group * h_group
    steer = weight*1.5

    # modify the over value
    if steer > 50:
        steer = 50
    elif steer < -50:
        steer = -50
    return steer


def analysis_contours(binary_img, image_width, image_height):
    steer = 0
    dts = copy.deepcopy(binary_img)
    contours, hierarchy = cv2.findContours(dts, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(dts, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # fill all contour with white
    image = cv2.drawContours(dts, contours, -1, 255, -1)
    points = []

    for cnt in range(len(contours)):
        if len(contours[cnt]) > 0:
            area = cv2.contourArea(contours[cnt])

            if not 10 <= area <= 100:
                # fill not necessary object with black
                image = cv2.drawContours(image, contours, cnt, (0, 0, 255), -1)
            else:
                # culculate center of necessary object
                rect = contours[cnt]
                x, y, w, h = cv2.boundingRect(rect)
                x = x + w/2
                y = y + h/2
                points.append([x, y])
                # culculate steer by where the object is
                steer = get_weight(points, image_width, image_height)
    return image, steer


def separate_steer(mode, steer):
    # separate values
    interval = np.linspace(-50, 50, 16)
    interval_r = np.round(interval)
    for index, item in enumerate(interval_r):
        if index == 15:
            continue
        # calculate which range apply
        if interval_r[index] <= steer <= interval_r[index+1]:
            steer_cat = index
            steer = (interval_r[index]+interval_r[index+1])/2
    if mode == 0:
        return steer_cat, steer
    elif mode == 1:
        return steer_cat


def create_folder():
    # create folder for save image
    if not os.path.exists('image_data'):
        os.mkdir('image_data')
        os.mkdir('image_data/images')
    if os.path.exists('image_data/images'):
        shutil.rmtree('image_data/images')
        os.mkdir('image_data/images')


def main():

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='Argparseを設定します。')
    parser.add_argument('-s','--skip_frame', type=int, help='指定フレーム数ごとに画像を保存する', default=3, required=False)
    args = parser.parse_args()

    capture = cv2.VideoCapture(0)
    if capture.isOpened() is False:
        raise('IO Error')

    '''spike serial setting'''
    ser = serial.Serial('/dev/ttyAMA1', 115200)
    print(ser)

    # define mode depend on key-type
    mode = Mode()
    key = Key()
    log_flag, cnt = mode.get_mode_log()
    mode = mode.get_mode_control()

    # make image and list save folder
    create_folder()
    top_dir = './image_data/'
    txt_file = top_dir + 'list.txt'

    # define input type ofkey_control mode
    non_blocking = True

    # initialize value
    throttle = 20
    steer = 0

    image_width = 160
    image_height = 120

    write_cnt = 0

    while True:
        # getImage
        ret, frame = capture.read()  # iamge.shape->(480,640,3)
        if ret is False:
            continue

        # resize image and apply white mask
        reshaped_img = cv2.resize(frame, (image_width, image_height))
        #reshaped_img = reshaped_img[:, len(reshaped_img)/2:]
        #image_height //=2
        wh_hsv = white_detect(reshaped_img)

        # controle ev3
        # mode -> 0:Logging / 1:JustRun
        print(mode)
        if mode == 0:
            throttle = throttle
            target, steer = analysis_contours(wh_hsv, image_width, image_height)
            print(target)

        if mode == 1:
            throttle, steer = key.key_control(throttle, steer, non_blocking)
            if steer == 200:
                print("\n --- end")
                cmd = 'end,@'
                ser.write(cmd.encode('utf-8'))
                break

        elif mode == 100:  # Illegal Input
            break

        # controle spike
        steer = int(steer)
        cmd = '{:3d},{:3d}@'.format(throttle, steer)
        ser.write(cmd.encode('utf-8'))

        # write image
        if log_flag is True:
            if write_cnt % args.skip_frame == 0:
                cnt_str = f"{cnt:08}"
                img_name = cnt_str + '_' +  dt.now().strftime('%s') + '.jpg'
                img_path = top_dir + './images/' + img_name
                #cv2.imwrite(img_path, reshaped_img)
                cv2.imwrite(img_path, wh_hsv)
                #cv2.imwrite(img_path, target)
                with open(txt_file, mode='a') as f:
                    f.write('{} {} {}\n'.format(img_name, int(steer), int(throttle)))
                    print('throttle={}, steer={} cnt={}'.format(throttle, int(steer), cnt))
                cnt += 1
                write_cnt = 0
            write_cnt += 1

        elif log_flag is False:
            print('throttle={}, steer={}'.format(throttle, steer))


if __name__ == '__main__':
    main()
