import cv2
import numpy as np
import serial
import time

from tensorflow import keras
from tensorflow.keras.models import Model, load_model

def main():
    capture = cv2.VideoCapture(0)
    if capture.isOpened() is False:
        raise('IO Error')

    '''spike serial setting'''
    ser = serial.Serial('/dev/ttyAMA1', 115200)
    print(ser)

    # initialize value
    throttle = 20
    steer = 0

    image_width = 160
    image_height = 120

    # load model path
    model = load_model('./model.h5', compile=True)

    while True:
        # getImage
        ret, frame = capture.read()
        reshaped_img = cv2.resize(frame, (image_width, image_height))

        # predict
        reshaped_img = reshaped_img[np.newaxis,:,:]
        # 画像データを0から1の範囲に変換
        reshaped_img = reshaped_img.astype('float32')
        reshaped_img = reshaped_img / 255.0
        output = model.predict(reshaped_img, verbose=0)

        # 出力された値を調整
        steer = int(output[0][0] * 0.8)
        throttle = int(output[0][1] * 0.8)

        # controle spike
        print('throttle = {}, steer = {}'.format(throttle, steer))
        cmd = '{:3d},{:3d}@'.format(throttle, steer)
        ser.write(cmd.encode('utf-8'))

main()
