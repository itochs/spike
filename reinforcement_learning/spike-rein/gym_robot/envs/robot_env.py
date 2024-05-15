import gym
import gym.spaces
import numpy as np
from gym_robot.envs.lib.ev3 import EV3
from gym_robot.envs.lib.vstream import VideoStream
from PIL import Image
import math 
import random
import datetime
import os
import time
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

touch_port = EV3.PORT_2
color_port = EV3.PORT_3
lmotor_port = EV3.PORT_B
rmotor_port = EV3.PORT_C

ev3 = EV3()
ev3.motor_config(lmotor_port, EV3.LARGE_MOTOR)
ev3.motor_config(rmotor_port, EV3.LARGE_MOTOR)
ev3.sensor_config(touch_port, EV3.TOUCH_SENSOR)
ev3.sensor_config(color_port, EV3.COLOR_SENSOR)


class Robot(gym.Env):
    MAX_STEPS = 1000
    reward = 0

    def __init__(self):
        super().__init__
        self.path = 'ml_rein_data/rf_data/images'
        self.img = np.array(Image.open('sample-1.png'))
        self.image_data = np.zeros((64,48))
        self.image_data = self.image_data/255.0
        self.reward_sum = 0
        self.action_space = gym.spaces.Discrete(4)

        self.min_action = -50
        self.max_action = 50
        self.min_width = 0
        self.max_width = 64
        self.min_height = 0
        self.max_height = 48
        
        self.low = np.array([self.min_width, self.min_height])
        self.high = np.array([self.max_width, self.max_height])
        
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)
        self.before = 0
        self.pos_x, self.pos_y, self.pos_z = self._find_pos()
        self.reward_range = [-1.,100.]
        self.step = 0
        self._reset()
    
    #状態を初期化する
    def _reset(self):
        self.steps = 0
        action = 0
        self.black_distance = 0
        self.reward_sum = 0
        ev3.reset_event()
        self.done_flag = False
        self.first_step = False
        return self.image_data
    
    #画面を表示させる
    def _render(self,mode='human'):
        self.vs = VideoStream((64,48), framerate=10, colormode='binary').start()
        return self.vs
    
    #actionを実行し, 結果を返す
    #仮想環境の中で画像とsteerの値を取得する
    def _step(self, action):
        self.steps += 1
        vs = VideoStream((64,48), framerate=10, colormode='binary').start() #画像を取得する
        ev3.motor_steer(lmotor_port, rmotor_port, 20, action)
        time.sleep(0.1) #0.1秒経過後の画像
        self.image_data = np.array(vs.read(), dtype = np.uint8)
        self.image_data = self.image_data.transpose(1, 0)
        
        analysis_distance = self._analysis_blob(self.image_data) #画像からライン上か一面黒白か判断し、_is_doneのフラグを操作
        self.image_data = self.image_data/255.0
        reward = self._get_reward(self.black_distance)
        self.done = self._is_done()
        observation = self.image_data
        return observation, reward, self.done, {} 
    
    def _is_done(self):
        if self.done_flag == True:
            self._reset()
            return True
        elif self.steps > self.MAX_STEPS:
            self._reset()
            return True
        else:
            self.reward = 0
            return False
        
    def _close(self):
        pass

    def _seed(self, seed=None):
        pass
    
    """
    視点を計算する関数(中央を算出)
    openしているsample.pngは撮影する画像と同じサイズの画像
    """
    def _find_pos(self):
        self.im = Image.open('sample-1.png')
        pos_x = round(self.im.size[0]/2)
        pos_y = round(self.im.size[1]/2)
        pos_z = np.array([pos_x, pos_y])
        
        return pos_x, pos_y, pos_z

    
    def _analysis_blob(self, img):
        # 2値画像のラベリング処理
        # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        
        # nlabels : number of labels
        # lavels : array of image
        # stats : bounding box's info of objects 
        # centroids : centroid of object
        label = cv2.connectedComponentsWithStats(img)
        self.done_flag = False        
        """
        ブロブ解析すると下にある各ラベルを抽出できる
        """
        # ブロブ情報を項目別に抽出
        object_num = label[0] # - 1
        data = label[2]
        center = label[3]

        data_del = np.delete(label[2], 0, 0)
        center_del = np.delete(label[3], 0, 0)

        # 面積最大ブロブの情報格納用
        maxblob = {}
        if len(data_del) != 0:
            # ブロブ面積最大のインデックス
            max_index = np.argmax(data_del[:, 4])
            # 面積最大ブロブの各種情報を取得
            maxblob_area = data_del[:, 4][max_index]   # 面積
            maxblob_center = np.array(center_del[max_index])  # 中心座標
        else:
            maxblob_area = 3072

        if object_num == 3: #ラベル数
            #print('after')
            black_center = np.array(center[0]) #ラベル0(黒)の重心取得
            #print('black_center:  '+str(black_center))
            pre_discalc = self.pos_z -black_center
            self.black_distance = np.linalg.norm(pre_discalc)
            #print('black_distance:  ' + str(self.black_distance))
            return self.black_distance
        elif object_num < 2:
            self.black_distance = 40
            self.done_flag = True
            return self.done_flag, self.black_distance
        
        if maxblob_area < 2700:
            pre_calc =  self.pos_z - maxblob_center
            self.distance = np.linalg.norm(pre_calc)
            return self.distance
        elif maxblob_area >= 2700:
            self.done_flag = True
            self.distance = 40 #画素の半分 32,14から求めた値
            return self.done_flag

    def _get_reward(self, distance):
        if 5 >= distance > 0 :
            self.reward_sum = 5
            return self.reward_sum
        elif 10 >= distance >5 :
            self.reward_sum = 3
            return self.reward_sum
        elif 15 >= distance > 10 :
            self.reward_sum = 1
            return self.reward_sum
        elif 30 >= distance > 15 :
            self.reward_sum = -2
            return self.reward_sum
        elif 40 > distance > 30 :
            self.reward_sum = -3
            return self.reward_sum
        elif distance >= 40:
            self.reward_sum = -6 #ラインから外れるときの判定は一瞬なので大きく罰を与える　インターバルの中ですぐ外れるようになると罰が溜まりやすくなる
            return self.reward_sum
        else:
            self.reward_sum = -6
            return self.reward_sum
