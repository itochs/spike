# -*- coding: utf-8 -*-
# インポート
import os
import glob
import argparse
import numpy as np
import zipfile
import time
import datetime

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import add
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Lambda
from tensorflow.python.keras.losses import Huber as huber_loss

from tensorflow.keras.callbacks import ModelCheckpoint

# from common_flags import FLAGS
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical

# 訓練用ニューラルネットワーク定義
def cnn(input_shape):
    drop = 0.5
    # Input
    img_input = Input(shape=(input_shape))
    x1 = Conv2D(24, (5,5), strides=(2,2), activation='relu', name="conv2d_1")(img_input)
    x2 = Conv2D(32, (5,5), strides=(2,2), activation='relu', name="conv2d_2")(x1)
    x3 = Conv2D(64, (5,5), strides=(2,2), activation='relu', name="conv2d_3")(x2)
    x4 = Conv2D(64, (3,3), strides=(2,2), activation='relu', name="conv2d_4")(x3)
    x5 = Conv2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_5")(x4)

    x = Flatten(name='flattened')(x5)
    x = Dense(100, activation='relu', name="fc_1")(x)
    x = Dropout(drop)(x)

    x = Dense(50, activation='relu', name="fc_2")(x)
    x = Dropout(drop)(x)

    output = Dense(2, name='output')(x)

    model = Model(inputs=[img_input], outputs=[output])

    return model

# 訓練結果表示用関数の定義
# loss
def plot_history_loss(history):
    # Plot the loss in the history
    plt.plot(history.history['loss'], label='loss for training')
    plt.plot(history.history['val_loss'], label='loss for validation')
    plt.title('model loss')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('./result.png')
    plt.show()
    plt.close()

# メイン関数の定義
def _main():
    start = time.time()

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='Argparseを設定します。')
    parser.add_argument('-u','--unzip', type=bool, help='zipを解答する場合はTrue', default=False, required=False)
    parser.add_argument('-i','--input_dir', type=str, help='読み込むファイルの場所を指定', default='./image_data', required=False)
    args = parser.parse_args()

    if args.unzip:
        with zipfile.ZipFile(args.input_dir + '.zip') as existing_zip:
            existing_zip.extractall('./')

    img_list = pd.read_csv(args.input_dir + '/list.txt', header=None, delim_whitespace=True)
    image_path = args.input_dir + '/images/' + img_list[0]

    print("\ninput dir -->", args.input_dir)

    # 訓練時の画像サイズ
    image_width = 160
    image_height = 120
    X = []
    Y = []

    # 画像データの読み込み
    for index, file_name in enumerate(image_path):
        image = cv2.imread(file_name)
        resize_image = cv2.resize(image, (image_width, image_height))
        data = np.asarray(resize_image)
        X.append(data)
        Y.append([img_list[1][index], img_list[2][index]])

    X = np.array(X)
    Y = np.array(Y)

    # 画像データを0から1の範囲に変換
    X = X.astype('float32')
    X = X / 255.0
    # 正解データのデータ型を変換
    Y = Y.astype('float32')

    print("image shape -->", X.shape, "\n")

    # 学習用データとテストデータ
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

    # CNNの構築
    shape = X_train.shape[1:]
    model = cnn(shape)

    # ハイパーパラメータ
    decay = 1e-7
    learning_rate = 1e-3
    batch_size = 64
    epochs = 100

    # コンパイル
    optimizer = optimizers.Adam(learning_rate = learning_rate, decay = decay)
    model.compile(loss='huber_loss', optimizer=optimizer)

    # 訓練
    history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test))

    # モデルの保存
    model.save('./model.h5')

    # 評価 & 評価結果出力
    print("score :", model.evaluate(X_test, Y_test))

    # モデルの図示化
    plot_model(model, to_file='model.png', show_shapes=True)

    # モデルテスト
    print("score :", model.evaluate(X_test, Y_test))

    # show result graph
    plot_history_loss(history)

    # 経過時間の集計
    process_time = time.time() - start
    td = datetime.timedelta(seconds = process_time)
    print('\nPROCESS TIME = {}'.format(td))

# プログラムの実行
_main()
