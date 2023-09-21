#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import cv2
import torch
import numpy as np

#--- 検出する際のモデルを読込 ---
model = torch.hub.load('ultralytics/yolov5','yolov5x')#--- webのyolov5sを使用
#model = torch.hub.load("../yolov5",'yolov5x',source='local')#--- localのyolov5sを使用

#--- 検出の設定 ---
model.conf = 0.3 #--- 検出の下限値（<1）。設定しなければすべて検出
#信号機のみ検出
model.classes = [9] 

#ウェブカメラの読み込み
#camera = cv2.VideoCapture(0)                #--- カメラ：Ch.(ここでは0)を指定
# ビデオの読み込み
camera = cv2.VideoCapture('traffic_light_test.mp4')

def TrafficLightSignal(results):
    #results = model(imgArray_or_imgPath)
    #DisplayAreaOfInterest(results.crop()[0]['im'])
    img = ReturnPedestrianTrafficLight(results)

    if type(img) == np.ndarray:  # 歩行者用信号機の画像があるか確かめる
        RedBlueImgs = extractRedBlueArea(img)
    else:
        return None

    return ReturnTrafficLightSignal(RedBlueImgs)

def ReturnTrafficLightSignal(img_list):
    
    # 画像データはBGRとする
    upper_img = img_list[0]
    lower_img = img_list[1]

    # 上側（赤）のランプの状態を検出
    upper_red_nums = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)[:,:,0].mean()   # 赤色成分の平均値
    upper_blue_nums = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)[:,:,2].mean()  # 青色成分の平均値
    # 差分を求める
    upper_delta = abs(upper_red_nums - upper_blue_nums)

    # 下側（青）のランプの状態を検出
    lower_red_nums = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)[:,:,0].mean()   # 赤色成分の平均値
    lower_blue_nums = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)[:,:,2].mean()  # 青色成分の平均値
    # 差分を求める
    lower_delta = abs(lower_red_nums - lower_blue_nums)

    if upper_delta >= lower_delta:
        return 'Red'
    else:
        return 'Green'
    
# 上下から色判定領域を抽出する関数
def extractRedBlueArea(img):
    img_shape = img.shape
    w_c = int(img_shape[1] / 2)
    s = int(img_shape[1] / 6)
    # 上（赤色領域）のエリアにおける抽出画像の中心点を設定
    upper_h_c = int(img_shape[0] / 4)
    # 下（青色領域）のエリアにおける抽出画像の中心点を設定
    lower_h_c = int(img_shape[0] * 3 / 4)

    return [img[upper_h_c - s:upper_h_c + s, w_c - s:w_c+s, :], img[lower_h_c - s:lower_h_c + s, w_c - s:w_c+s, :]]

# 歩行者用信号機かチェックし、歩行者用信号機であればその画像を返す
def ReturnPedestrianTrafficLight(results):
    
    # 結果が0でないことを確認する(ただし、resultsに格納されているのはtraffic lightのみ)
    if len(results.crop(save=False)) == 0:
        return None
    # 何かしらの信号機が認識されている場合に以下のコードが実行される
    for traffic_light in results.crop(save=False):    # 全ての画像に適用
        img = traffic_light['im']
        img_shape = img.shape
        
        # 縦長画像であれば良い
        if img_shape[0] > img_shape[1]:
            return img    # 歩行者用信号機だったら、その部分を切り抜いた画像を返す
        else:
            return None   # 歩行者用信号機ではなかったらNoneを返す


while True:
    #--- 画像の取得 ---
    #  imgs = 'https://ultralytics.com/images/bus.jpg'#--- webのイメージファイルを画像として取得
    #  imgs = ["../pytorch_yolov3/data/dog.png"] #--- localのイメージファイルを画像として取得
    ret, imgs = camera.read()              #--- 映像から１フレームを画像として取得
    #動画が終了するとループから抜ける
    if not ret:
        break
    # フレームのサイズを取得
    height, width = imgs.shape[:2]

    # フレームのサイズを半分にする
    #imgs = cv2.resize(imgs, (width // 2, height // 2))


    #--- 推定の検出結果を取得 ---
    results = model(imgs) #--- サイズを指定しない場合は640ピクセルの画像にして処理
    #results = model(imgs, size=160) #--- 160ピクセルの画像にして処理
    Tcolor = TrafficLightSignal(results)

    #--- 出力 ---
    #--- 検出結果を画像に描画して表示 ---
    #--- 各検出について
    for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class

        #--- クラス名と信頼度を文字列変数に代入
        s = str(Tcolor)+" "+model.names[int(cls)]+":"+'{:.1f}'.format(float(conf)*100)

        #--- ヒットしたかどうかで枠色（cc）と文字色（cc2）の指定
        if str(Tcolor) == 'Red':
            cc = (0,0,255)
            cc2 = (0,0,0)
        elif str(Tcolor) == 'Green':
            cc = (0,255,0)
            cc2 = (0,0,0)
        else:
            cc = (255,255,255)
            cc2 = (0,0,225)

        #--- 枠描画
        cv2.rectangle(
          imgs,
          (int(box[0]), int(box[1])),
          (int(box[2]), int(box[3])),
          color=cc,
          thickness=2,
          )
        #--- 文字枠と文字列描画
        cv2.rectangle(imgs, (int(box[0]), int(box[1])-20), (int(box[0])+len(s)*10, int(box[1])), cc, -1)
        cv2.putText(imgs, s, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)


    #--- 描画した画像を表示
    cv2.imshow('color',imgs)

    #yolo標準の画面を画像取得してopencvで表示 ---
    #--- 「q」キー操作があればwhileループを抜ける ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()  


# In[ ]:




