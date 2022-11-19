#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys

square_size = 4.0      # 正方形の1辺のサイズ[cm]
pattern_size = (8, 6)  # 交差ポイントの数

reference_img = 40 # 参照画像の枚数

pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
objpoints = []
imgpoints = []

cam_id = 6

if len(sys.argv) > 1:
    cam_id = int(sys.argv[1])

capture = cv2.VideoCapture(cam_id)

while len(objpoints) < reference_img:
# 画像の取得
    ret, img = capture.read()
    if img is None:
        print("Camera not detected")
        exit()

    height = img.shape[0]
    width = img.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # チェスボードのコーナーを検出
    ret, corner = cv2.findChessboardCorners(gray, pattern_size)
    # コーナーがあれば
    cv2.imshow('image', img)
    key = cv2.waitKey(30) & 0xFF

    if ret == True and key == ord('s'):
        print("detected coner!")
        print(str(len(objpoints)+1) + "/" + str(reference_img))
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(gray, corner, (5,5), (-1,-1), term)
        imgpoints.append(corner.reshape(-1, 2))   #appendメソッド：リストの最後に因数のオブジェクトを追加
        objpoints.append(pattern_points)
    elif key == ord('q'):
        break

print("calculating camera parameter...")
# 内部パラメータを計算
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 計算結果を保存
np.save("mtx", mtx) # カメラ行列
np.save("dist", dist.ravel()) # 歪みパラメータ
# 計算結果を表示
print("RMS = ", ret)
print("mtx = \n", mtx)
print("dist = ", dist.ravel())
