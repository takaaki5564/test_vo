#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import time
import math

from slam.utils import LoadDataset, PinholeCamera
from slam.feature import DescriptorFeatureTracker, BfFeatureMatcher
from slam.frame import Frame, match_frames

class calcTime:

    def __init__(self):
        self.st = cv2.getTickCount()
        self.st0 = self.st

    def init_time(self):
        self.st = cv2.getTickCount()

    def get_timediff(self):
        ed = cv2.getTickCount()
        diff = (ed - self.st) / cv2.getTickFrequency() * 1000
        self.st = ed
        return diff


def main():

    videopath = "/home/spiral/work/pyslam/videos/kitti00/video.mp4"

    dataset = LoadDataset(videopath, 0)

    # Read intrinsic parameters
    K = np.load("./camera_matrix/kitti/mtx.npy")
    D = np.load("./camera_matrix/kitti/dist.npy")
    print("# [main] >>> Load camera params: K= {}, D= {}".format(K, D))

    # Camera info
    w = dataset.width
    h = dataset.height
    fps = dataset.fps
    cam = PinholeCamera(w, h, K, D, fps)
    print("# [main] >>> Init Camera")

    resize_scale = 1.0

    # FAST detector / ORB descriptor / BF tracker
    num_features = 2000
    feature_tracker = DescriptorFeatureTracker(num_features)
    print("# [main] >>> Init Descriptor and Feature Tracker")

    # BF matcher
    matcher = BfFeatureMatcher()

    flg_initialized = False
    f_cur = None
    f_ref = None
    kps_cur = None
    des_cur = None

    ave_n_kps = 0
    ave_n_match = 0
    ave_count = 0

    timer = calcTime()
    time_prev = 0

    end_img_id = 500

    img_id = 0
    while True:
        time_prev = cv2.getTickCount()
        
        img_id += 1
        if img_id > end_img_id:
            print("# [main] Imgid={}, so finished".format(end_img_id))
            break

        img = dataset.getImage(img_id)

        if img is not None:
            ave_count += 1
            # Copy from previous frame
            f_ref = f_cur
            kps_ref = kps_cur
            des_ref = des_cur

            img = cv2.resize(img, dsize=(int(w/resize_scale), int(h/resize_scale)))

            kps_cur, des_cur = feature_tracker.detectAndCompute(img)
            print("# [main] detect kpts n= {}".format(len(kps_cur)))
            ave_n_kps += len(kps_cur)

            if flg_initialized == False:
                flg_initialized = True
                continue

            if len(kps_cur) != 0:
                kps_data = np.array([[x.pt[0], x.pt[1], x.octave, x.size, x.angle] for x in kps_cur], dtype=np.float32)
                kps_array = kps_data[:, :2]
                octaves = np.uint32(kps_data[:, 2])
                sizes = kps_data[:, 3]
                angles = kps_data[:, 4]
                print("# [main] angles= {}".format(angles))
            else:
                flg_initialized = False
                continue

            idxs_cur, idxs_ref = matcher.match(des_cur, des_ref)
            print("# [main] match idx cur= {}, ref= {}".format(len(idxs_cur), len(idxs_ref)))
            ave_n_match += len(idxs_cur)

            time_curr = cv2.getTickCount()
            time_diff = (time_curr - time_prev) / cv2.getTickFrequency() # [s]
            print("# [main] time for featmatch: {} [s]".format(time_diff))

            img_draw = cv2.drawKeypoints(img, kps_cur, None, color=(0,255,0), flags=0)

            if len(idxs_cur) == 0 or len(idxs_ref) == 0:
                flg_initialized = False
                continue

            for idx_cur, idx_ref in zip(idxs_cur, idxs_ref):
                kps_cur0 = kps_cur[idx_cur]
                kps_ref0 = kps_ref[idx_ref]
                pt_cur = (int(kps_cur0.pt[0]), int(kps_cur0.pt[1]))
                pt_ref = (int(kps_ref0.pt[0]), int(kps_ref0.pt[1]))
                img_draw = cv2.line(img_draw, pt_cur, pt_ref, (0,0,255), 2)

            cv2.imshow('Camera+FP', img_draw)
            time_diff = (time_curr - time_prev) / cv2.getTickFrequency() # [s]
            print("# [main] time for draw: {} [s]".format(time_diff))
        
        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            break

    ave_n_kps /= ave_count
    ave_n_match /= ave_count
    print("# [main] average kps= {:.2f}, matched= {:.2f} for total_frame= {}/{}".format(
        ave_n_kps, ave_n_match, ave_count, img_id))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()