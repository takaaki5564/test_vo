#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from cv2 import aruco
import numpy as np
from numpy import array, cross
from numpy.linalg import solve, norm
import sys
import time
import math
from enum import Enum
from multiprocessing import Process, Queue, Value
from threading import RLock, Thread

from slam.slam import *
from slam.utils import * #LoadDataset, PinholeCamera, Viewer3D, KeyPointFilterTypes, hamming_distance, hamming_distances
from slam.map import * #Map
from slam.frame import * #Frame, KeyFrame
from slam.feature import * #DescriptorFeatureTracker
from slam.tracking import * #Tracking

#import logging
#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)7s %(message)s")
#log = logging.getLogger(__name__)

import OpenGL.GL as gl
import pangolin


def main():

    videopath = "/home/spiral/work/pyslam/videos/kitti00/video.mp4"
    #videopath = "/home/spiral/work/dataset/spiral_inside1.mp4"
    #videopath = 0 # camera id

    skip_frame = 240
    dataset = LoadDataset(videopath, skip_frame)

    # Read intrinsic parameters
    K = np.load("./logicool720p/mtx.npy")
    D = np.load("./logicool720p/dist.npy")

    # Camera info
    w = dataset.width
    h = dataset.height
    fps = dataset.fps
    cam = PinholeCamera(w, h, K, D, fps)
    print("# [main] >>> Initialized Camera")

    # FAST detector / ORB descriptor / BF tracker
    num_features = 2000
    feature_tracker = DescriptorFeatureTracker(num_features)
    print("# [main] >>> Initialized Descriptor and Feature Tracker")

    # Initialize system for SLAM
    slam = Slam(cam, feature_tracker)
    print("# [main] >>> Initialized SLAM")

    is_draw_traj_img = False
    is_draw_3d = True
    is_draw_2d = True

    if is_draw_3d:
        viewer3D = Viewer3D()
    else:
        viewer3D = None
        
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5*traj_img_size)
    draw_scale = 3.0

    img_id = 0
    while True:
        img = dataset.getImage(img_id)
        img = cv2.resize(img, dsize=(int(w/2), int(h/2)))

        if img is not None:
            slam.track(img, img_id)
            
            if is_draw_3d:
                viewer3D.draw_map(slam)
            
            img_draw = slam.map.draw_feature_trails(img)

            if is_draw_2d:
                cv2.imshow('Camera+FP', img_draw)

            if img_id > 2 and is_draw_traj_img:
                x, y, z = slam.tracking.traj3d_est[-1]                
                draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
                cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
                # write text on traj_img
                cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                cv2.imshow('Trajectory', traj_img)
                print("# [main] Traj: x=%2fm y=%2fm z=%2fm" % (x, y, z))

        ley = cv2.waitKey(100)
        if  0xFF == ord('q'):
            if viewer3D is not None:
                viewer3D.quit()
            break
        img_id += 1

    slam.quit()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()