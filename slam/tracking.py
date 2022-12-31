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
from collections import defaultdict

from slam.utils import *
from slam.map import *
from slam.frame import *

#import logging
#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)7s %(message)s")
#log = logging.getLogger(__name__)

import OpenGL.GL as gl
import pangolin


class SlamState(Enum):
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    LOST=3


class VoStage(Enum):
    NO_IMAGES_YET = 0       # no image received
    GOT_FIRST_IMAGE = 1     # got first image


class TrackingHistory():
    def __init__(self):
        self.relative_frame_poses = []  # list of relative frame poses as g2o.Isometry3d() (see camera_pose.py)
        self.kf_references = []         # list of reference keyframes  
        self.timestamps = []            # list of frame timestamps 
        self.slam_states = []           # list of slam states 




def estimate_pose_ess_mat(kpn_ref, kpn_cur, method=cv2.RANSAC, prob=0.999, threshold=0.0003):	
    # here, the essential matrix algorithm uses the five-point algorithm solver by D. Nister (see the notes and paper above )     
    E, mask_match = cv2.findEssentialMat(kpn_cur, kpn_ref, focal=1, pp=(0., 0.), method=method, prob=prob, threshold=threshold)                         
    _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0., 0.))   
    return poseRt(R,t.T), mask_match  # Trc, mask_mat         

class MotionModel():
    def __init__(self):
        initial_position = None
        initial_orientation = None
        initial_covariance = None

        self.is_ok = False
        self.initialized = False

        self.delta_position = np.zeros(3)
        self.delta_orientation = g2o.Quaternion()
    
    def current_pose(self):
        return (g2o.Isometry3d(self.orientation, self.position), self.covariance)
    
    def predict_pose(self, prev_position=None, prev_orientation=None):
        # Predict next camera pose
        if prev_position is not None:
            self.position = prev_position
        if prev_orientation is not None:
            self.orientation = prev_orientation
        
        if not self.initialized:
            return (g2o.Isometry3d(self.orientation, self.position), self.covariance)
        
        orientation = self.delta_orientation * self.orientation
        position = self.position + self.delta_orientation * self.delta_position

        return (g2o.Isometry3d(orientation, position), self.covariance)
    
    def update_pose(self, new_position, new_orientation, new_covariance=None):
        if initialized:
            self.delta_position = new_position - self.position
            self.delta_orientation = new_orientation * self.orientation.inverse()
            self.delta_orientation.normalize()
        self.position = new_position
        self.orientation = new_orientation
        self.covariane = new_covariance
        self.initialized = True

class SLAMDynamicConfig:
    def __init__(self):
        self.descriptor_distance_sigma = None



class RotationHistogram:
    def __init__(self, histogram_length = 30):
        self.histogram_length = histogram_length
        self.factor = 1.0 / histogram_length
        self.histo = [ [] for i in range(self.histogram_length)]

    def push(sef, rot, idx):
        if rot < 0.0:
            rot += 360.0
        bin = int(round(rot * self.factor))
        if bin == self.histogram_length:
            bin = 0
        assert(bin >= 0 and bin < self.histogram_length)
        self.histo[bin].append(dx)

    def compute_3_max(self):
        max1 = max2 = max3 = 0.
        ind1 = ind2 = ind3 = -1
        for i in range(self.histogram_length):
            s = len(self.histo[i])
            if (s > max1):
                max3 = max2
                
    
    def get_invalid_idxs(self):
        ind1, ind2, ind3 = self.compute_3_max()



