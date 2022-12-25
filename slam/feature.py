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

ORBextractor = import_from('orbslam2_features', 'ORBextractor')  


class VoStage(Enum):
    NO_IMAGES_YET = 0       # no image received
    GOT_FIRST_IMAGE = 1     # got first image


class FeatureTrackingResult(object): 
    def __init__(self):
        self.kps_ref = None          # all reference keypoints (numpy array Nx2)
        self.kps_cur = None          # all current keypoints   (numpy array Nx2)
        self.idxs_ref = None         # indexes of matches in kps_ref so that kps_ref_matched = kps_ref[idxs_ref]  (numpy array of indexes)
        self.idxs_cur = None         # indexes of matches in kps_cur so that kps_cur_matched = kps_cur[idxs_cur]  (numpy array of indexes)
        self.kps_ref_matched = None  # reference matched keypoints, kps_ref_matched = kps_ref[idxs_ref]
        self.kps_cur_matched = None  # current matched keypoints, kps_cur_matched = kps_cur[idxs_cur]



# class FeatureDetector:
#     def __init__(self, num_features, quality_level = 0.01, min_corner_distance = 3):
#         self.num_features = num_features
#         self.quality_level = quality_level
#         self.min_corner_distance = min_corner_distance
#         self.blockSize = 5
    
#     def detect(self, frame, mask=None):
#         pts = cv2.goodFeaturesToTrack(frame, self.num_features, self.quality_level, self.min_corner_distance, blockSize=self.blockSize, mask=mask)
#         if pts is not None:
#             kps = [cv2.KeyPoint(p[0][0], p[0][1], self.blockSize) for p in pts]
#         else:
#             kps = []
#         return kps



class FeatureMatcherTypes(Enum):
    NONE = 0
    BF = 1     
    FLANN = 2

kSigmaLevel0 = 1.0                        # default value; can be changed by selected feature        
kRatioTest = 0.7
class BfFeatureMatcher:
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check=False):
        self.cross_check = cross_check
        self.matches = []
        self.ratio_test = kRatioTest
        self.matcher = cv2.BFMatcher(norm_type, cross_check)
        self.matcher_name = ''
    
    def match(self, des1, des2, ratio_test=None):
        matches = self.matcher.knnMatch(des1, des2, k=2)
        self.matches = matches
        return self.goodMatches(matches, des1, des2, ratio_test)
    
    # This function returns matches where each trainIdx is associated to only one query Idx index (not doubled)
    def goodMatches(self, matches, des1, des2, ratio_test=None):
        len_des2 = len(des2)
        idx1, idx2 = [], []
        if ratio_test is None:
            ratio_test = self.ratio_test
        if matches is not None:
            float_inf = float('inf')
            dist_match = defaultdict(lambda: float_inf)
            index_match = dict()
            for m, n in matches:
                if m.distance > ratio_test * n.distance:
                    continue
                dist = dist_match[m.trainIdx]
                if dist == float_inf:
                    dist_match[m.trainIdx] = m.distance
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    index_match[m.trainIdx] = len(idx2) - 1
                else:
                    if m.distance < dist:
                        index = index_match[m.trainIdx]
                        assert(idx2[index] == m.trainIdx)
                        idx1[index] = m.queryIdx
                        idx2[index] = m.trainIdx
        return idx1, idx2

class DescriptorFeatureTracker:
    def __init__(self, num_features, num_levels = 3, scale_factor = 1.2):
        # detector & descriptor type = ORB
        self.min_features = 50
        self.num_features = num_features
        self.sigma_level0 = 1.0
        self.num_levels = num_levels

        # init FAST detector
        self.use_pyramid_adaptor = self.num_levels > 1
        self.need_nms = self.num_levels > 1
        self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS
        self.do_keypoints_size_rescaling = True

        # init pyramid adaptor
        self.scale_factors = np.zeros(num_levels)
        self.scale_factors[0] = 1.0
        self.sigma0 = 1.0
        self.scale_factor = 1.2
        self.num_features = 2000
        self.pyramid_imgs = []
        self.inv_level_sigmas2 = []

        self.init_sigma_levels()

        # init descriptor
        self.orb_params = dict(nfeatures = num_features,
                               scaleFactor = self.scale_factor,
                               nlevels = self.num_levels,
                               patchSize = 31,
                               edgeThreshold = 10,
                               fastThreshold = 20,
                               firstLevel = 0,
                               WTA_K = 2,
                               scoreType=cv2.ORB_FAST_SCORE)
        self.feature_descriptor = cv2.ORB_create(**self.orb_params)
        self.max_descriptor_distance = 100

        # set descriptor distance HAMMING NORM
        self.descriptor_distance = hamming_distance
        self.descriptor_distances = hamming_distances

        # init NF matcher
        self.matcher = BfFeatureMatcher()

    
    # Initialize scale factors, sigma for each octave level
    def init_sigma_levels(self):
        kNumLevelsInitSigma = 40
        kSigmaLevel0 = 1.0
        print("num levels: ", self.num_levels)
        num_levels = max(kNumLevelsInitSigma, self.num_levels)
        self.inv_scale_factor = 1./self.scale_factor
        self.scale_factors = np.zeros(num_levels)
        self.level_sigmas2 = np.zeros(num_levels)
        self.level_sigmas = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.inv_level_sigmas2 = np.zeros(num_levels)
        self.log_scale_factor = math.log(self.scale_factor)

        self.scale_factors[0] = 1.0
        self.level_sigmas2[0] = self.sigma_level0*self.sigma_level0
        self.level_sigmas[0] = math.sqrt(self.level_sigmas2[0])

        for i in range(1, num_levels):
            self.scale_factors[i] = self.scale_factors[i-1]*self.scale_factor
            self.level_sigmas2[i] = self.scale_factors[i]*self.scale_factors[i]*self.level_sigmas2[0]
            self.level_sigmas[i] = math.sqrt(self.level_sigmas2[i])
        for i in range(num_levels):
            self.inv_scale_factors[i] = 1.0 / self.scale_factors[i]
            self.inv_level_sigmas2[i] = 1.0 / self.level_sigmas2[i]

    
    def detectAndCompute(self, frame, mask=None, filter=True):
        if frame.ndim > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kps, des = self.feature_descriptor.detectAndCompute(frame, mask)
        print("####[detectAndCompute] des= {}".format(des))
        return kps, des

    
    def pyramid_compute(self, frame):
        inv_scale = 1./self.scale_factor
        self.imgs = []
        self.img_filtered = []

        pyr_cur = frame
        for i in range(0, self.num_levels):
            self.imgs.append(pyr_cur)
            if i < self.num_levels - 1:
                pyt_down = cv2.resize(pyt_cur, (0,0), fx=inv_scale, fy=inv_scale)
                pyt_cur = pyt_down

    def track(self, image_ref, image_cur, kps_ref):
        kps_cur, des_cur = self.detectAndCompute(image_cur)
        kps_cur = np.array([x.pt for x in kps_cur], dtype=np.float32)



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

