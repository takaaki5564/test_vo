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
from slam.utils import *

#import logging
#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)7s %(message)s")
#log = logging.getLogger(__name__)

import OpenGL.GL as gl
import pangolin


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


class PinholeCamera:
    def __init__(self, width, height, K, D, fps = 1):
        self.width = width
        self.height = height
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.D = np.array(D, dtype=np.float32)
        self.fps = fps
        self.is_distorted = norm(self.D) > 1e-10
        self.K = K
        self.Kinv = np.array([[1/self.fx, 0, -self.cx/self.fx], [0, 1/self.fy, -self.cy/self.fy], [0, 0, 1]])
        self.u_min, self.u_max = 0, self.width
        self.v_min, self.v_max = 0, self.height
        self.initialized = False
        self.init()
    
    def init(self):
        if not self.initialized:
            self.initialized = True
            self.undistort_image_bounds()
    
    # project 3D points to 2D image points
    # [out] Nx2 image points, [Nx1] map point depths
    def project(self, xcs):
        projs = self.K @ xcs.T
        zs = projs[-1]
        projs = projs[:2] / zs # assume depth=1 on image plane
        return projs.T, zs

    # unproject 2D point to 3D points on depth=1
    def unproject(self, uv):
        x = (uv[0] - self.cx) / self.fx
        y = (uv[1] - self.cy) / self.fy
        return x, y
    
    # unproject 2D point to 3D normalized coordinates
    def unproject_points(self, uvs):
        return np.dot(self.Kinv, add_ones(uvs).T).T[:, 0:2]

    # undistort 2D points using camera distortion parameter
    def undistort_points(self, uvs):
        if self.is_distorted:
            #print("K={} D={}".format(self.K, self.D))
            uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape(uvs.shape[0], 1, 2) # continuous array in memory
            uvs_undistorted = cv2.undistortPoints(uvs_contiguous, self.K, self.D, None, self.K)
            return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
        else:
            return uvs

    def undistort_image_bounds(self):
        uv_bounds = np.array([[self.u_min, self.v_min],
                                [self.u_min, self.v_max],
                                [self.u_max, self.v_min],
                                [self.u_max, self.v_max]], dtype=np.float32).reshape(4,2)
        if self.is_distorted:
            #print("K={} D={}".format(self.K, self.D))
            uv_bounds_undistorted = cv2.undistortPoints(np.expand_dims(uv_bounds, axis=1), self.K, self.D, None, self.K)      
            uv_bounds_undistorted = uv_bounds_undistorted.ravel().reshape(uv_bounds_undistorted.shape[0], 2)
        else:
            uv_bounds_undistorted = uv_bounds 
        self.u_min = min(uv_bounds_undistorted[0][0],uv_bounds_undistorted[1][0])
        self.u_max = max(uv_bounds_undistorted[2][0],uv_bounds_undistorted[3][0])        
        self.v_min = min(uv_bounds_undistorted[0][1],uv_bounds_undistorted[2][1])    
        self.v_max = max(uv_bounds_undistorted[1][1],uv_bounds_undistorted[3][1])  

class FeatureDetector:
    def __init__(self, num_features, quality_level = 0.01, min_corner_distance = 3):
        self.num_features = num_features
        self.quality_level = quality_level
        self.min_corner_distance = min_corner_distance
        self.blockSize = 5
    
    def detect(self, frame, mask=None):
        pts = cv2.goodFeaturesToTrack(
            frame, self.num_features, self.quality_level, self.min_corner_distance, blockSize=self.blockSize, mask=mask)
        # convert matrix of points into list of keypoints
        if pts is not None:
            kps = [cv2.KeyPoint(p[0][0], p[0][1], self.blockSize) for p in pts]
        else:
            kps = []
        return kps

class FeatureTracker:
    def __init__(self, num_features, num_levels = 3, scale_factor = 1.2):
        self.min_features = 50
        self.num_features = num_features
        self.feature_detector = FeatureDetector(num_features)

        self.lk_params = dict(winSize = (21, 21), 
                              maxLevel = num_levels, 
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    def detect(self, frame, mask=None):
        return self.feature_detector.detect(frame, mask)

    def track(self, image_ref, image_cur, kps_ref):
        kps_cur, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, kps_ref, None, **self.lk_params)
        st = st.reshape(st.shape[0])
        res = FeatureTrackingResult()
        res.idxs_ref = [i for i,v in enumerate(st) if v==1]
        res.idxs_cur = res.idxs_ref.copy()
        res.kps_ref_matched = kps_ref[res.idxs_ref]
        res.kps_cur_matched = kps_cur[res.idxs_cur]
        res.kps_ref = res.kps_ref_matched
        res.kps_cur = res.kps_cur_matched
        return res

class VisualOdometry:
    def __init__(self, cam, feature_tracker):
        self.cam = cam
        self.cur_image = None
        self.prev_image = None
        self.stage = VoStage.NO_IMAGES_YET

        self.kps_ref = None
        self.kps_cur = None

        self.cur_R = np.eye(3, 3)
        self.cur_t = np.zeros((3, 1))

        self.feature_tracker = feature_tracker
        self.track_result = None
        self.init_history = True
        self.poses = []
        self.t0_est = None
        self.traj3d_est = []

        self.num_matched_kps = None
        self.num_inliers = None
        
        self.kUseEssentialMatrixEstimation = True
        self.kRansacThresholdNormalized = 0.0003  # metric threshold used for normalized image coordinates 
        self.kRansacThresholdPixels = 0.1         # pixel threshold used for image coordinates 
        self.kUseEssentialMatrixEstimation = True # using the essential matrix fitting algorithm is more robust RANSAC given five-point algorithm solver 
        self.kRansacProb = 0.999

    def getAbsoluteScale(self, frame_id):
        # if ground truth is there, get current scale
        return 0.5
    
    def computeFundamentalMatrix(self, kps_ref, kps_cur):
        F, mask = cv2.findFundamentalMat(kps_ref, kps_cur, cv2.FM_RANSAC, param1=kRansacThresholdPixels, param2=kRansacProb)
        if F is None or F.shape == (1, 1):
            # no Fmat found
            raise Exception('No fundamental matrix found')
        elif F.shape[0] > 3:
            F = F[0:3, 0:3]
        return np.matrix(F), mask

    def estimatePose(self, kps_ref, kps_cur):
        #print("kps_ref={}, kps_cur={}".format(kps_ref, kps_cur))
        kp_ref_u = self.cam.undistort_points(kps_ref)
        kp_cur_u = self.cam.undistort_points(kps_cur)
        self.kpn_ref = self.cam.unproject_points(kp_ref_u)
        self.kpn_cur = self.cam.unproject_points(kp_cur_u)
        if self.kUseEssentialMatrixEstimation:
            # five-point algorithm is more robust for degenerate planar case
            E, self.mask_match = cv2.findEssentialMat(
                self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=self.kRansacProb, threshold=self.kRansacThresholdNormalized)
        else:
            # for test
            F, self.mask_match = self.computeFundamentalMatrix(kp_cur_u, kp_ref_u)
            E = self.cam.K.T @ F @ self.cam.K
        _, R, t, mask = cv2.recoverPose(E, self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.))
        return R, t
    
    def processFirstFrame(self):
        # detect keypoints for current image
        self.kps_ref = self.feature_tracker.detect(self.cur_image)
        # convert from list to keypoints
        self.kps_ref = np.array([x.pt for x in self.kps_ref], dtype=np.float32)

        if self.kps_ref is not None and self.kps_ref.shape[0] > 0:
            #print("kps_ref shape={}".format(self.kps_ref.shape[0]))
            self.draw_img = self.drawFeatureTracks(self.cur_image)

    def processFrame(self, frame_id):
        # track features
        self.track_result = self.feature_tracker.track(self.prev_image, self.cur_image, self.kps_ref)
        # estimate pose
        R, t = self.estimatePose(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)
        # update keypoints history
        self.kps_ref = self.track_result.kps_ref
        self.kps_cur = self.track_result.kps_cur
        self.num_matched_kps = self.kpn_ref.shape[0]
        self.num_inliers = np.sum(self.mask_match)
        print("##matched points: {}, # inliers= {}".format(self.num_matched_kps, self.num_inliers))
        print("##estimated norm= {}".format(norm(t)))

        absolute_scale = self.getAbsoluteScale(frame_id)
        self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
        self.cur_R = self.cur_R.dot(R)

        # draw image
        self.draw_img = self.drawFeatureTracks(self.cur_image)

        # check if number of features is enough
        if (self.kps_ref.shape[0] < self.feature_tracker.num_features):
            self.kps_cur = self.feature_tracker.detect(self.cur_image)
            # convert list to keypoints
            self.kps_cur = np.array([x.pt for x in self.kps_cur], dtype=np.float32)
            print("##detected points: {}".format(self.kps_cur.shape[0]))
        
        self.kps_ref = self.kps_cur
        self.updateHistory()

    def track(self, img, frame_id):
        print("##frame: {}".format(frame_id))
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        assert(img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width), "Frame provided image has not the same size in configuration"
        self.cur_image = img
        if (self.stage == VoStage.GOT_FIRST_IMAGE):
            print("##processFrame")
            self.processFrame(frame_id)
        elif (self.stage == VoStage.NO_IMAGES_YET):
            print("##processFirstFrame")
            self.processFirstFrame()
            if self.kps_ref.shape[0] > self.feature_tracker.min_features:
                self.stage = VoStage.GOT_FIRST_IMAGE
        self.prev_image = self.cur_image

    def drawFeatureTracks(self, image, reinit = False):
        draw_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        num_outliers = 0
        if (self.stage == VoStage.GOT_FIRST_IMAGE):
            if reinit:
                for p1 in self.kps_cur:
                    a, b = p1.ravel()
                    cv2.circle(draw_img, (a, b), 2, (0, 255, 0), -1)
            else:
                for i, pts in enumerate(zip(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)):
                    # except outlier
                    if self.mask_match[i]:
                        p1, p2 = pts
                        a, b = p1.astype(int).ravel()
                        c, d = p2.astype(int).ravel()
                        cv2.line(draw_img, (a, b), (c, d), (0, 255, 0), 1)
                        cv2.circle(draw_img, (a, b), 1, (0, 0, 255), -1)
                    else:
                        num_outliers += 1
        elif self.kps_ref is not None:
            for p1 in self.kps_ref:
                a, b = p1.astype(int).ravel()
                cv2.circle(draw_img, (a, b), 1, (0, 255, 0), -1)
        return draw_img

    def updateHistory(self):
        if (self.init_history is True):
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  # starting translation 
            self.init_history = False 
        if (self.t0_est is not None):
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
            self.traj3d_est.append(p)
            self.poses.append(poseRt(self.cur_R, p))   


class LoadDataset():
    def __init__(self, path, skip_frame=0):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.skip_frame = skip_frame

        if not self.cap.isOpened():
            raise IOError('Cannot open video file: ', self.path)
        else:
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.Ts = 1./self.fps
            print("##num frames: {}".format(self.num_frames))
            print("##fps: {}".format(self.fps))

        self.is_init = False

        if skip_frame > 0:
            for _ in range(skip_frame):
                ret, image = self.cap.read()

    def getImage(self, frame_id):
        if self.is_init is False and frame_id + self.skip_frame > 0:
            self.is_init = True
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id + self.skip_frame)
        self.is_init = True
        ret, image = self.cap.read()
        if ret is False:
            log.error('ERROR while reading from file: ', self.filename)
        return image


def main():
    # Input video path or camera id
    #videopath = "/home/spiral/work/dataset/202207121609.mp4"
    videopath = 4 # camera id

    # Load dataset
    skip_frame = 240
    dataset = LoadDataset(videopath, skip_frame)

    # Read intrinsic parameters
    K = np.load("./logicool720p/mtx.npy")
    D = np.load("./logicool720p/dist.npy")

    # Camera info object
    w = dataset.width
    h = dataset.height
    fps = dataset.fps
    cam = PinholeCamera(w, h, K, D, fps)

    # initialize detector and tracker:
    # SHI_TOMASI feature detector and
    # Optical flow with Lucas-Kanade method
    num_features = 2000
    feature_tracker = FeatureTracker(num_features)

    vo = VisualOdometry(cam, feature_tracker)

    is_draw_traj_img = True
    is_draw_3d = True

    viewer3D = Viewer3D()

    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5*traj_img_size)
    draw_scale = 3.0

    img_id = 0
    while True:
        # Load image
        img = dataset.getImage(img_id)

        if img is not None:
            # Process Visual Odometry
            vo.track(img, img_id)

            # Drawing process
            if img_id > 2:
                x, y, z = vo.traj3d_est[-1]
                if is_draw_traj_img:
                    draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
                    cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
                    # write text on traj_img
                    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                    cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                    cv2.imshow('Trajectory', traj_img)
                if is_draw_3d:
                    viewer3D.draw_vo(vo)
            cv2.imshow('Camera', vo.draw_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        img_id += 1

if __name__ == '__main__':
    main()