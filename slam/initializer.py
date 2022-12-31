#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from numpy import array, cross
from numpy.linalg import solve, norm
import sys
import time
import math
from enum import Enum
from multiprocessing import Process, Queue, Value
from threading import RLock, Thread

from slam.optimizer import *

from slam.utils import triangulate_normalized_points, poseRt, inv_T
from slam.map import Map, MapPoint
from slam.frame import Frame, match_frames, KeyFrame

kMaxIdDistBetweenIntializingFrames = 5
kFeatureMatchRatioTestInitializer = 0.8
kRansacProb = 0.999
kRansacThresholdNormalized = 0.0003
kInitializerDesiredMedianDepth = 20
kInitializerNumMinTriangulatedPoints = 10

class InitializerOutput(object):
    def __init__(self):    
        self.pts = None    # 3d points [Nx3]
        self.kf_cur = None 
        self.kf_ref = None 
        self.idxs_cur = None 
        self.idxs_ref = None 


class Initializer():
    def __init__(self):
        self.mask_match = None
        self.mask_recover = None
        self.frames = deque(maxlen=20) # container type memory for frames
        self.idx_f_ref = 0
        self.f_ref = None

        self.num_min_features = 10
        self.num_min_triangulated_points = kInitializerNumMinTriangulatedPoints
        self.num_failures = 0
    
    def reset(self):
        self.frames.clear()
        self.f_ref = None
    
    # fit essential matrix E with RANSAC using 5-point algorithm solver
    def estimatePose(self, kpn_ref, kpn_cur):
        E, self.mask_match = cv2.findEssentialMat(
            kpn_cur, kpn_ref, focal=1, pp=(0.,0.), method=cv2.RANSAC, 
            prob=kRansacProb, threshold=kRansacThresholdNormalized)
        _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0.,0.))
        return poseRt(R, t.T) # Trc homogeneous transformation matrix with respect to reference frame

    def init(self, f_cur):
        self.frames.append(f_cur)
        self.f_ref = f_cur
        print("#### [initializer] init f_cur pose={}".format(self.f_ref.pose))

    # initialize with two available images
    def initialize(self, f_cur, img_cur):
        if self.num_failures > 10:
            self.num_min_triangulated_points = 0.5 * kInitializerNumMinTriangulatedPoints
            self.num_failures = 0
            print("#### [initializer] halved min num triangulated features to {}".format(self.num_min_triangulated_points))
        out = InitializerOutput()
        is_ok = False

        if self.f_ref is not None:
            self.f_ref = self.frames[-1]
            print("#### [initializer] using previous frame as reference")
        else:
            print("#### [initializer] f_ref is None")

        f_ref = self.f_ref

        # append current frame
        self.frames.append(f_cur)

        # Exit when current frame don't have enough features
        if f_ref is None or f_cur is None or len(f_ref.kps) < self.num_min_features or len(f_cur.kps) < self.num_min_features:
            if f_ref is not None:
                print("#### [initializer] not enough features in ref: ", len(f_ref.kps))
            else:
                print("f_ref is None")
            if f_cur is not None:
                print("#### [initializer] not enough features in cur: ", len(f_cur.kps))
            else:
                print("#### [initializer] f_cur is None")
            self.num_failures += 1
            return out, is_ok

        # find keypoint matches        
        idxs_cur, idxs_ref = match_frames(f_cur, f_ref, kFeatureMatchRatioTestInitializer) ### feature point match
        print("#### [initializer] init frames: {}, {}".format(f_cur.id, f_ref.id))
        print("#### [initializer] keypoint matched: {}, {}".format(len(idxs_cur), idxs_cur))

        # estimation pose of f_cur with respect to f_ref
        Trc = self.estimatePose(f_ref.kpsn[idxs_ref], f_cur.kpsn[idxs_cur]) # relative pose of f_cur to f_ref
        Tcr = inv_T(Trc)
        f_ref.update_pose(np.eye(4))
        f_cur.update_pose(Tcr)
        print("#### [initializer] set new pose: ref={} cur={}".format(np.eye(4), Tcr))

        mask_idxs = (self.mask_match.ravel() == 1)
        self.num_inliers = sum(mask_idxs)
        print("#### [initializer] keypoint inliers: {}".format(self.num_inliers))
        idx_cur_inliers = idxs_cur[mask_idxs]
        idx_ref_inliers = idxs_ref[mask_idxs]

        map = Map()
        f_ref.reset_points()
        f_cur.reset_points()

        kf_ref = KeyFrame(f_ref)

        kf_cur = KeyFrame(f_cur, img_cur) # if image added, feature point detected
        map.add_keyframe(kf_ref)
        map.add_keyframe(kf_cur)

        # calculate 3D point coordinate of matched feature points
        pts3d, mask_pts3d = triangulate_normalized_points(
            kf_cur.Tcw, kf_ref.Tcw, kf_cur.kpsn[idx_cur_inliers], kf_ref.kpsn[idx_ref_inliers])

        new_pts_count, mask_points, _ = map.add_points(pts3d, mask_pts3d, kf_cur, kf_ref, idx_cur_inliers, idx_ref_inliers, img_cur, do_check=True, cos_max_parallax=0.99998)
        print("#### [initializer] triangulate points n= ", new_pts_count)

        if new_pts_count > self.num_min_triangulated_points: # enough number of points
            err = map.optimize(rounds=20, use_robust_kernel=True)
            print("#### [initializer] init optimization error^2: ", err)

            num_map_points = len(map.points)
            print("#### [initializer] map points: ", num_map_points)
            is_ok = num_map_points > self.num_min_triangulated_points

            out.pts = pts3d[mask_points]
            out.kf_cur = kf_cur
            out.idxs_cur = idx_cur_inliers[mask_points]
            out.kf_ref = kf_ref
            out.idxs_ref = idx_ref_inliers[mask_points]

            # set scene median depth to equal desired_median_depth'
            desired_median_depth = kInitializerDesiredMedianDepth
            median_depth = kf_cur.compute_points_median_depth(out.pts)
            depth_scale = desired_median_depth / median_depth
            print("#### [initializer] forcing current median depth {} to {}".format(median_depth, desired_median_depth))

            out.pts[:,:3] = out.pts[:,:3] * depth_scale # scale points
            tcw = kf_cur.tcw * depth_scale # scale initial baseline
            kf_cur.update_translation(tcw)

        map.delete()

        if is_ok:
            print("#### [initializer] Initializer: OK")
        else:
            self.num_failures += 1
            print("#### [initializer] Initializer: NG !!!!!!!")
        return out, is_ok


    # initialize with two available images    
    def relocalize(self, f_cur, img_cur, init_pose = True):
        print("#### [relocalize] start initialize")
        if self.num_failures > 10:
            self.num_min_triangulated_points = 0.5 * kInitializerNumMinTriangulatedPoints
            self.num_failures = 0
            print("#### [relocalize] halved min num triangulated features to {}".format(self.num_min_triangulated_points))
        out = InitializerOutput()
        is_ok = False

        print("#### [relocalize] initialize1 f_cur pose={}".format(self.f_ref.pose))
        if self.f_ref is not None:
            self.f_ref = self.frames[-1]
            print("#### [relocalize] using previous frame as reference")
        else:
            print("#### [relocalize] f_ref is None")

        print("#### [relocalize] initialize2 f_cur pose={}".format(self.f_ref.pose))
        f_ref = self.f_ref

        # append current frame
        self.frames.append(f_cur)

        # Exit when current frame don't have enough features
        if f_ref is None or f_cur is None or len(f_ref.kps) < self.num_min_features or len(f_cur.kps) < self.num_min_features:
            if f_ref is not None:
                print("#### [relocalize] not enough features in ref: ", len(f_ref.kps))
            else:
                print("f_ref is None")
            if f_cur is not None:
                print("#### [relocalize] not enough features in cur: ", len(f_cur.kps))
            else:
                print("#### [relocalize] f_cur is None")
            self.num_failures += 1
            return out, is_ok

        # find keypoint matches        
        idxs_cur, idxs_ref = match_frames(f_cur, f_ref, kFeatureMatchRatioTestInitializer) ### feature point match
        print("#### [relocalize] init frames: {}, {}".format(f_cur.id, f_ref.id))
        print("#### [relocalize] keypoint matched: {}, {}".format(len(idxs_cur), idxs_cur))

        Trc = self.estimatePose(f_ref.kpsn[idxs_ref], f_cur.kpsn[idxs_cur]) # relative pose of f_cur to f_ref
        Tcr = inv_T(Trc)

        f_ref.update_pose(f_ref.pose.copy())
        f_cur.update_pose(np.dot(f_ref.pose, Tcr))
        print("#### [relocalize] set new pose: ref={} cur={}".format(f_ref.pose, f_cur.pose))

        mask_idxs = (self.mask_match.ravel() == 1)
        self.num_inliers = sum(mask_idxs)
        print("#### [relocalize] keypoint inliers: {}".format(self.num_inliers))
        idx_cur_inliers = idxs_cur[mask_idxs]
        idx_ref_inliers = idxs_ref[mask_idxs]

        map = Map()
        f_ref.reset_points()
        f_cur.reset_points()

        kf_ref = KeyFrame(f_ref)

        kf_cur = KeyFrame(f_cur, img_cur) # if image added, feature point detected
        map.add_keyframe(kf_ref)
        map.add_keyframe(kf_cur)

        print("#### [relocalize] kf_ref.Tcw,={}".format(kf_ref.Tcw))
        print("#### [relocalize] kf_cur.Tcw,={}".format(kf_cur.Tcw))

        # calculate 3D point coordinate of matched feature points
        pts3d, mask_pts3d = triangulate_normalized_points(kf_cur.pose, kf_ref.pose, kf_cur.kpsn[idx_cur_inliers], kf_ref.kpsn[idx_ref_inliers])

        new_pts_count, mask_points, _ = map.add_points(pts3d, mask_pts3d, kf_cur, kf_ref, idx_cur_inliers, idx_ref_inliers, img_cur, do_check=True, cos_max_parallax=0.99998)
        print("#### [relocalize] triangulate points: ", new_pts_count)

        if new_pts_count > self.num_min_triangulated_points:
            err = map.optimize(rounds=20, use_robust_kernel=True)
            print("#### [relocalize] init optimization error^2: ", err)

            num_map_points = len(map.points)
            print("#### [relocalize] map points: ", num_map_points)
            is_ok = num_map_points > self.num_min_triangulated_points

            out.pts = pts3d[mask_points]
            out.kf_cur = kf_cur
            out.idxs_cur = idx_cur_inliers[mask_points]
            out.kf_ref = kf_ref
            out.idxs_ref = idx_ref_inliers[mask_points]

            desired_median_depth = kInitializerDesiredMedianDepth
            median_depth = kf_cur.compute_points_median_depth(out.pts)
            depth_scale = desired_median_depth / median_depth
            print("#### [relocalize] forcing current median depth {} to {}".format(median_depth, desired_median_depth))

            out.pts[:,:3] = out.pts[:,:3] * depth_scale # scale points
            tcw = kf_cur.tcw * depth_scale # scale initial baseline
            kf_cur.update_translation(tcw)

        map.delete()

        if is_ok:
            print("#### [relocalize] Initializer: OK")
        else:
            self.num_failures += 1
            print("#### [relocalize] Initializer: NG !!!!!!!")
        return out, is_ok
 