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

from slam.optimizer import *

from slam.utils import *
from slam.map import *
from slam.frame import *
from slam.feature import *
from slam.tracking import *
from slam.initializer import *
from slam.local_mapping import *


kMaxReprojectionDistanceFrame = 7
kMinNumMatchedFeaturesSearchFrameByProjection = 20
kNumMinInliersPoseOptimizationTrackFrame = 10

kMaxDescriptorDistance = 50 #100
kMatchRatioTestMap = 0.8
kUseMotionModel = False


class SlamState(Enum):
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    LOST=3
    

class Slam:
    def __init__(self, camera, feature_tracker):
        self.camera = camera
        self.init_feature_tracker(feature_tracker)
        self.map = Map()
        self.local_mapping = LocalMapping(self.map) # single thread
        self.tracking = Tracking(self)
    
    def quit(self):
        return
    
    def init_feature_tracker(self, tracker):
        Frame.set_tracker(tracker)
        tracker.matcher.ratio_test = 0.8
    
    def track(self, img, frame_id):
        return self.tracking.track(img, frame_id)


class Tracking():
    def __init__(self, slam_system):
        self.system = slam_system
        self.camera = slam_system.camera
        self.map = slam_system.map
        
        self.local_mapping = slam_system.local_mapping
        
        self.initializer = Initializer()
        
        self.motion_model = MotionModel() # motion model for current frame pose prediction without damping

        self.descriptor_distance_sigma = 100 # for ORB
        self.reproj_err_frame_map_sigma = 3

        self.max_frames_between_kfs = int(slam_system.camera.fps)
        self.min_frames_between_kfs = 0
        
        self.state = SlamState.NO_IMAGES_YET

        self.num_matched_kps = None
        self.num_inliers = None
        self.num_matched_map_points = None
        self.num_kf_ref_tracked_points = None

        self.mask_match = None

        self.pose_is_ok = False
        self.predicted_pose = None
        self.velocity = None

        self.f_cur = None
        self.idxs_cur = None
        self.f_ref = None
        self.idxs_ref = None

        self.kf_ref = None
        self.kf_last = None
        self.kid_last_BA = -1

        self.local_keyframes = []
        self.local_points = []

        self.tracking_history = TrackingHistory()

        self.init_history = True
        self.poses = []
        self.t0_est = None
        self.traj3d_est = []

        self.cur_R = None
        self.cur_t = None


    def track(self, img, frame_id):
        # Register current frame: Detect feature and Compute descriptor
        f_cur = Frame(img, self.camera, id=frame_id)
        self.f_cur = f_cur
        print("============== [slam] frame id={}, state={} ==============".format(f_cur.id, self.state))

        self.idxs_ref = []
        self.idxs_cur = []

        if self.state == SlamState.NO_IMAGES_YET:
            print("### [slam] push first frame in the initializer")
            self.initializer.init(f_cur)
            self.state = SlamState.NOT_INITIALIZED
            return # jumpt to second frame
        
        if self.state == SlamState.NOT_INITIALIZED or self.state == SlamState.LOST:
            print("### [slam] try to initialize")
            # initialize frame: estimate pose, calculate 3D map point coordinates
            if self.state == SlamState.NOT_INITIALIZED:
                initializer_output, initializer_is_ok = self.initializer.initialize(f_cur, img,)
            else:
                initializer_output, initializer_is_ok = self.initializer.relocalize(f_cur, img, init_pose = False)

            if initializer_is_ok:
                kf_ref = initializer_output.kf_ref
                kf_cur = initializer_output.kf_cur
                # add the two initialized frame in the map
                self.map.add_frame(kf_ref)
                self.map.add_frame(kf_cur)
                # add the two initialized frame as keyframe in the map
                self.map.add_keyframe(kf_ref)
                self.map.add_keyframe(kf_cur)
                # set 3D map points to keyframe.points
                kf_ref.init_observations()
                kf_cur.init_observations()
                # add new detected map points to map
                new_pts_count, _, _ = self.map.add_points(
                    initializer_output.pts, None, kf_cur, kf_ref, initializer_output.idxs_cur, initializer_output.idxs_ref, img, do_check=False)
                print("###[slam] map: initialized {} new points".format(new_pts_count))
                # update covisibility graph
                kf_ref.update_connections()
                kf_cur.update_connections()

                # update tracking info
                self.f_cur = kf_cur
                self.f_cur.kf_ref = kf_ref
                self.kf_ref = kf_cur        # set reference keyframe
                self.kf_last = kf_cur       # set last added keyframe
                self.map.local_map.update(self.kf_ref)
                self.state = SlamState.OK

                self.initializer.reset()
            else:
                print("###[slam] initialized NG!!!")
            return # jump to next frame

        # get previous frame in map as reference
        f_ref = self.map.get_frame(-1)
        self.f_ref = f_ref

        # add current frame to map
        print("###[slam] added frame to map")
        self.map.add_frame(f_cur)
        self.f_cur.kf_ref = self.kf_ref

        # reset pose state flag
        self.pose_is_ok = False

        with self.map.update_lock:
            # check for map point replacements in previous frame f_ref
            # (some points might have been replaced by local mapping during point fusion)
            self.f_ref.check_replaced_map_points()

            # set initial guess for current pose optimization
            f_cur.update_pose(f_ref.pose)

            # track camera motion from f_ref to f_cur
            self.track_previous_frame(f_ref, f_cur) # matching

            if not self.pose_is_ok:
                # tracking failed, then track camera motion using kf_ref
                self.track_keyframe(self.kf_ref, f_cur)
            
            if self.pose_is_ok:
               # find matches between local map points and unmatched keypoints of f_cur
               self.track_local_map(f_cur)

        with self.map.update_lock:
            if self.pose_is_ok:
                self.state = SlamState.OK
            else:
                # if pose estimation failed, set current frame as reference frame for next processing
                self.initializer.init(f_cur) 
                self.state = SlamState.LOST
                return

            if self.pose_is_ok: # tracking was successful
                f_cur.clean_vo_map_points()

                #need_new_kf = self.need_new_keyframe(f_cur)
                need_new_kf = True

                if need_new_kf:
                    kf_new = KeyFrame(f_cur, img)
                    self.kf_last = kf_new
                    self.kf_ref = kf_new
                    f_cur.kf_ref = kf_new

                    # do local mapping sequentially (not in separate thread)
                    self.local_mapping.push_keyframe(kf_new)
                    #self.local_mapping.do_local_mapping()
                
                f_cur.clean_outlier_map_points()
            else:
                print("###[slam] pose FAILED")
            
        if self.f_cur.kf_ref is None:
            self.f_cur.kf_ref = self.kf_ref
        
        self.update_history()
        print("#[slam.track] map: %d points, %d keyframes" % (self.map.num_points(), self.map.num_keyframes()))


    # track camera motion of f_cur with reference to f_ref
    def track_previous_frame(self, f_ref, f_cur):

        print("###[slam.track_previous_frame] track previsou frame ")
        is_search_frame_by_projection_failure = False

        # search close match map points between f_ref and f_cur by projection
        print("###[slam.track_previous_frame] search frame by projection")
        search_radius = kMaxReprojectionDistanceFrame
        f_cur.reset_points()
        idxs_ref, idxs_cur, num_found_map_pts = self.search_frame_by_projection(f_ref, f_cur,
                                max_reproj_distance=search_radius,
                                max_descriptor_distance=self.descriptor_distance_sigma)
        self.num_matched_kps = len(idxs_cur)
        print("####[slam.track_previous_frame] keypoints matched1: ref={}, cur={} n_found={}".format(len(idxs_ref), len(idxs_cur), num_found_map_pts))

        # if not enough map point matches consider a larger search radius 
        if self.num_matched_kps < kMinNumMatchedFeaturesSearchFrameByProjection:
            f_cur.remove_frame_views(idxs_cur) # remove matched map point 
            f_cur.reset_points()   # reset all map points of f_cur to [None]
            idxs_ref, idxs_cur, num_found_map_pts = self.search_frame_by_projection(f_ref, f_cur,
                                                                            max_reproj_distance=2*search_radius,
                                                                            max_descriptor_distance=0.5*self.descriptor_distance_sigma)
            self.num_matched_kps = len(idxs_cur)
            print("###[slam.track_previous_frame] keypoints matched2: ref={}, cur={}".format(len(idxs_ref), len(idxs_cur)))
            print("###[slam.track_previous_frame] matched map points in prev frame (wider search): %d " % self.num_matched_kps)

            is_search_frame_by_projection_failure = True

        if self.num_matched_kps < kMinNumMatchedFeaturesSearchFrameByProjection:
            f_cur.remove_frame_views(idxs_cur)
            f_cur.reset_points()
            is_search_frame_by_projection_failure = True
            print('###[slam.track_previous_frame] Not enough matches in search frame by projection: ', self.num_matched_kps)
        else:
            self.idxs_ref = idxs_ref
            self.idxs_cur = idxs_cur

            self.pose_optimization(f_cur)
            num_matched_points = f_cur.clean_outlier_map_points()
            print("###[slam.track_previous_frame] num_matced_map_points= {}".format(num_matched_points))

            if not self.pose_is_ok or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame:
               print('###[slam.track_previous_frame] failure in tracking previous frame, # matched map points: ', self.num_matched_map_points)                    
               self.pose_is_ok = False       

        if is_search_frame_by_projection_failure:
            print("###[slam.track_previous_frame] track reference frame")
            self.track_reference_frame(f_ref, f_cur)


    def search_frame_by_projection(self, f_ref, f_cur,
                    max_reproj_distance=kMaxReprojectionDistanceFrame,
                    max_descriptor_distance=kMaxDescriptorDistance,
                    ratio_test=kMatchRatioTestMap):
        found_pts_count = 0
        idxs_ref = []
        idxs_cur = []
        print("###[slam.search_frame_by_projection] start search_frame_by_projection")

        # get all matched points of f_ref
        matched_ref_idxs = np.flatnonzero((f_ref.points != None) & (f_ref.outliers == False))
        print("###[slam.search_frame_by_projection] matched_ref_idxs n= ", len(matched_ref_idxs))

        if matched_ref_idxs.size == 0:
            print("###[slam.search_frame_by_projection] matched 0 skip !!!!!!!!!!!!!!!!!")
            return np.array(idxs_ref), np.array(idxs_cur), found_pts_count

        # remove none from the array
        matched_ref_points = f_ref.points[matched_ref_idxs]

        # project 3D f_ref points on 2D frame f_cur
        projs, depths = f_cur.project_map_points(matched_ref_points)
        # check if 2D projected points are visible
        is_visible = f_cur.are_in_image(projs, depths)

        kp_ref_octaves = f_ref.octaves[matched_ref_idxs]
        kp_ref_scale_factors = Frame.tracker.scale_factors[kp_ref_octaves]
        radiuses = max_reproj_distance * kp_ref_scale_factors
        kd_idxs = f_cur.kd.query_ball_point(projs, radiuses)          # search near points

        for i, p, j in zip(matched_ref_idxs, matched_ref_points, range(len(matched_ref_points))):
            if not is_visible[j]:
               continue
            
            kp_ref_octave = f_ref.octaves[i]

            best_dist = math.inf
            best_level = -1
            best_k_idx = -1
            best_ref_idx = -1
            mean_des_dist = 0
            cnt_des_dist = 0

            for kd_idx in kd_idxs[j]:
                p_f_cur = f_cur.points[kd_idx]
                if p_f_cur is not None:
                    # check num of observed keyframes for the map point
                    if p_f_cur.num_observations > 0:
                        #print("skip 1, ", p_f_cur.num_observations)
                        continue
                # check ovtaves
                p_f_cur_octave = f_cur.octaves[kd_idx]
                if p_f_cur_octave < (kp_ref_octave - 1) or p_f_cur_octave > (kp_ref_octave + 1):
                    #print("skip 2, octave: cur={}, ref={}".format(p_f_cur_octave, kp_ref_octave))
                    continue
                # check smallest descriptor distance
                descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])
                mean_des_dist += descriptor_dist
                cnt_des_dist += 1
                if descriptor_dist < best_dist:
                    best_dist = descriptor_dist
                    best_k_idx = kd_idx
                    best_ref_idx = i
            # add f_cur with smallest descriptor distance to map point as frame view
            if best_dist < max_descriptor_distance:
                if p.add_frame_view(f_cur, best_k_idx):
                    found_pts_count += 1
                    idxs_ref.append(best_ref_idx)
                    idxs_cur.append(best_k_idx)
            # else:
            #     print("###[slam.search_frame_by_projection] best dist={} > {}".format(best_dist, max_descriptor_distance))

            # if cnt_des_dist != 0:
            #     print("###[slam.search_frame_by_projection] mean_des_dist= {}".format(mean_des_dist / cnt_des_dist))
            # else:
            #     print("###[slam.search_frame_by_projection] matchcnt zero ")

        return np.array(idxs_ref), np.array(idxs_cur), found_pts_count


    def need_new_keyframe(self, f_cur):
        kNumMinObsForKeyFrameDefault = 3
        kThNewKfRefRatio = 0.9
        kNumMinPointsForNewKf = 15

        num_keyframes = self.map.num_keyframes()
        nMinObs = kNumMinObsForKeyFrameDefault
        if num_keyframes <= 2:
            nMinObs = 2  # if just two keyframes then we can have just two observations 
        num_kf_ref_tracked_points = self.kf_ref.num_tracked_points(nMinObs)  # number of tracked points in k_ref
        num_f_cur_tracked_points = f_cur.num_matched_inlier_map_points()     # number of inliers in f_cur 
        print('####[need_new_keyframe] F_id=(%d) #tracked_points: %d, KF_id=(%d) #tracked_points: %d ' %(f_cur.id, num_f_cur_tracked_points, self.kf_ref.id, num_kf_ref_tracked_points))
        
        self.num_kf_ref_tracked_points = num_kf_ref_tracked_points

        # condition 1: more than "max_frames_between_kfs" have passed from last keyframe insertion
        cond1 = f_cur.id >= (self.kf_last.id + self.max_frames_between_kfs) 
        
        # condition 2: more than "min_frames_between_kfs" have passed
        cond2 = (f_cur.id >= (self.kf_last.id + self.min_frames_between_kfs))
                  
        # condition 3: few tracked features compared to reference keyframe 
        #cond3 = (num_f_cur_tracked_points < num_kf_ref_tracked_points * kThNewKfRefRatio) and (num_f_cur_tracked_points > kNumMinPointsForNewKf)
        cond3 = (num_f_cur_tracked_points > kNumMinPointsForNewKf)
        
        ret = (cond1 or cond2 ) and cond3
        print('####[need_new_keyframe] KF conditions: ret=%d (%d %d %d)' % (ret, cond1, cond2, cond3) )

        return ret


    # track camera motion of f_cur w.r.t. f_ref
    # estimate motion by matching keypoint descriptors                    
    def track_reference_frame(self, f_ref, f_cur, name=''):

        if f_ref is None:
            return 

        # find keypoint matches between f_cur and kf_ref   
        idxs_cur, idxs_ref = match_frames(f_cur, f_ref)
        self.num_matched_kps = idxs_cur.shape[0]
        print("###[slam.track_reference_frame] keypoints matched: %d " % self.num_matched_kps)  

        # estimate camera orientation and inlier matches by fitting and essential matrix (see the limitations above)             
        idxs_ref, idxs_cur = self.estimate_pose_by_fitting_ess_mat(f_ref, f_cur, idxs_ref, idxs_cur)      
        print("###[slam.track_reference_frame] matched points after emat est: ref={} cur={}".format(len(idxs_ref), len(idxs_cur)))
                               
        # propagate map point matches from kf_ref to f_cur  (do not override idxs_ref, idxs_cur)
        num_found_map_pts_inter_frame, idx_ref_prop, idx_cur_prop, idx_none = self.propagate_map_point_matches(
                        f_ref, f_cur, idxs_ref, idxs_cur, max_descriptor_distance=self.descriptor_distance_sigma) 
        print("###[slam.track_reference_frame] matched map points in prev frame: %d " % num_found_map_pts_inter_frame)
        print("###[slam.track_reference_frame] idxs_ref={} idxs_cur={} -> idx_ref_prop={}, idx_cur_prop={}".format(
            len(idxs_ref), len(idxs_cur), len(idx_ref_prop), len(idx_cur_prop)))
        #print("###[slam.track_reference_frame] idx_cur_prop={}".format(idx_cur_prop))
                                
        # store tracking info (for possible reuse)
        self.idxs_ref = idxs_ref
        self.idxs_cur = idxs_cur

        # f_cur pose optimization using last matches with kf_ref:  
        # here, we use first guess of f_cur pose and propated map point matches from f_ref (matched keypoints) 
        self.pose_optimization(f_cur, name)

        # update matched map points; discard outliers detected in last pose optimization 
        num_matched_points = f_cur.clean_outlier_map_points()

        print('###[slam.track_reference_frame] num_matched_map_points: %d' % (self.num_matched_map_points) ) 
        if not self.pose_is_ok or self.num_matched_map_points < 10:
            f_cur.remove_frame_views(idxs_cur)
            f_cur.reset_points()
            self.pose_is_ok = False
            print("###[slam.track_reference_frame] Failure in tracking previous frame, so removed all frames and points !!!!")



    def estimate_pose_by_fitting_ess_mat(self, f_ref, f_cur, idxs_ref, idxs_cur):
        kNumMinInliersEssentialMat = 8

        kpn_ref = f_ref.kpsn[idxs_ref]
        kpn_cur = f_cur.kpsn[idxs_cur]

        # Estimate inter frame camera motion
        Mrc, self.mask_match = estimate_pose_ess_mat(
            f_ref.kpsn[idxs_ref], f_cur.kpsn[idxs_cur], method=cv2.RANSAC, prob=0.999, threshold=0.003)

        Mcr = inv_T(Mrc)
        estimated_Tcw = np.dot(Mcr, f_ref.pose)
        
        mask_idxs = (self.mask_match.ravel() == 1)
        self.num_inliers = sum(mask_idxs)
        print("####[estimate_pose_by_fitting_ess_mat] inliers: ", self.num_inliers)
        idxs_ref = idxs_ref[mask_idxs]
        idxs_cur = idxs_cur[mask_idxs]

        if self.num_inliers < kNumMinInliersEssentialMat:
            print("####[estimate_pose_by_fitting_ess_mat] not enought inliers !!")
        else:
            # use the estimated pose as an initial guess for the subsequent pose optimization 
            # set only the estimated rotation (essential mat computation does not provide a scale for the translation, see above) 
            Rcw = estimated_Tcw[:3, :3]
            tcw = f_ref.pose[:3, 3]
            f_cur.update_rotation_and_translation(Rcw, tcw)
        
        return idxs_ref, idxs_cur
    
    def pose_optimization(self, f_cur, name=''):
        print("###[slam.pose_optimization] pose opt start", name)
        pose_before = f_cur.pose.copy()
        pose_opt_error, self.pose_is_ok, self.num_matched_map_points = g2o_pose_optimization(f_cur)
        print("###[slam.pose_optimization] err={}, is_ok={}, num_matched_map_points={}".format(pose_opt_error, self.pose_is_ok, self.num_matched_map_points))

        if not self.pose_is_ok:
            # if current pose optimization failed, reset f_cur pose    
            print("###[slam.pose_optimization] pose_is not OK, so replace by previous pose")
            f_cur.update_pose(pose_before)
        
        return self.pose_is_ok
    

    # track camera motion of f_cur w.r.t. given keyframe
    # estimate motion by matching keypoint descriptors                    
    def track_keyframe(self, keyframe, f_cur): 
        print("####[slam.track_keyframe] start")
        f_cur.update_pose(self.f_ref.pose.copy()) # start pose optimization from last frame pose                    
        self.track_reference_frame(keyframe, f_cur)


    # propagate map point matches from f_ref to f_cur (access frames from tracking thread, no need to lock)
    def propagate_map_point_matches(self, f_ref, f_cur, idxs_ref, idxs_cur,
                                    max_descriptor_distance=0):
        idx_ref_out = []
        idx_cur_out = []
        
        #rot_histo = RotationHistogram()

        # populate f_cur with map points by propagating map point matches of f_ref; 
        # to this aim, we use map points observed in f_ref and keypoint matches between f_ref and f_cur  
        num_matched_map_pts = 0
        num_outlier = 0
        num_none = 0
        idx_none = []
        for i, idx in enumerate(idxs_ref): # iterate over keypoint matches 
            p_ref = f_ref.points[idx]
            if p_ref is None: # we don't have a map point P for i-th matched keypoint in f_ref
                #print("p_ref is None idx={} p_ref={}".format(idx, p_ref))
                num_none += 1
                idx_none.append(idx)
                continue 
            if f_ref.outliers[idx] or p_ref.is_bad: # do not consider pose optimization outliers or bad points 
                #print("###[propagate_map_point_matches] propagate_mp: f_ref is outlier idx={}".format(idx))
                num_outlier += 1
                continue  
            idx_cur = idxs_cur[i]
            p_cur = f_cur.points[idx_cur]
            if p_cur is not None: # and p_cur.num_observations > 0: # if we already matched p_cur => no need to propagate anything  
                print("###[propagate_map_point_matches] propagate_mp: p_cur is not none: {}".format(p_cur))
                continue
            des_distance = p_ref.min_des_distance(f_cur.des[idx_cur])
            if des_distance > max_descriptor_distance:
                print("###[propagate_map_point_matches] propagate_mp: des_distance {} > max {}".format(des_distance, max_descriptor_distance))
                continue 
            if p_ref.add_frame_view(f_cur, idx_cur): # => P is matched to the i-th matched keypoint in f_cur
                num_matched_map_pts += 1
                idx_ref_out.append(idx)
                idx_cur_out.append(idx_cur)
        print("###[propagate_map_point_matches] num_matched={} num_none={} num_outlier={}".format(num_matched_map_pts, num_none, num_outlier))
        #print("###[propagate_map_point_matches] idx_none={}".format(idx_none))
        return num_matched_map_pts, idx_ref_out, idx_cur_out, idx_none




      
    # track camera motion of f_cur w.r.t. the built local map  
    # find matches between {local map points} (points in the built local map) and {unmatched keypoints of f_cur}   
    def track_local_map(self, f_cur): 
        if self.map.local_map.is_empty():
            return 
        print('####[slam.track_local_map] tracking local map...')

        self.update_local_map()
        
        num_found_map_pts, reproj_err_frame_map_sigma, matched_points_frame_idxs = search_map_by_projection(self.local_points, f_cur,
                                    max_reproj_distance=self.reproj_err_frame_map_sigma, #Parameters.kMaxReprojectionDistanceMap, 
                                    max_descriptor_distance=self.descriptor_distance_sigma) # use the updated local map          
        #print('reproj_err_sigma: ', reproj_err_frame_map_sigma, ' used: ', self.reproj_err_frame_map_sigma)        
        print("####[slam.track_local_map] new matched map points in local map: %d " % num_found_map_pts)                   
        print("####[slam.track_local_map] local map points ", self.map.local_map.num_points())         
        
        # f_cur pose optimization 2 with all the matched local map points 
        self.pose_optimization(f_cur,'proj-map-frame')    
        f_cur.update_map_points_statistics()  # here we do not reset outliers; we let them reach the keyframe generation 
                                              # and then bundle adjustment will possible decide if remove them or not;
                                              # only after keyframe generation the outliers are cleaned!
        # print('####[slam.track_local_map] num_matched_map_points: %d' % (self.num_matched_map_points) )
        if not self.pose_is_ok:
            print('####[slam.track_local_map] failure in tracking local map, # matched map points: ', self.num_matched_map_points) 
            self.pose_is_ok = False                                        

    def update_local_map(self):
        self.f_cur.clean_bad_map_points()
        self.kf_ref, self.local_keyframes, self.local_points = self.map.local_map.get_frame_covisibles(self.f_cur)       
        self.f_cur.kf_ref = self.kf_ref  
        

    def update_history(self):
        f_cur = self.map.get_frame(-1)
        self.cur_R = f_cur.pose[:3,:3].T
        self.cur_t = np.dot(-self.cur_R,f_cur.pose[:3,3])
        #self.cur_R = f_cur._pose.Rcw
        #self.cur_t = np.dot(-self.cur_R, f_cur._pose.tcw)
        print("### [slam.update_history] curR={}, curT={}".format(self.cur_R, self.cur_t))
        if (self.init_history is True):
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  # starting translation 
            print("### [slam.update_history] Added t0: {}".format([self.cur_t[0], self.cur_t[1], self.cur_t[2]]))
            self.init_history = False
        if (self.t0_est is not None):
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
            self.traj3d_est.append(p)
            self.poses.append(poseRt(self.cur_R, p))
            print("### [slam.update_history] Added t0: {}".format([self.cur_t[0], self.cur_t[1], self.cur_t[2]]))

