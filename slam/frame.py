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
from scipy.spatial import cKDTree

from collections import Counter, OrderedDict
import g2o

from slam.utils import LoadDataset, PinholeCamera, Viewer3D, KeyPointFilterTypes, hamming_distance, hamming_distances

# colors from https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
myjet = np.array([[0.5       , 0.5       , 1.        ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])


class CameraPose:
    def __init__(self, pose=None):
        if pose is None:
            pose = g2o.Isometry3d()
            #print("#### [frame.camera_pose] set pose: {}".format(pose))
        self.set(pose)
        self.covariance = np.identity(6)
    
    def set(self, pose):
        #print("#### [frame.camera_pose] pose={}".format(pose))
        if isinstance(pose, g2o.SE3Quat) or isinstance(pose, g2o.Isometry3d):
            self._pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self._pose = g2o.Isometry3d(pose)
        self.Tcw = self._pose.matrix()  # homogeneous transformation matrix (4,4)
        self.Rcw = self.Tcw[:3,:3]
        self.tcw = self.Tcw[:3,3]       # pc = Rcw * pw + tcw
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)    # origin of camera in world coordinate

    def update(self,pose):
        self.set(pose)
        
    @property    
    def isometry3d(self):  # pose as g2o.Isometry3d 
        return self._pose 
    
    @property    
    def quaternion(self): # g2o.Quaternion(),  quaternion_cw  
        return self._pose.orientation()  
    
    @property    
    def orientation(self): # g2o.Quaternion(),  quaternion_cw  
        return self._pose.orientation()     
    
    @property    
    def position(self):    # 3D vector tcw (world origin w.r.t. camera frame) 
        return self._pose.position()        
    
    def get_rotation_angle_axis(self):
        angle_axis = g2o.AngleAxis(self._pose.orientation())
        #angle = angle_axis.angle()
        #axis = angle_axis.axis()  
        return angle_axis  
    
    def get_inverse_matrix(self):
        return self._pose.inverse().matrix()

    def set_from_rotation_and_translation(self, Rcw, tcw): 
        self.set(g2o.Isometry3d(g2o.Quaternion(Rcw), tcw))     


# Frame mainly collects keypoints, descriptors and their corresponding 3D points
class Frame:
    _id = 0
    _id_lock = RLock()

    tracker = None
    #feature_detector = None
    feature_matcher = None

    def __init__(self, img, camera, pose=None, id=None, timestamp=None, kps_data=None):
        self._lock_features = RLock()
        self._lock_pose = RLock()

        self.is_keyframe = False
        self.camera = camera

        if pose is None:
            self._pose = CameraPose()
            print("####[Frame] set default camerapose")
        else:
            self._pose = CameraPose(pose)
            print("####[Frame] set specified camerapose")
        
        if id is not None:
            self.id = id
        else:
            with Frame._id_lock:
                self.id = Frame._id
                Frame._id += 1

        # Keypoints information arrays
        self.kps = None             # keepoint coordinates
        self.kpsu = None            # undistorted keypoint coordinates
        self.kpsn = None            # normalized keypoint coordinates
        self.octaves = None         # keypoint octaves
        self.sizes = None           # keypoint sizes
        self.angles = None          # keypoint angles
        self.des = None             # keypoint descriptors

        # map points information arrays 
        self.points = None  # map points
        self.outliers = None

        self.kf_ref = None
        self.img = None

        # Detect keypoints and Compute descriptor
        if img is not None:
            self.img = img.copy()
            if kps_data is None:
                self.kps, self.des = Frame.tracker.detectAndCompute(img)
                print("####[Frame] detectAndCompute keypoints n={}".format(len(self.kps)))
                if len(self.kps) != 0:
                    kps_data = np.array([[x.pt[0], x.pt[1], x.octave, x.size, x.angle] for x in self.kps], dtype=np.float32)
                    self.kps = kps_data[:, :2]
                    self.octaves = np.uint32(kps_data[:, 2])
                    #print("### set frame octaves={}".format(self.octaves))
                    self.sizes = kps_data[:, 3]
                    self.angles = kps_data[:, 4]
            else:
                print("#### [Frame] kps_data is not None, skip initialize kps in frame")
                pass
            if len(self.kps) != 0:
                self.kpsu = self.camera.undistort_points(self.kps)
                self.kpsn = self.camera.unproject_points(self.kpsu)
                self.points = np.array([None]*len(self.kpsu))
                self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool)

    def __hash__(self):
        return self.id   
    
    def __eq__(self, rhs):
        return (isinstance(rhs, Frame) and self.id == rhs.id)
    
    def __lt__(self, rhs):
        return self.id < rhs.id 
    
    def __le__(self, rhs):
        return self.id <= rhs.id         

    @property
    def width(self):
        return self.camera.width
    
    @property
    def height(self):
        return self.camera.height    
                
    @property    
    def isometry3d(self):  # pose as g2o.Isometry3d 
        with self._lock_pose:          
            return self._pose.isometry3d
                    
    @property
    def Tcw(self):
        with self._lock_pose:           
            return self._pose.Tcw  
    @property
    def Twc(self):
        with self._lock_pose:           
            return self._pose.get_inverse_matrix()      
    @property
    def Rcw(self):
        with self._lock_pose:           
            return self._pose.Rcw       
    @property
    def Rwc(self):
        with self._lock_pose:           
            return self._pose.Rwc     
    @property
    def tcw(self):
        with self._lock_pose:           
            return self._pose.tcw       
    @property
    def Ow(self):
        with self._lock_pose:           
            return self._pose.Ow          
    @property
    def pose(self):
        with self._lock_pose:   
            return self._pose.Tcw      
    @property    
    def quaternion(self): # g2o.Quaternion(),  quaternion_cw  
        with self._lock_pose:           
            return self._pose.quaternion  
    @property    
    def orientation(self): # g2o.Quaternion(),  quaternion_cw  
        with self._lock_pose:           
            return self._pose.orientation  
                
    @property    
    def position(self):    # 3D vector tcw (world origin w.r.t. camera frame) 
        with self._lock_pose:           
            return self._pose.position    
               
          
    # update pose from transformation matrix or g2o.Isometry3d
    def update_pose(self, pose):
        with self._lock_pose:              
            self._pose.set(pose)    

   # update pose from transformation matrix 
    def update_translation(self, tcw):
        with self._lock_pose:              
            self._pose.set_translation(tcw)           
   # update pose from transformation matrix 
    def update_rotation_and_translation(self, Rcw, tcw):
        with self._lock_pose:          
            self._pose.set_from_rotation_and_translation(Rcw, tcw)            
                
    # transform a world point into a camera point 
    def transform_point(self, pw):
        with self._lock_pose:          
            return (self._pose.Rcw @ pw) + self._pose.tcw # p w.r.t. camera 
    # transform a world points into camera points [Nx3] 
    # out: points  w.r.t. camera frame  [Nx3] 
    def transform_points(self, points):    
        with self._lock_pose:          
            Rcw = self._pose.Rcw
            tcw = self._pose.tcw.reshape((3,1))
            print("#### [Frame] Rcw={}, tcw={}, points len={}".format(Rcw, tcw, len(points)))       
            return (Rcw @ points.T + tcw).T  # get points  w.r.t. camera frame  [Nx3]      

    def num_tracked_points(self, minObs = 1):
        with self._lock_features:          
            num_points = 0
            for i,p in enumerate(self.points):
                if p is not None and not p.is_bad: 
                    if p.num_observations >= minObs:  
                        num_points += 1   
            return num_points 

    def num_matched_inlier_map_points(self):
        with self._lock_features:          
            num_matched_points = 0
            for i,p in enumerate(self.points):
                if p is not None and not self.outliers[i]: 
                    if p.num_observations > 0:
                        num_matched_points += 1             
            return num_matched_points

    # project an [Nx3] array of map point vectors on this frame 
    # out: [Nx2] array of image points, [Nx1] array of map point depths   
    def project_points(self, points):                   
        pcs = self.transform_points(points)      
        return self.camera.project(pcs)

    # project a list of N MapPoint objects on this frame
    # out: Nx2 image points, [Nx1] array of map point depths 
    def project_map_points(self, map_points):
        points = np.array([p.pt for p in map_points])
        return self.project_points(points)

    # project a 3d point vector pw on this frame 
    # out: image point, depth      
    def project_point(self, pw):                
        pc = self.transform_point(pw) # p w.r.t. camera 
        return self.camera.project(pc)   
    # project a MapPoint object on this frame
    # out: image point, depth 
    def project_map_point(self, map_point):           
        return self.project_point(map_point.pt)  
    
    def is_in_image(self, uv, z): 
        return self.camera.is_in_image(uv,z)        
                
    # input: [Nx2] array of uvs, [Nx1] of zs 
    # output: [Nx1] array of visibility flags             
    def are_in_image(self, uvs, zs):
        return self.camera.are_in_image(uvs,zs)   
    
    # input: map_point
    # output: visibility flag, projection uv, depth z
    def is_visible(self, map_point):
        #with self._lock_pose:    (no need, project_map_point already locks the pose)   
        uv,z = self.project_map_point(map_point)
        PO = map_point.pt-self.Ow
                        
        if not self.is_in_image(uv, z):
            return False, uv, z

        dist3D = np.linalg.norm(PO)   
        # point depth must be inside the scale pyramid of the image
        if dist3D < map_point.min_distance or dist3D > map_point.max_distance:
            return False, uv, z
        # viewing angle must be less than 60 deg
        if np.dot(PO,map_point.get_normal()) < 0.5 * dist3D:
            return False, uv, z
        return True, uv, z 

    # input: a list of map points 
    # output: [Nx1] array of visibility flags, [Nx2] array of projections, [Nx1] array of depths, [Nx1] array of distances PO
    # check a) points are in image b) good view angle c) good distance range  
    def are_visible(self, map_points):
        points = []
        point_normals = []
        min_dists = []
        max_dists = []
        for i, p in enumerate(map_points):
            points.append(p.pt)
            point_normals.append(p.get_normal())
            min_dists.append(p.min_distance)
            max_dists.append(p.max_distance)
        points = np.array(points)
        point_normals = np.array(point_normals)
        min_dists = np.array(min_dists)
        max_dists = np.array(max_dists)
        
        #with self._lock_pose:  (no need, project_points already locks the pose) 
        uvs, zs = self.project_points(points)    
        POs = points - self.Ow 
        dists   = np.linalg.norm(POs, axis=-1, keepdims=True)    
        POs /= dists
        cos_view = np.sum(point_normals * POs, axis=1)
                
        are_in_image = self.are_in_image(uvs, zs)     
        are_in_good_view_angle = cos_view > 0.5     
        dists = dists.reshape(-1,)              
        are_in_good_distance = (dists > min_dists) & (dists < max_dists)        
            
        out_flags = are_in_image & are_in_good_view_angle & are_in_good_distance
        return out_flags, uvs, zs, dists        

    @staticmethod
    def set_tracker(tracker):
        Frame.tracker = tracker
        #Frame.feature_detector = tracker.feature_detector # instead of feature manager
        Frame.feature_matcher = tracker.matcher
        Frame.descriptor_distance = tracker.descriptor_distance
        Frame.descriptor_distances = tracker.descriptor_distances
        Frame._id = 0


    # KD tree of undistorted keypoints
    @property
    def kd(self):
        if not hasattr(self, '_kd'):
            self._kd = cKDTree(self.kpsu)
        return self._kd

               
    def get_point_match(self, idx):
        with self._lock_features:          
            return self.points[idx] 
        
    def set_point_match(self, p, idx):
        with self._lock_features:  
            self.points[idx] = p 

    def remove_point_match(self, idx):       
        with self._lock_features:  
            self.points[idx] = None 
            
    def replace_point_match(self, p, idx):                
        self.points[idx] = p    # replacing is not critical (it does not create a 'None jump')
        
    def remove_point(self, p):
        with self._lock_features:          
            try: 
                p_idxs = np.where(self.points == p)[0]  # remove all instances 
                self.points[p_idxs] = None        
            except:
                pass 
            
    def remove_frame_views(self, idxs):
        with self._lock_features:    
            if len(idxs) == 0:
                return 
            for idx,p in zip(idxs,self.points[idxs]): 
                if p is not None: 
                    p.remove_frame_view(self,idx)

    def reset_points(self):
        with self._lock_features:          
            self.points = np.array([None]*len(self.kpsu))
            self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool)   

    def get_matched_points(self):
        with self._lock_features:                   
            matched_idxs = np.flatnonzero(self.points!=None) 
            matched_points = self.points[matched_idxs]            
            return matched_points #, matched_idxs 

    def count_notnone(self): # for debug
        n_notnone = 0
        if self.points is None:
            print("#### [Frame] frame points is None")
        else:
            for i, p in enumerate(self.points):
                if p is not None:
                    n_notnone += 1
            print("#### [Frame] frame points not-none={} / {}".format(n_notnone, len(self.points)))

    # update found count for map points        
    def update_map_points_statistics(self):
        with self._lock_features:           
            num_matched_points = 0
            for i,p in enumerate(self.points):
                if p is not None and not self.outliers[i]: 
                        p.increase_found() # update point statistics 
                        if p.num_observations > 0:
                            num_matched_points +=1
            return num_matched_points            

    def clean_vo_map_points(self):
        with self._lock_features:
            num_cleaned_points = 0
            n_notnone = 0
            n_none = 0
            for i, p in enumerate(self.points):
                if p is not None:
                    n_notnone += 1
                    if p.num_observations < 1:
                        self.points[i] = None
                        self.outliers[i] = False
                        num_cleaned_points += 1
                else:
                    n_none += 1
            print("#### [Frame] #cleaned vo map points: {}, none={}, not-none={}".format(num_cleaned_points, n_none, n_notnone))


    # reset outliers detected in last pose optimization       
    def clean_outlier_map_points(self):
        with self._lock_features:
            num_matched_points = 0
            num_outliers = 0
            num_out_observations = 0
            for i,p in enumerate(self.points):
                if p is not None:
                    if self.outliers[i]: 
                        p.remove_frame_view(self,i)
                        self.points[i] = None 
                        self.outliers[i] = False
                        p.last_frame_id_seen = self.id
                        num_outliers += 1
                    else:
                        if p.num_observations > 0:
                            num_matched_points +=1
                        else:
                            num_out_observations += 1
            print("####[clean_outlier_map_points] num_outlier={}/{}, out_frame={}".format(
                num_outliers, len(self.points), num_out_observations))
            return num_matched_points    

    # reset bad map points and update visibility count          
    def clean_bad_map_points(self):
        with self._lock_features:           
            for i,p in enumerate(self.points):
                if p is not None: 
                    if p.is_bad: 
                        p.remove_frame_view(self,i)         
                        self.points[i] = None 
                        self.outliers[i] = False      
                    else:
                        p.last_frame_id_seen = self.id   
                        p.increase_visible()   

    # check for point replacements 
    def check_replaced_map_points(self):
        with self._lock_features:  
            num_replaced_points = 0
            if self.points is not None: 
                print("#### [frame]: try to replace points n={}".format(len(self.points)))
                for i,p in enumerate(self.points):      
                    if p is not None: 
                        replacement = p.get_replacement()
                        if replacement is not None: 
                            replaced = p
                            self.points[i] = replacement   
                            del replaced        
                            num_replaced_points +=1
            print('#### [frame]: replaced points: n=', num_replaced_points)    

    def reset_points(self):
        with self._lock_features:
            if self.kpsu is not None:
                self.points = np.array([None]*len(self.kpsu))
                self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool)

    def get_points(self):
        with self._lock_features:          
            return self.points.copy()    


    def draw_feature_trails(self, img, kps_idxs, trail_max_length=9):
        img = img.copy()
        with self._lock_features:
            if self.kps is None:
                print("#### [Frame] --- no kps, so skip drowing")
                return img
            if self.points is None:
                print("#### [Frame] --- no points, so skip drowing")
                return img
            else:
                print("#### [Frame] --- draw kps : n={}".format(len(self.kps)))
            uvs = np.rint(self.kps[kps_idxs]).astype(np.intp)

            n_none = 0
            n_bad = 0
            for kp_idx in kps_idxs:
                uv = tuple(uvs[kp_idx])
                radius = kDrawFeatureRadius[self.octaves[kp_idx]]

                point = self.points[kp_idx]
                if point is not None and not point.is_bad:
                    p_frame_views = point.frame_views()
                    # there is a correspoinding 3D points
                    if len(p_frame_views) > 2:
                        cv2.circle(img, uv, color=(0, 255, 0), radius=radius, thickness=1)
                    else:
                        cv2.circle(img, uv, color=(255, 0, 0), radius=radius, thickness=1)
                    # draw the trail
                    pts = []
                    lfid = None
                    for f, idx in p_frame_views[-trail_max_length:][::-1]:
                        if lfid is not None and lfid-1 != f.id:
                            break
                        pts.append(tuple(map(lambda x: int(round(x)), f.kps[idx])))
                        lfid = f.id
                    if len(pts) > 1:
                        #print("#### [Frame] draw matched point traj: {}".format(pts))
                        #cv2.polylines(img, np.array([pts], dtype=np.int32), False, myjet[len(pts)]*255, thickness=1, lineType=16)
                        cv2.polylines(img, np.array([pts], dtype=np.int32), False, (255,255,255), thickness=1, lineType=16)
                    else:
                        cv2.circle(img, uv, color=(0, 0, 0), radius=2)
                else:
                    cv2.circle(img, uv, color=(0, 0, 255), radius=2, thickness=-1)
                    if point is None:
                        n_none += 1
                    else:
                        n_bad += 1
            print("#### [Frame] ### frame: skip kps, bad={}, none={}/ all={}".format(n_bad, n_none, len(self.points)))
            return img

    def draw_all_feature_trails(self, img):
        kps_idxs = range(len(self.kps))
        return self.draw_feature_trails(img, kps_idxs)

    def get_matched_good_points(self):       
        with self._lock_features:               
            good_points = [p for p in self.points if p is not None and not p.is_bad]         
            return good_points    
    
kDrawFeatureRadius = [r*5 for r in range(1,100)]
kDrawOctaveColor = np.linspace(0, 255, 12)

class KeyFrameGraph:
    def __init__(self):
        self._lock_connections = RLock()
        self.init_parent = False
        self.parent = None
        self.children = set()
        self.loop_edges = set()
        self.connected_keyframes_weights = Counter()
        self.ordered_keyframes_weights = OrderedDict()
        self.is_first_connection = True
    

    # ===============================    
    # spanning tree     
    def add_child(self, keyframe):
        with self._lock_connections:
            self.children.add(keyframe)
    
    def erase_child(self, keyframe):
        with self._lock_connections:
            try: 
                self.children.remove(keyframe)     
            except:
                pass 
            
    def set_parent(self, keyframe):
        with self._lock_connections:
            if self == keyframe: 
                return 
            self.parent = keyframe 
            keyframe.add_child(self)
        
    def get_children(self):
        with self._lock_connections:
            return self.children.copy()
                    
    def get_parent(self):
        with self._lock_connections:
            return self.parent        
        
    def has_child(self, keyframe):            
        with self._lock_connections:
            return keyframe in self.children        
        
    # Loop edge
    def add_loop_edge(self, keyframe):
        with self._lock_connections:
            self.not_to_erase = True
            self.loop_edges.add(keyframe)

    def get_loop_edges(self):
        with self._lock_connections:
            return self.loop_edges.copy()

    # get an ordered list of covisible keyframes     
    def get_covisible_keyframes(self):
        with self._lock_connections:                  
            return list(self.ordered_keyframes_weights.keys()) # returns a copy   

    def add_connection(self, keyframe, weigth):
        with self._lock_connections: 
            self.connected_keyframes_weights[keyframe]=weigth
            self.update_best_covisibles()

    def update_best_covisibles(self):
        with self._lock_connections:         
            self.ordered_keyframes_weights = OrderedDict(sorted(self.connected_keyframes_weights.items(), key=lambda x: x[1], reverse=True)) # order by value (decreasing order)
     
    # get an ordered list of covisible keyframes     
    def get_best_covisible_keyframes(self,N):
        with self._lock_connections:                  
            return list(self.ordered_keyframes_weights.keys())[:N] # returns a copy       



class KeyFrame(Frame, KeyFrameGraph):
    def __init__(self, frame, img=None):
        KeyFrameGraph.__init__(self)
        Frame.__init__(self, img=None, camera=frame.camera, pose=frame.pose, id=frame.id)
        print("##[KeyFrame class] init id={}".format(self.id))

        if frame.img is not None:
            self.img = frame.img
        else:
            if img is not None:
                self.img = img.copy()

        self.map = None

        self.is_keyframe = True
        self.kid = None           # keyframe id 
        
        self._is_bad = False 
        self.to_be_erased = False 
        
        # pose relative to parent (this is computed when bad flag is activated)
        self._pose_Tcp = CameraPose() 

        # share keypoints info with frame (these are computed once for all on frame initialization and they are not changed anymore)
        self.kps     = frame.kps      # keypoint coordinates                  [Nx2]
        self.kpsu    = frame.kpsu     # [u]ndistorted keypoint coordinates    [Nx2]
        self.kpsn    = frame.kpsn     # [n]ormalized keypoint coordinates     [Nx2] (Kinv * [kp,1])    
        self.octaves = frame.octaves  # keypoint octaves                      [Nx1]
        self.sizes   = frame.sizes    # keypoint sizes                        [Nx1] 
        self.angles  = frame.angles   # keypoint angles                       [Nx1] 
        self.des     = frame.des      # keypoint descriptors                  [NxD] where D is the descriptor length 
        
        if hasattr(frame, '_kd'):     
            self._kd = frame._kd 
        else: 
            self._kd = cKDTree(self.kpsu)
    
        # map points information arrays (copy points coming from frame)
        self.points   = frame.get_points()     # map points => self.points[idx] is the map point matched with self.kps[idx] (if is not None)
        self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool)     # used just in propagate_map_point_matches()   

    @property
    def is_bad(self):
        with self._lock_connections:
            return self._is_bad     

    # associate matched map points to observations
    def init_observations(self):
        with self._lock_features:
            for idx,p in enumerate(self.points):
                if p is not None and not p.is_bad:
                    if p.add_observation(self, idx):
                        p.update_info()

    def update_connections(self):
        kMinNumOfCovisiblePointsForCreatingConnection = 15
        # for all map points of this keyframe check in which other keyframes they are seen
        # build a counter for these other keyframes    
        points = self.get_matched_good_points()
        assert len(points) > 0
        viewing_keyframes = [kf for p in points for kf in p.keyframes() if kf.kid != self.kid] # exclude this keyframe 
        viewing_keyframes = Counter(viewing_keyframes)   
        if not viewing_keyframes: # if empty   (https://www.pythoncentral.io/how-to-check-if-a-list-tuple-or-dictionary-is-empty-in-python/)
            return 
        # order the keyframes 
        covisible_keyframes = viewing_keyframes.most_common() 
        # get keyframe that shares most points 
        kf_max, w_max = covisible_keyframes[0]
        # if the counter is greater than threshold add connection
        # otherwise add the one with maximum counter
        with self._lock_connections:
            if w_max >= kMinNumOfCovisiblePointsForCreatingConnection:
                self.connected_keyframes_weights = viewing_keyframes 
                self.ordered_keyframes_weights = OrderedDict()
                for kf,w in covisible_keyframes:
                    if w >= kMinNumOfCovisiblePointsForCreatingConnection:
                        kf.add_connection(self,w)
                        self.ordered_keyframes_weights[kf] = w
                    else:
                        break 
            else:
                self.connected_keyframes_weights = Counter({kf_max,w_max}) 
                self.ordered_keyframes_weights = OrderedDict({kf_max,w_max})    
                kf_max.add_connection(self,w_max)        
            # update spanning tree                     
            if self.is_first_connection and self.kid!=0: 
                self.set_parent(kf_max)
                self.is_first_connection = False 
        #print('ordered_keyframes_weights: ', self.ordered_keyframes_weights)                               
                


def match_frames(f1, f2, ratio_test=None):
    idx1, idx2 = Frame.feature_matcher.match(f1.des, f2.des, ratio_test)
    print("#### [Frame.match_frames] match frame idx1 n={}".format(len(idx1)))
    print("#### [Frame.match_frames] match frame idx2 n={}".format(len(idx2)))
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    return idx1, idx2
