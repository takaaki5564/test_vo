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


import OpenGL.GL as gl
import pangolin


def hamming_distance(a, b):
    #r = (1 << np.arange(8))[:,None]
    #return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)   
    return np.count_nonzero(a!=b)

def hamming_distances(a, b):
    return np.count_nonzero(a!=b,axis=1)
    
        
class KeyPointFilterTypes(Enum):
    NONE         = 0
    SAT          = 1      # sat the number of features (keep the best N features: 'best' on the basis of the keypoint.response)
    KDT_NMS      = 2      # Non-Maxima Suppression based on kd-tree
    SSC_NMS      = 3      # Non-Maxima Suppression based on https://github.com/BAILOOL/ANMS-Codes
    OCTREE_NMS   = 4      # Distribute keypoints by using a octree (as a matter of fact, a quadtree): from ORBSLAM2
    GRID_NMS     = 5      # NMS by using a grid 


class Viewer3DMapElement(object): 
    def __init__(self):
        self.cur_pose = None 
        self.predicted_pose = None 
        self.reference_pose = None 
        self.poses = [] 
        self.points = [] 
        self.colors = []         
        self.covisibility_graph = []
        self.spanning_tree = []        
        self.loops = []            



def import_from(module, name, method=None):
    try:      
        imported_module = __import__(module, fromlist=[name])
        imported_name = getattr(imported_module, name)
        if method is None: 
            return imported_name
        else:
            return getattr(imported_name, method) 
    except: 
        if method is not None: 
            name = name + '.' + method 
        return None   



# [4x4] homogeneous inverse T^-1 from T represented with [3x3] R and [3x1] t  
def inv_poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R.T
    ret[:3, 3] = -R.T @ t
    return ret     

# [4x4] homogeneous inverse T^-1 from [4x4] T     
def inv_T(T):
    ret = np.eye(4)
    R_T = T[:3,:3].T
    t   = T[:3,3]
    ret[:3, :3] = R_T
    ret[:3, 3] = -R_T @ t
    return ret       

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1.e-10: 
       return v, norm
    return v/norm, norm

def normalize_vector2(v):
    norm = np.linalg.norm(v)
    if norm < 1.e-10: 
       return v
    return v/norm

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    if len(x.shape) == 1:
        return add_ones_1D(x)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# [4x4] homogeneous T from [3x3] R and [3x1] t             
def poseRt(R, t):
    #print("R={}, t={}".format(R, t))
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret   


def triangulate_normalized_points(pose_1w, pose_2w, kpn_1, kpn_2):
    # P1w = np.dot(K1,  M1w) # K1*[R1w, t1w]
    # P2w = np.dot(K2,  M2w) # K2*[R2w, t2w]
    # since we are working with normalized coordinates x_hat = Kinv*x, one has         
    P1w = pose_1w[:3,:] # [R1w, t1w]
    P2w = pose_2w[:3,:] # [R2w, t2w]

    point_4d_hom = cv2.triangulatePoints(P1w, P2w, kpn_1.T, kpn_2.T)
    good_pts_mask = np.where(point_4d_hom[3]!= 0)[0]
    #print("good_pts_mask={}".format(good_pts_mask))
    point_4d = point_4d_hom / point_4d_hom[3] 
    '''    
    if __debug__:
        if False: 
            point_reproj = P1w @ point_4d;
            point_reproj = point_reproj / point_reproj[2] - add_ones(kpn_1).T
            err = np.sum(point_reproj**2)
            print('reproj err: ', err)     
    '''
    #return point_4d.T
    points_3d = point_4d[:3, :].T
    return points_3d, good_pts_mask  


    
class Viewer3DVoElement(object): 
    def __init__(self):
        self.poses = [] 
        self.traj3d_est = []   # estimated trajectory 


class Viewer3D:
    def __init__(self):
        self.kUiWidth = 180
        self.kViewportWidth = 800
        self.kViewpoerHeight = 800
        self.kDefaultPointSize = 2
        
        self.map_state = None
        self.qmap = Queue()
        self.vo_state = None
        self.qvo = Queue()
        self._is_running = Value('i', 1)
        self._is_paused = Value('i', 1)
        self.vp = Process(target=self.viewer_thread, args=(self.qmap, self.qvo, self._is_running, self._is_paused))
        self.vp.daemon = True
        self.vp.start()
        print("##Viewer3D process thread launghed")
    
    def quit(self):
        self._is_running.value = 0
        self.vp.join()
    
    def is_paused(self):
        return (self._is_paused.value == 1)
    
    def viewer_thread(self, qmap, qvo, is_running, is_paused):
        print("##Viewer3D thread start")
        self.viewer_init(self.kViewportWidth, self.kViewpoerHeight)
        while not pangolin.ShouldQuit() and (is_running.value == 1):
            self.viewer_refresh(qmap, qvo, is_paused)
    
    def viewer_init(self, w, h):
        print("##start viewer init")
        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        viewpoint_x = 0
        viewpoint_y = -40
        viewpoint_z = -80
        viewpoint_f = 1000
        
        self.proj = pangolin.ProjectionMatrix(w, h, viewpoint_f, viewpoint_f, w/2, h/2, 0.1, 5000)
        self.look_view = pangolin.ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)
        self.scam = pangolin.OpenGlRenderState(self.proj, self.look_view)
        self.handler = pangolin.Handler3D(self.scam)

        # Create interactive view
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, self.kUiWidth/w, 1.0, -w/h)
        self.dcam.SetHandler(pangolin.Handler3D(self.scam))

        self.panel = pangolin.CreatePanel('ui')
        self.panel.SetBounds(0.0, 1.0, 0.0, self.kUiWidth/w)

        self.do_follow = True
        self.is_following = True

        self.draw_cameras = True

        self.checkboxFollow = pangolin.VarBool('ui.Follow', value=True, toggle=True)
        self.checkboxCam = pangolin.VarBool('ui.Draw Cameras', value=True, toggle=True)
        self.checkboxGrid = pangolin.VarBool('ui.Grid', value=True, toggle=True)
        self.checkboxPause = pangolin.VarBool('ui.Pause', value=False, toggle=True)
        self.int_slider = pangolin.VarInt('ui.Point Size', value=self.kDefaultPointSize)
        
        self.pointSize = self.int_slider.Get()

        self.Twc = pangolin.OpenGlMatrix()
        self.Twc.SetIdentity()

    def viewer_refresh(self, qmap, qvo, is_paused):
        while not qmap.empty():
            self.map_state = qmap.get()
        while not qvo.empty():
            self.vo_state = qvo.get()

        self.do_follow = self.checkboxFollow.Get()
        self.is_grid = self.checkboxGrid.Get()
        self.draw_cameras = self.checkboxCam.Get()

        if self.checkboxPause.Get():
            is_paused.value = 0
        else:
            is_paused.value = 1
        
        self.pointSize = self.int_slider.Get()

        if self.do_follow and self.is_following:
            self.scam.Follow(self.Twc, True)
        elif self.do_follow and not self.is_following:
            self.scam.SetModelViewMatrix(self.look_view)
            self.scam.Follow(self.Twc, True)
            self.is_following = True
        elif not self.do_follow and self.is_following:
            self.is_following = True
        
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        
        self.dcam.Activate(self.scam)

        if self.is_grid:
            Viewer3D.drawPlane()

        # ============
        # draw 3D map
        if self.map_state is not None:
            if self.map_state.cur_pose is not None:
                # draw current pose (blue)
                gl.glColor3f(0.0, 0.0, 1.0)
                gl.glLineWidth(2)
                pangolin.DrawCamera(self.map_state.cur_pose)
                gl.glLineWidth(1)
                self.updateTwc(self.map_state.cur_pose)
        
            if self.map_state.predicted_pose is not None:
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawCamera(self.map_state.predicted_pose)
            
            if len(self.map_state.poses) > 1:
                if self.draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(self.map_state.poses[:])
            
            if len(self.map_state.points) > 0:
                gl.glPointSize(self.pointSize)
                pangolin.DrawPoints(self.map_state.points, self.map_state.colors)
            
            if self.map_state.reference_pose is not None:
                gl.glColor3f(0.5, 0.0, 0.5)
                gl.glLineWidth(2)
                pangolin.DrawCamera(self.map_state.reference_pose)
                gl.glLineWidth(1)


        # ============
        # draw vo
        if self.vo_state is not None:
            if self.vo_state.poses.shape[0] >= 2:
                # draw pose (green)
                if self.draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(self.vo_state.poses[:-1])
            
            if self.vo_state.poses.shape[0] >= 1:
                gl.glColor3f(0.0, 0.0, 1.0)
                current_pose = self.vo_state.poses[-1:]
                pangolin.DrawCameras(current_pose)
                self.updateTwc(current_pose[0])
            
            if self.vo_state.traj3d_est.shape[0] != 0:
                gl.glPointSize(self.pointSize)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawLine(self.vo_state.traj3d_est)
        
        pangolin.FinishFrame()


    def draw_vo(self, vo):
        if self.qvo is None:
            return
        vo_state = Viewer3DVoElement()
        vo_state.poses = array(vo.poses)
        vo_state.traj3d_est = array(vo.traj3d_est).reshape(-1, 3)

        self.qvo.put(vo_state)

    def draw_map(self, slam):
        if self.qmap is None:
            return
        map = slam.map 
        map_state = Viewer3DMapElement()        
        
        if map.num_frames() > 0: 
            map_state.cur_pose = map.get_frame(-1).Twc.copy()
        
        #if slam.tracking.predicted_pose is not None: 
        #    map_state.predicted_pose = slam.tracking.predicted_pose.inverse().matrix().copy()

        if slam.tracking.kf_ref is not None:
            reference_pose = slam.tracking.kf_ref.Twc.copy()
            
        num_map_keyframes = map.num_keyframes()
        keyframes = map.get_keyframes()
        if num_map_keyframes>0:
            for kf in keyframes:
                map_state.poses.append(kf.Twc)  
        map_state.poses = np.array(map_state.poses)
        print("####[util.draw_map] num_map_keyframes={}".format(num_map_keyframes))

        num_map_points = map.num_points()
        if num_map_points>0:
            for i,p in enumerate(map.get_points()):                
                map_state.points.append(p.pt)
                map_state.colors.append(np.flip(p.color))
        map_state.points = np.array(map_state.points)
        map_state.colors = np.array(map_state.colors)/256.
        print("####[util.draw_map] num_map_points={}".format(num_map_points))
        
        # for kf in keyframes:
        #     for kf_cov in kf.get_covisible_by_weight(kMinWeightForDrawingCovisibilityEdge):
        #         if kf_cov.kid > kf.kid:
        #             map_state.covisibility_graph.append([*kf.Ow, *kf_cov.Ow])
        #     if kf.parent is not None: 
        #         map_state.spanning_tree.append([*kf.Ow, *kf.parent.Ow])
        #     for kf_loop in kf.get_loop_edges():
        #         if kf_loop.kid > kf.kid:
        #             map_state.loops.append([*kf.Ow, *kf_loop.Ow])                
        # map_state.covisibility_graph = np.array(map_state.covisibility_graph)   
        # map_state.spanning_tree = np.array(map_state.spanning_tree)   
        # map_state.loops = np.array(map_state.loops)                     
                                             
        self.qmap.put(map_state)

    def updateTwc(self, pose):
        self.Twc.m = pose
    
    @staticmethod
    def drawPlane(num_divs=200, div_size=10):
        # plane parallel to x-z
        minx = -num_divs * div_size
        minz = -num_divs * div_size
        maxx = num_divs * div_size
        maxz = num_divs * div_size
        gl.glColor3f(0.7, 0.7, 0.7)
        gl.glBegin(gl.GL_LINES)

        for n in range(2 * num_divs):
            gl.glVertex3f(minx + div_size * n, 0, minz)
            gl.glVertex3f(minx + div_size * n, 0, maxz)
            gl.glVertex3f(minx, 0, minz + div_size * n)
            gl.glVertex3f(maxx, 0, minz + div_size * n)
        gl.glEnd()

class VoStage(Enum):
    NO_IMAGES_YET = 0       # no image received
    GOT_FIRST_IMAGE = 1     # got first image


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
        print("##distorted= {}".format(self.is_distorted))
    
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

    # undistort 2D points
    def undistort_points(self, uvs):
        if self.is_distorted:
            #print("K={} D={}".format(self.K, self.D))
            uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape(uvs.shape[0], 1, 2) # continuous array in memory
            #print("uvs_contiguous.shape=", uvs_contiguous.shape)
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
    
    def is_in_image(self, uv, z):
        return (uv[0] > self.u_min) & (uv[0] < self.u_max) & \
               (uv[1] > self.v_min) & (uv[1] < self.v_max) & (z > 0)

    def are_in_image(self, uvs, zs):
        return (uvs[:, 0] > self.u_min) & (uvs[:, 0] < self.u_max) & \
               (uvs[:, 1] > self.v_min) & (uvs[:, 1] < self.v_max) & (zs > 0)



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

