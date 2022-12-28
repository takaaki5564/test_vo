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
from slam.map import MapPoint

import g2o
import OpenGL.GL as gl
import pangolin

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

class Draw3D:

    def __init__(self):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 1000),
            pangolin.ModelViewLookAt(-2, 1, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640/480.0)
        self.dcam.SetHandler(pangolin.Handler3D(self.scam))

        self.pose = np.identity(4)

        self.axis_x = [[-1,0,0],[1,0,0]]
        self.axis_y = [[0,-1,0],[0,1,0]]
        self.axis_z = [[0,0,-1],[0,0,1]]

    def show(self, rvec, tvec, trajectory, traj_scale, map_points, map_color):
        # Panglon show srart
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # Draw coordinates
        gl.glLineWidth(2)
        gl.glColor3f(0.0, 0.0, 0.0)
        pangolin.DrawLine(self.axis_x)
        gl.glColor3f(0.0, 0.0, 0.0)
        pangolin.DrawLine(self.axis_y)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawLine(self.axis_z)

        # Draw points to positive direction
        gl.glPointSize(10)
        gl.glColor3f(1.0, 0.0, 0.0)
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        pangolin.DrawPoints(points, colors)

        # Draw map points
        if len(map_points) > 0:
            gl.glPointSize(2)
            gl.glColor3f(1.0, 0.0, 0.0)
            map_points = np.array(map_points)*traj_scale
            colors = np.array(map_color)
            pangolin.DrawPoints(map_points, colors)

        # Draw current camera
        if rvec is not None and tvec is not None:
            self.pose[:3, 3] = np.array(tvec[:3, 0])*traj_scale
            #self.pose[:3, 3] = np.array(tvec[:3])*traj_scale
            self.pose[:3, :3] = np.array(rvec)
        else:
            self.pose = np.identity(4)
            self.pose[:3, 3] = np.array([0, 0, 0])

        gl.glColor3f(0.0, 0.0, 1.0) # blue
        gl.glLineWidth(2)
        pangolin.DrawCamera(self.pose, 0.25, 0.3, 0.4)

        # Draw trajectory
        if len(trajectory) > 0:
            pangolin.DrawLine(trajectory)

        pangolin.FinishFrame()


def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret   

# turn [[x,y]] -> [[x,y,1]]
def add_ones_1D(x):
    return np.array([x[0], x[1], 1])

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    if len(x.shape) == 1:
        return add_ones_1D(x)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# unproject 2D point to 3D normalized coordinates
def unproject_points(uvs, Kinv):
    return np.dot(Kinv, add_ones(uvs).T).T[:, 0:2]

# undistort 2D points
def undistort_points(uvs, K, D):
    if np.linalg.norm(D) > 1e-10:
        uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape(uvs.shape[0], 1, 2) # continuous array in memory
        uvs_undistorted = cv2.undistortPoints(uvs_contiguous, K, D, None, K)
        return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
    else:
        return uvs

def inv_T(T):
    ret = np.eye(4)
    R_T = T[:3,:3].T
    t   = T[:3,3]
    ret[:3, :3] = R_T
    ret[:3, 3] = -R_T @ t
    return ret       

def triangulate_normalized_points(pose1, pose2, kpn1, kpn2):
    P1w = pose1[:3, :] # [R1, t1]
    P2w = pose2[:3, :] # [R2, t2]
    # Reconstrcut points (4xN)
    point_4d_hom = cv2.triangulatePoints(P1w, P2w, kpn1.T, kpn2.T)
    good_pts_mask = np.where(point_4d_hom[3] != 0)[0]
    point_4d = point_4d_hom / point_4d_hom[3] # from homogeneous coord to cartesian coord
    if False: # debug
        point_reproj = P1w @ point_4d
        point_reproj = point_reproj / point_reproj[2] - add_ones(kpn1).T
        err = np.sum(point_reproj**2)
        print("# [triangulate_normalized_points] Reproj err= {}".format(err))
    points_3d = point_4d[:3, :].T
    return points_3d, good_pts_mask

def img_range_elem(ranges, i):
    return ranges[:, i]

def main():

    videopath = "/home/spiral/work/pyslam/videos/kitti00/video.mp4"

    dataset = LoadDataset(videopath, 0)

    # Read intrinsic parameters
    K = np.load("./camera_matrix/kitti/mtx.npy")
    D = np.load("./camera_matrix/kitti/dist.npy")
    print("# [main] >>> Load camera params: K= {}, D= {}".format(K, D))

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    print("# [main] fx={}, fy={}, cx={}, cy={}".format(fx, fy, cx, cy))

    Kinv = np.array([[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]])

    threshNumFP = 200
    lk_params = dict(
            winSize=(21,21), 
            maxLevel=3, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Camera info
    w = dataset.width
    h = dataset.height
    print("# [main] w={}, h={}".format(w, h))
    fps = dataset.fps
    cam = PinholeCamera(w, h, K, D, fps)
    print("# [main] >>> Init Camera")

    resize_scale = 1.0

    # 3D draw
    draw3d = Draw3D()

    # FAST detector / ORB descriptor / BF tracker
    num_features = 4000
    feature_tracker = DescriptorFeatureTracker(num_features)
    print("# [main] >>> Init Descriptor and Feature Tracker")

    # BF matcher
    matcher = BfFeatureMatcher()

    flg_initialized = False
    flg_mapadded = False

    good_fp_curr = None
    pts_prev = None

    Pcr_curr = None
    Pcr_ref = None

    pose = g2o.Isometry3d()
    print("# [main] set init pose= {}".format(pose.matrix()))
    Prw = np.eye(4)
    Pcw = np.eye(4)

    Rcw = np.eye(3,3)
    tcw = np.zeros((3,1))

    trajectory = []

    map_points = []
    map_color = []

    gray_curr = None

    timer = calcTime()
    time_prev = 0

    end_img_id = -1

    img_id = 0
    while True:
        time_prev = cv2.getTickCount()
        
        img_id += 1
        if end_img_id > 0 and img_id > end_img_id:
            print("# [main] Imgid={}, so finished".format(end_img_id))
            break

        print("# [main] <<<<< start Imgid= {} <<<<<<".format(img_id))
        gray_prev = gray_curr

        img = dataset.getImage(img_id)
        gray_curr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if flg_initialized == False:
            flg_initialized = True
            continue

        if img is not None:
            # Copy from previous frame
            Prw = Pcw
            fp_prev = good_fp_curr

            # Resize frame
            if resize_scale != 1.0:
                gray_curr = cv2.resize(gray_curr, dsize=(int(w/resize_scale), int(h/resize_scale)))

            # Detect feature points
            if pts_prev is None or len(fp_prev) < threshNumFP:
                print("# [main] ===== initialize feature points again ======= ")
                pts_prev = cv2.goodFeaturesToTrack(
                    gray_prev, mask=None, maxCorners=num_features, qualityLevel=0.01, minDistance=3, blockSize=5)
                flg_mapadded = True
                if pts_prev is not None: 
                    kps_prev = [cv2.KeyPoint(p[0][0], p[0][1], 5) for p in pts_prev]
                else:
                    kps_prev = []
                fp_prev = np.array([x.pt for x in kps_prev], dtype=np.float32) 

                if fp_prev.shape[0] < threshNumFP:
                    flg_initialized = True
                    continue

            fp_curr, st, _ = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_curr, fp_prev, None, **lk_params)

            st = st.reshape(st.shape[0])
            good_fp_prev = fp_prev[st == 1]
            good_fp_curr = fp_curr[st == 1]

            if not good_fp_prev.shape[0] > 0:
                print("# [main] good fps n= {} is too small, so init again".format(good_fp_prev.shape[0]))
                flg_initialized = False
                continue

            flow = good_fp_curr - good_fp_prev

            # Stop pose estimation when flow size is small enough
            th_flowsize = 5
            mean_x, mean_y = np.mean(np.absolute(flow), axis=0)
            print("# [main] mean flowx= {}, flowy= {}".format(mean_x, mean_y))
            if mean_x < th_flowsize and mean_y < th_flowsize:
                flg_initialized = False
                continue

            good_fpu_curr = undistort_points(good_fp_curr, K, D)
            good_fpu_prev = undistort_points(good_fp_prev, K, D)

            good_fpn_curr = unproject_points(good_fpu_curr, Kinv)
            good_fpn_prev = unproject_points(good_fpu_prev, Kinv)

            # Pose estimation
            E, mask_match = cv2.findEssentialMat(
                good_fpn_curr, good_fpn_prev, focal=1, pp=(0.,0.), method=cv2.RANSAC,
                prob=0.999, threshold=0.0003)
            if E is None:
                flg_initialized = False
                continue
            if E.shape[0] != 3:
                E = E[:3, :3]

            # Pose from current frame to previous frame
            _, Rrc, trc, mask = cv2.recoverPose(E, good_fpn_curr, good_fpn_prev, focal=1, pp=(0.,0.))

            if Rrc is None or trc is None:
                flg_initialized = False
                continue
            print("# [main] norm trc= {}".format(np.linalg.norm(trc)))

            # Projection from current camera coord to previous camera coord
            Prc = poseRt(Rrc, trc.T)
            # Projection from previous camera coord to current camera coord
            Pcr = inv_T(Prc)
            # Projection from world coordinate to current camera coord
            Pcw = Pcr @ Prw

            # Camera pose in world coordinate
            tcw = tcw + Rcw @ trc
            Rcw = Rcw @ Rrc

            traj_scale = 0.01
            trajectory.append([traj_scale*tcw[0, 0], traj_scale*tcw[1, 0], traj_scale*tcw[2, 0]])
            print("# [main] time for Recover pose: {:.2f} [ms]".format(timer.get_timediff()))

            # Calculate 3D coordinates of feature points
            min_depth = 0
            max_depth = 1000
            pts3d = None
            idxs_inlier = np.array([True] * len(good_fp_curr))
            if Pcw is not None and Prw is not None:
                # Get 3D coordinates by triangulation
                pts3d, mask_pts3d = triangulate_normalized_points(
                    Pcw, Prw, good_fpn_curr, good_fpn_prev)
                #print("# [main] mean depth= {}".format(np.mean(np.array(pts3d)[..., 2])))
                # Add 3D points in array with its color
                n_masked = 0
                n_negative = 0
                delta = 2
                for i, p in enumerate(pts3d):
                    ptx, pty = good_fp_curr[i, :2]
                    x_start = int(resize_scale*(ptx - delta))
                    x_end = int(resize_scale*(ptx + delta))
                    y_start = int(resize_scale*(pty - delta))
                    y_end = int(resize_scale*(pty + delta))
                    color_patch = img[y_start:y_end, x_start:y_end, :3]
                    color = cv2.mean(color_patch/255.0)[:3]
                    if not mask_pts3d[i]:
                        n_masked += 1
                        continue
                    if p[2] < min_depth or p[2] > max_depth: # check depth
                        n_negative += 1
                        idxs_inlier[i] = False
                        continue
                    elif flg_mapadded:
                        map_points.append(p)
                        map_color.append(color)
            draw3d.show(Rcw, tcw, trajectory, traj_scale, map_points, map_color)
            print("# [main] illegal depth n= {}".format(n_negative))

            # Draw 2D fature points
            img_draw = np.zeros_like(img)
            for i_pts, (kps_cur0, kps_ref0) in enumerate(zip(good_fp_curr, good_fp_prev)):
                size = 2
                fill = -1
                if pts3d is not None:
                    is_inlier = idxs_inlier[i_pts]
                    if not is_inlier:
                        color = [180,255,255] # Red
                        size = 5
                        fill = 2
                    else:
                        depth = pts3d[i_pts][2]
                        color = [min(int(depth*2),255),255,255]
                        size = 2
                else:
                        color = [255,0,0] # Red
                pt_cur = (int(kps_cur0[0]), int(kps_cur0[1]))
                pt_ref = (int(kps_ref0[0]), int(kps_ref0[1]))
                img_draw = cv2.line(img_draw, pt_cur, pt_ref, color, 2)
                img_draw = cv2.circle(img_draw, pt_cur, size, color, -1)

            img_draw = cv2.cvtColor(img_draw, cv2.COLOR_HSV2BGR)
            img_draw = cv2.addWeighted(img, 0.5, img_draw, 0.5, 0)

            n_good1 = len(good_fp_curr)
            good_fp_curr = good_fp_curr[idxs_inlier == True]
            print("# [main] remove outliers: {} -> {}".format(n_good1, len(good_fp_curr)))

            cv2.imshow('Camera+FP', img_draw)
            print("# [main] time for draw: {:.2f} [ms]".format(timer.get_timediff()))
        
        flg_mapadded = False

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()