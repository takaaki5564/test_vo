#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import time
import math

import OpenGL.GL as gl
import pangolin


def calc_rmat(theta_x, theta_y, theta_z):
    rx = np.array([
        [1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)]])

    ry = np.array([
        [math.cos(theta_y), 0, math.sin(theta_y)],
        [0, 1, 0],
        [-math.sin(theta_y), 0, math.cos(theta_y)]])

    rz = np.array([
        [math.cos(theta_z), -math.sin(theta_z), 0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1]])

    # The order is RxRyRz
    rmat = np.dot(rx, np.dot(ry, rz))

    return rmat

def pose_estimation(corners, ids, rejected, image, kmat, dmat):
    rvec = None
    tvec = None
    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.10, kmat, dmat)
            
            cv2.aruco.drawDetectedMarkers(image, corners)
            cv2.drawFrameAxes(image, kmat, dmat, rvec, tvec, 0.1)

    return image, rvec, tvec

def main():
    cam_id = 4
    if len(sys.argv) > 1:
        cam_id = int(sys.argv[1])
    capture = cv2.VideoCapture(cam_id)
    #time.sleep(2.0)

    ret, frame = capture.read()
    if frame is None:
        print("Camera not detected")
        exit()
    else:
        h, w = frame.shape[:2]
        print("Image size: (w={}, h={})".format(w, h))

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    arucoParams = cv2.aruco.DetectorParameters_create()

    # Camera intrinsic / distortion parameters
    k = np.load("./logicool720p/mtx.npy")
    d = np.load("./logicool720p/dist.npy")

    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]

    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 1000),
        pangolin.ModelViewLookAt(-2, 1, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640/480.0)
    dcam.SetHandler(pangolin.Handler3D(scam))

    axis_x = [[-1,0,0],[1,0,0]]
    axis_y = [[0,-1,0],[0,1,0]]
    axis_z = [[0,0,-1],[0,0,1]]

    Rt = None
    Rmat = None
    distance = 0
    pose = np.identity(4)

    i_frame = 0

    while not pangolin.ShouldQuit():
        ret, frame = capture.read()
        if frame is None:
            print("Camera not detected")
            exit()
        i_frame += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find brightest spot
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
        # Pose estimation and draw
        detected_markers, rvec, tvec = pose_estimation(corners, ids, rejected, frame, k, d)

        if rvec is not None and tvec is not None:
            tvec = tvec.reshape(3,1)
            distance = np.linalg.norm(tvec)
            print("rvec={}".format(rvec))
            print("tvec={}".format(tvec))
            Rmat = cv2.Rodrigues(rvec)[0].transpose()
            print("Rmat={}".format(Rmat))
            tvec_wc = - Rmat @ tvec
            print("tvec_wc={}".format(tvec_wc))

        # Panglon show srart
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)

        # Draw lines
        gl.glLineWidth(2)
        gl.glColor3f(0.0, 0.0, 0.0)
        pangolin.DrawLine(axis_x)
        gl.glColor3f(0.0, 0.0, 0.0)
        pangolin.DrawLine(axis_y)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawLine(axis_z)

        # Draw current pose
        if rvec is not None and tvec_wc is not None:
            pose[:3, 3] = np.array(tvec_wc[:3, 0])
            pose[:3, :3] = np.array(Rmat)
        else:
            pose = np.identity(4)
            pose[:3, 3] = np.array([0, 0, 0])

        gl.glColor3f(0.0, 0.0, 1.0) # blue
        gl.glLineWidth(2)
        pangolin.DrawCamera(pose, 0.25, 0.3, 0.4)

        # Draw points to positive direction
        gl.glPointSize(10)
        gl.glColor3f(1.0, 0.0, 0.0)
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        pangolin.DrawPoints(points, colors)    

        # Calucate position in Camera coordinate
        px, py = maxLoc
        z_cc = 1.0
        x_cc = z_cc / fx * (px - cx)
        y_cc = z_cc / fy * (py - cy)
        pvec_cc = np.array([[x_cc], [y_cc], [z_cc]])
        print("pvec_cc= {}".format(pvec_cc))

        # Convert to World(Marker) coordinate
        if Rmat is not None and tvec is not None:
            pvec_wc = Rmat @ (pvec_cc - tvec)
            print("pvec_wc= {}".format(pvec_wc))

            gl.glPointSize(20)
            gl.glColor3f(0.0, 1.0, 0.0)
            points = np.array([[pvec_wc[0,0], pvec_wc[1,0], pvec_wc[2,0]]])
            colors = [[0.0, 1.0, 0.0]]
            print("##pt.shape={}".format(points.shape))
            pangolin.DrawPoints(points, colors)    

        # Draw brightest spot
        cv2.circle(detected_markers, maxLoc, 5, (255,0,0), 2)

        cv2.imshow('image', detected_markers)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

        pangolin.FinishFrame()

    cv2.destroyAllWindows()
    capture.release()

    print("Finished program")

if __name__ == '__main__':
    main()