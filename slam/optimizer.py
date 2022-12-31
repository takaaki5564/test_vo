import math 
import numpy as np

from threading import RLock

import g2o

from slam.utils import *
from slam.frame import *
from slam.map import *


def bundle_adjustment(keyframes, points, local_window, fixed_points=False, verbose=False, rounds=10, use_robust_kernel=False):
    pass


# optimize points reprojection error:
# - frame pose is optimized
# - 3D points observed in frame are considered fixed
# output:
# - mean_squared_error
# - is_ok: is the pose optimization successeful?
# - num_valid_points: number of inliers
def g2o_pose_optimization(frame, verbose=False, rounds=10):
    is_ok = True

    # create g2o optimizer
    opt = g2o.SparseOptimizer()

    #thHuberMono = math.sqrt(5.991);  # chi-square 2 DOFS 
    thHuberMono = math.sqrt(50);  # chi-square 2 DOFS 

    block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
    opt.set_algorithm(solver)

    point_edge_pairs = {}
    num_point_edges = 0


    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_estimate(g2o.SE3Quat(frame.Rcw, frame.tcw))
    v_se3.set_id(0)  
    v_se3.set_fixed(False)
    opt.add_vertex(v_se3)

    n_none = 0
    with MapPoint.global_lock:
        # add point vertices to graph 
        for idx, p in enumerate(frame.points):
            if p is None:
                n_none += 1
                continue

            # reset outlier flag 
            frame.outliers[idx] = False 

            # add edge
            #print('adding edge between point ', p.id,' and frame ', frame.id)
            edge = g2o.EdgeSE3ProjectXYZOnlyPose()

            edge.set_vertex(0, opt.vertex(0))
            edge.set_measurement(frame.kpsu[idx])
            invSigma2 = Frame.tracker.inv_level_sigmas2[frame.octaves[idx]]
            edge.set_information(np.eye(2)*invSigma2)
            edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

            edge.fx = frame.camera.fx 
            edge.fy = frame.camera.fy
            edge.cx = frame.camera.cx
            edge.cy = frame.camera.cy
            edge.Xw = p.pt[0:3]
            
            opt.add_edge(edge)

            point_edge_pairs[p] = (edge, idx) # one edge per point 
            num_point_edges += 1

    print('####[g2o_pose_optimization] n_none= {}, other= {}'.format(n_none, len(frame.points) - n_none) )

    if num_point_edges < 3:
        print('####[g2o_pose_optimization] pose_optimization: not enough correspondences!') 
        is_ok = False 
        return 0, is_ok, 0

    # perform 4 optimizations: 
    # after each optimization we classify observation as inlier/outlier;
    # at the next optimization, outliers are not included, but at the end they can be classified as inliers again
    chi2Mono = 50 #5.991 # chi-square 2 DOFs
    num_bad_point_edges = 0
    ave_outlier_chi2 = 0

    num_good_point_edges = 0
    ave_inlier_chi2 = 0

    for it in range(4):
        v_se3.set_estimate(g2o.SE3Quat(frame.Rcw, frame.tcw))
        opt.initialize_optimization()        
        opt.optimize(rounds)

        num_bad_point_edges = 0

        for p, edge_pair in point_edge_pairs.items(): 
            edge, idx = edge_pair
            if frame.outliers[idx]:
                edge.compute_error()

            chi2 = edge.chi2()
            
            if chi2 > chi2Mono:
                frame.outliers[idx] = True 
                edge.set_level(1)
                num_bad_point_edges +=1
                ave_outlier_chi2 += chi2
            else:
                frame.outliers[idx] = False
                edge.set_level(0)      
                num_good_point_edges += 1                          
                ave_inlier_chi2 += chi2

            if it == 2:
                edge.set_robust_kernel(None)

        if len(opt.edges()) < 10:
            print('####[g2o_pose_optimization] pose_optimization: stopped - not enough edges!')   
            is_ok = False           
            break                 
    
    if num_bad_point_edges != 0:
        ave_outlier_chi2 /= num_bad_point_edges
    if num_good_point_edges != 0:
        ave_inlier_chi2 /= num_good_point_edges
    print('####[g2o_pose_optimization] available ', num_point_edges, ' points, found ', num_bad_point_edges, ' bad points')     
    print('####[g2o_pose_optimization] ave_outlier_chi2= ', ave_outlier_chi2, ' ave_inlier_achi2= ', ave_inlier_chi2)     
    if num_point_edges == num_bad_point_edges:
        print('####[g2o_pose_optimization] pose_optimization: all the available correspondences are bad!')           
        is_ok = False      

    # update pose estimation
    if is_ok: 
        est = v_se3.estimate()
        # R = est.rotation().matrix()
        # t = est.translation()
        # frame.update_pose(poseRt(R, t))
        frame.update_pose(g2o.Isometry3d(est.orientation(), est.position()))

    # since we have only one frame here, each edge corresponds to a single distinct point
    num_valid_points = num_point_edges - num_bad_point_edges   
    
    mean_squared_error = opt.active_chi2()/max(num_valid_points,1)

    return mean_squared_error, is_ok, num_valid_points
