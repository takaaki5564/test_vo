
import sys
import time
import numpy as np
import cv2
import g2o
from enum import Enum
import time

from collections import defaultdict 

from queue import Queue 
from concurrent.futures import ThreadPoolExecutor


from slam.utils import *
from slam.map import *
from slam.frame import *
from slam.feature import *
from slam.tracking import *
from slam.initializer import *



class LocalMapping(object):
    def __init__(self, map):
        self.map = map
        
        self.recently_added_points = set()
        
        self.kf_cur = None   # current processed keyframe  
        self.kid_last_BA = -1 # last keyframe id when performed BA  
        
        self.queue = []#Queue()
        self.descriptor_distance_sigma = 0        
        
        self.lock_accept_keyframe = RLock()
        
        #self.opt_abort_flag = g2o.Flag(False)  
         
    def push_keyframe(self, keyframe):
        print("####[local_mapping.push_keyframe] added keyframe! {}".format(len(self.queue) ))
        #self.queue.put(keyframe)
        self.queue.append(keyframe)
        print("####[local_mapping.push_keyframe] added keyframe! {}".format(len(self.queue) ))

    def do_local_mapping(self):
        if len(self.queue) == 0:
            print("####[local_mapping] queue empty, so out")
            return  
    
        if self.map.local_map.is_empty():
            print("####[local_mapping] local_map empty, so out")
            return 
                
        self.kf_cur = self.queue[-1] #get()   
                
        self.process_new_keyframe()

        # do map points culling 
        num_culled_points = self.cull_map_points()
        print('####[local_mapping] culled points: ', num_culled_points)    
                
        # create new points by triangulation 
        total_new_pts = self.create_new_map_points()
        print("####[local_mapping] new map points: %d " % (total_new_pts))   
        
        if len(self.queue) == 0:
            # fuse map points of close keyframes
            total_fused_pts = self.fuse_map_points()
            print("####[local_mapping] fused map points: %d " % (total_fused_pts)) 

            # check redundant local Keyframes
            #num_culled_keyframes = self.cull_keyframes()
            #print("####[local_mapping] culled keyframes: %d " % (num_culled_keyframes))    

        # reset optimization flag 
        #self.opt_abort_flag.value = False                

        if len(self.queue) == 0:

            # launch local bundle adjustment 
            self.local_BA()      

            num_culled_keyframes = self.cull_keyframes() 

    def local_BA(self):
        # local optimization 
        kNumMinObsForKeyFrameDefault = 3
        err = self.map.locally_optimize(kf_ref=self.kf_cur, abort_flag=self.opt_abort_flag)
        print("local optimization error^2:   %f" % err)       
        num_kf_ref_tracked_points = self.kf_cur.num_tracked_points(kNumMinObsForKeyFrameDefault) # number of tracked points in k_ref
        print('KF(%d) #points: %d ' %(self.kf_cur.id, num_kf_ref_tracked_points))           
          

    def process_new_keyframe(self):
        # associate map points to keyframe observations (only good points)
        # and update normal and descriptor
        for idx,p in enumerate(self.kf_cur.get_points()):
            if p is not None and not p.is_bad:  
                if p.add_observation(self.kf_cur, idx):
                    p.update_info() 
                else: 
                    self.recently_added_points.add(p)       
                            
        self.kf_cur.update_connections()
        self.map.add_keyframe(self.kf_cur)   # add kf_cur to map        
                        
        
    def cull_map_points(self):
        print('####[local_mapping] culling map points...')  
        th_num_observations = 2      
        min_found_ratio = 0.25  
        current_kid = self.kf_cur.kid 
        remove_set = set() 
        for p in self.recently_added_points:
            if p.is_bad:
                remove_set.add(p)
            elif p.get_found_ratio() < min_found_ratio:
                self.map.remove_point(p)  
                remove_set.add(p)                  
            elif (current_kid - p.first_kid) >= 2 and p.num_observations <= th_num_observations:             
                self.map.remove_point(p)  
                remove_set.add(p)        
            elif (current_kid - p.first_kid) >= 3:  # after three keyframes we do not consider the point a recent one         
                remove_set.add(p)   
        self.recently_added_points = self.recently_added_points - remove_set  
        num_culled_points = len(remove_set)                                             
        return num_culled_points           
           
           
    def cull_keyframes(self): 
        print('####[local_mapping.cull_keyframes] culling keyframes...')
        num_culled_keyframes = 0
        # check redundant keyframes in local keyframes: a keyframe is considered redundant if the 90% of the MapPoints it sees, 
        # are seen in at least other 3 keyframes (in the same or finer scale)
        th_num_observations = 3
        for kf in self.kf_cur.get_covisible_keyframes(): 
            if kf.kid==0:
                continue 
            kf_num_points = 0     # num good points for kf          
            kf_num_redundant_observations = 0   # num redundant observations for kf       
            for i,p in enumerate(kf.get_points()): 
                if p is not None and not p.is_bad:
                    kf_num_points += 1
                    if p.num_observations>th_num_observations:
                        scale_level = kf.octaves[i]  # scale level of observation in kf 
                        p_num_observations = 0
                        for kf_j,idx in p.observations():
                            if kf_j is kf:
                                continue
                            assert(not kf_j.is_bad)
                            scale_level_i = kf_j.octaves[idx]  # scale level of observation in kfi
                            if scale_level_i <= scale_level+1:  # N.B.1 <- more aggressive culling  (expecially when scale_factor=2)
                            #if scale_level_i <= scale_level:     # N.B.2 <- only same scale or finer                            
                                p_num_observations +=1
                                if p_num_observations >= th_num_observations:
                                    break 
                        if p_num_observations >= th_num_observations:
                            kf_num_redundant_observations += 1
            # if (kf_num_redundant_observations > 0.9 * kf_num_points) and \
            #    (kf_num_points > 50) and \
            #    (kf.timestamp - kf.parent.timestamp < 0.5):
            #     print('####[local_mapping.cull_keyframes] setting keyframe ', kf.id,' bad - redundant observations: ', kf_num_redundant_observations/max(kf_num_points,1),'%')
            #     kf.set_bad()
            #     num_culled_keyframes += 1
        return num_culled_keyframes


    def precompute_kps_matches(self, match_idxs, local_keyframes):            
        kf_pairs = []
        kLocalMappingParallelKpsMatching = True
        kLocalMappingParallelKpsMatchingNumWorkers = 4

        if not kLocalMappingParallelKpsMatching: 
            # do serial computation 
            for kf in local_keyframes:
                if kf is self.kf_cur or kf.is_bad:
                    continue   
                idxs1, idxs2 = Frame.feature_matcher.match(self.kf_cur.des, kf.des)             
                match_idxs[(self.kf_cur,kf)]=(idxs1,idxs2)  
        else: 
            # do parallell computation 
            def thread_match_function(kf_pair):
                kf1,kf2 = kf_pair        
                idxs1, idxs2 = Frame.feature_matcher.match(kf1.des, kf2.des)             
                match_idxs[(kf1, kf2)]=(idxs1,idxs2)                   
            for kf in local_keyframes:
                if kf is self.kf_cur or kf.is_bad:
                    continue    
                kf_pairs.append((self.kf_cur, kf))                       
            with ThreadPoolExecutor(max_workers = kLocalMappingParallelKpsMatchingNumWorkers) as executor:
                executor.map(thread_match_function, kf_pairs) # automatic join() at the end of the `width` block 
        return match_idxs
            
            
    # triangulate matched keypoints (without a corresponding map point) amongst recent keyframes      
    def create_new_map_points(self):
        print('#####[local_mapping.create_new_mp] creating new map points')
        total_new_pts = 0
        
        local_keyframes = self.map.local_map.get_best_neighbors(self.kf_cur)
        print('#####[local_mapping.create_new_mp] local map keyframes: ', [kf.id for kf in local_keyframes if not kf.is_bad], ' + ', self.kf_cur.id, '...')            
        
        match_idxs = defaultdict(lambda: (None,None))   # dictionary of matches  (kf_i, kf_j) -> (idxs_i,idxs_j)         
        # precompute keypoint matches 
        match_idxs = self.precompute_kps_matches(match_idxs, local_keyframes)
                    
        for i,kf in enumerate(local_keyframes):
            if kf is self.kf_cur or kf.is_bad:
                continue 
            if i>0 and not len(self.queue) == 0:
                print('#####[local_mapping.create_new_mp] creating new map points *** interruption ***')
                return total_new_pts
            #print("adding map points for KFs (%d, %d)" % (self.kf_cur.id, kf.id))  
            
            # extract matches from precomputed map  
            idxs_kf_cur, idxs_kf = match_idxs[(self.kf_cur,kf)]
            
            # find keypoint matches between self.kf_cur and kf
            # N.B.: all the matched keypoints computed by search_frame_for_triangulation() are without a corresponding map point              
            idxs_cur, idxs, num_found_matches, _ = self.search_frame_for_triangulation(self.kf_cur, kf, idxs_kf_cur, idxs_kf,
                                                                                   max_descriptor_distance=0.5*self.descriptor_distance_sigma)
                        
            if len(idxs_cur) > 0:
                # try to triangulate the matched keypoints that do not have a corresponding map point   
                pts3d, mask_pts3d = triangulate_normalized_points(self.kf_cur.pose, kf.pose, self.kf_cur.kpsn[idxs_cur], kf.kpsn[idxs])
                    
                new_pts_count,_,list_added_points = self.map.add_points(pts3d, mask_pts3d, self.kf_cur, kf, idxs_cur, idxs, self.kf_cur.img, do_check=True)
                print("# added map points: %d for KFs (%d, %d)" % (new_pts_count, self.kf_cur.id, kf.id))        
                total_new_pts += new_pts_count 
                self.recently_added_points.update(list_added_points)
        return total_new_pts
        
        
    # fuse close map points of local keyframes 
    def fuse_map_points(self):
        print('>>>> fusing map points')
        total_fused_pts = 0
        
        local_keyframes = self.map.local_map.get_best_neighbors(self.kf_cur)
        print('local map keyframes: ', [kf.id for kf in local_keyframes if not kf.is_bad], ' + ', self.kf_cur.id, '...')   
                
        # search matches by projection from current KF in close KFs        
        for kf in local_keyframes:
            if kf is self.kf_cur or kf.is_bad:  
                continue      
            num_fused_pts = search_and_fuse(self.kf_cur.get_points(), kf,
                                            max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
                                            max_descriptor_distance=0.5*self.descriptor_distance_sigma) # finer search
            print("# fused map points: %d for KFs (%d, %d)" % (num_fused_pts, self.kf_cur.id, kf.id))  
            total_fused_pts += num_fused_pts    
               
        # search matches by projection from local points in current KF  
        good_local_points = [p for kf in local_keyframes if not kf.is_bad for p in kf.get_points() if (p is not None and not p.is_bad) ]  # all good points in local frames 
        good_local_points = np.array(list(set(good_local_points))) # be sure to get only one instance per point                
        num_fused_pts = search_and_fuse(good_local_points, self.kf_cur,
                                        max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
                                        max_descriptor_distance=0.5*self.descriptor_distance_sigma) # finer search
        print("# fused map points: %d for local map into KF %d" % (num_fused_pts, self.kf_cur.id))  
        total_fused_pts += num_fused_pts   
            
        # update points info 
        for p in self.kf_cur.get_points():
            if p is not None and not p.is_bad: 
                p.update_info() 
                
        # update connections in covisibility graph 
        self.kf_cur.update_connections()            
        
        return total_fused_pts               

kMaxDescriptorDistance = 100
def search_frame_for_triangulation(kf1, kf2, idxs1=None, idxs2=None, 
                                   max_descriptor_distance=0.5*kMaxDescriptorDistance):   
    idxs2_out = []
    idxs1_out = []
    num_found_matches = 0
    img2_epi = None     

    O1w = kf1.Ow
    O2w = kf2.Ow
    # compute epipoles
    e1,_ = kf1.project_point(O2w)  # in first frame 
    e2,_ = kf2.project_point(O1w)  # in second frame  
    
    baseline = np.linalg.norm(O1w-O2w) 

    medianDepth = kf2.compute_points_median_depth()
    if medianDepth == -1:
        print("search for triangulation: f2 with no points")        
        medianDepth = kf1.compute_points_median_depth()        
    ratioBaselineDepth = baseline/medianDepth
    if ratioBaselineDepth < 0.01:
        print("search for triangulation: impossible with too low ratioBaselineDepth!")
        return idxs1_out, idxs2_out, num_found_matches, img2_epi # EXIT        

    # compute the fundamental matrix between the two frames by using their estimated poses 
    F12, H21 = computeF12(kf1, kf2)

    if idxs1 is None or idxs2 is None:
        timerMatch = Timer()
        timerMatch.start()        
        idxs1, idxs2 = Frame.feature_matcher.match(kf1.des, kf2.des)        
        print('search_frame_for_triangulation - matching - timer: ', timerMatch.elapsed())        
    
    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and Frame.oriented_features     

    # check epipolar constraints 
    for i1,i2 in zip(idxs1,idxs2):
        if kf1.get_point_match(i1) is not None or kf2.get_point_match(i2) is not None: # we are searching for keypoint matches where both keypoints do not have a corresponding map point 
            continue 
        
        descriptor_dist = Frame.descriptor_distance(kf1.des[i1], kf2.des[i2])
        if descriptor_dist > max_descriptor_distance:
            continue     
        
        kp1 = kf1.kpsu[i1]
        
        kp2 = kf2.kpsu[i2]
        kp2_scale_factor = Frame.feature_manager.scale_factors[kf2.octaves[i2]]        
        delta = kp2-e2        
        if np.inner(delta,delta) < kMinDistanceFromEpipole2 * kp2_scale_factor:   # OR.            
             continue           
        
        # check epipolar constraint         
        sigma2_kp2 = Frame.feature_manager.level_sigmas2[kf2.octaves[i2]]
        if check_dist_epipolar_line(kp1,kp2,F12,sigma2_kp2):
            idxs1_out.append(i1)
            idxs2_out.append(i2)

    num_found_matches = len(idxs1_out)
             
    return idxs1_out, idxs2_out, num_found_matches, img2_epi