import time
import numpy as np
import json
import math 
import cv2

from collections import Counter, deque
from ordered_set import OrderedSet # from https://pypi.org/project/ordered-set/

from threading import RLock, Thread

from slam.utils import *
from slam.frame import *
from slam.frame import *

import g2o
#import slam.optimizer_g2o 

from slam import optimizer_g2o
from slam import optimizer


kMaxLenFrameDeque = 20
class Map:
    def __init__(self):
        self._lock = RLock()
        self._update_lock = RLock()

        self.frames = deque(maxlen=kMaxLenFrameDeque)
        self.keyframes = OrderedSet()
        self.points = set() # set empty set

        self.max_frame_id = 0
        self.max_point_id = 0
        self.max_keyframe_id = 0

        self.local_map = LocalCovisibilityMap(map=self)

    @property    
    def update_lock(self):  
        return self._update_lock     

    def get_points(self): 
        with self._lock:       
            return self.points.copy()  
        
    def num_points(self):
        with self._lock: 
            return len(self.points)                

    def get_frame(self,idx): 
        with self._lock:       
            return self.frames[idx] 
                
    def get_frames(self): 
        with self._lock:       
            return self.frames.copy()       
        
    def num_frames(self):
        with self._lock: 
            return len(self.frames)
     
    def get_keyframes(self): 
        with self._lock:       
            return self.keyframes.copy()       
    
    def num_keyframes(self):
        with self._lock: 
            return len(self.keyframes)  

    def add_keyframe(self, keyframe):
        print("###added keyframe ####################")
        with self._lock:
            assert(keyframe.is_keyframe)
            ret = self.max_keyframe_id
            keyframe.kid = ret
            keyframe.is_keyframe = True
            keyframe.map = self
            self.keyframes.add(keyframe)
            self.max_keyframe_id += 1
            return ret


    def add_point(self, point):
        with self._lock:
            ret = self.max_point_id
            point.id = ret
            point.map = self
            self.max_point_id += 1
            self.points.add(point)
            return ret


    def remove_point(self, point):
        with self._lock:    
            try:                    
                self.points.remove(point)
            except:
                pass 
            point.delete()        


    def add_frame(self, frame, ovverride_id=False):
        #print("### add frame in map")
        with self._lock:          
            ret = frame.id
            if ovverride_id: 
                ret = self.max_frame_id
                frame.id = ret # override original id    
                self.max_frame_id += 1
            self.frames.append(frame)     
            return ret
        
    def remove_frame(self, frame): 
        with self._lock:                           
            try: 
                self.frames.remove(frame) 
            except: 
                pass 
    
    def add_keyframe(self, keyframe):
        with self._lock:          
            assert(keyframe.is_keyframe)
            ret = self.max_keyframe_id
            keyframe.kid = ret # override original keyframe kid    
            keyframe.is_keyframe = True 
            keyframe.map = self 
            self.keyframes.add(keyframe)            
            self.max_keyframe_id += 1                                  
            return ret    
    
    def remove_keyframe(self, keyframe): 
        with self._lock:
            assert(keyframe.is_keyframe)                               
            try: 
                self.keyframes.remove(keyframe) 
            except: 
                pass     
    
    def num_keyframes(self):      
        return self.max_keyframe_id

    def delete(self):
        with self._lock:          
            for f in self.frames:
                f.reset_points()
            #for kf in self.keyframes:
            #    kf.reset_points()

    def draw_feature_trails(self, img):
        print("#### [map] frame in map len={}".format(len(self.frames)))
        if len(self.frames) > 0:
            img_draw = self.frames[-1].draw_all_feature_trails(img)
            return img_draw
        return img

    def add_points(self, points3d, mask_pts3d, kf1, kf2, idxs1, idxs2, img1, do_check=True, cos_max_parallax=0.9999):
        with self._lock:
            assert(kf1.is_keyframe and kf2.is_keyframe)
            assert(points3d.shape[0] == len(idxs1))
            assert(len(idxs2) == len(idxs1))

            added_points = []
            out_mask_pts3d = np.full(points3d.shape[0], False, dtype=bool)
            if mask_pts3d is None:
                mask_pts3d = np.full(points3d.shape[0], True, dtype=bool)
            
            ratio_scale_consistency = 1.5 * Frame.tracker.scale_factor

            if do_check:
                # project points (2D to 3D)
                uvs1, depths1 = kf1.project_points(points3d)
                bad_depths1 = depths1 <= 0
                uvs2, depths2 = kf2.project_points(points3d)
                bad_depths2 = depths2 <= 0

                # compute back-projected rays (3D to 2D)
                rays1 = np.dot(kf1.Rwc, add_ones(kf1.kpsn[idxs1]).T).T
                norm_rays1 = np.linalg.norm(rays1, axis=-1, keepdims=True)
                rays1 /= norm_rays1
                rays2 = np.dot(kf2.Rwc, add_ones(kf2.kpsn[idxs2]).T).T
                norm_rays2 = np.linalg.norm(rays2, axis=-1, keepdims=True)
                rays2 /= norm_rays2

                # compute dot product of rays
                cos_parallaxs = np.sum(rays1 * rays2, axis=1)
                bad_cos_parallaxs = np.logical_or(cos_parallaxs < 0, cos_parallaxs > cos_max_parallax)

                # compute projection error
                errs1 = uvs1 - kf1.kpsu[idxs1]
                errs1_sqr = np.sum(errs1 * errs1, axis=1)
                kps1_levels = kf1.octaves[idxs1]
                invSigmas2_1 = Frame.tracker.inv_level_sigmas2[kps1_levels]
                chis2_1 = errs1_sqr * invSigmas2_1  # Chi square
                bad_chis2_1 = chis2_1 > 5.991

                # scale consistency 
                #scale_factors_x_depths1 =  Frame.tracker.scale_factors[kps1_levels] * depths1
                #scale_factors_x_depths1_x_ratio_scale_consistency = scale_factors_x_depths1*ratio_scale_consistency                             
                #scale_factors_x_depths2 =  Frame.tracker.scale_factors[kps2_levels] * depths2   
                #scale_factors_x_depths2_x_ratio_scale_consistency = scale_factors_x_depths2*ratio_scale_consistency      
                #bad_scale_consistency = np.logical_or( (scale_factors_x_depths1 > scale_factors_x_depths2_x_ratio_scale_consistency), 
                #                                       (scale_factors_x_depths2 > scale_factors_x_depths1_x_ratio_scale_consistency) )   
                
                # combine all checks 
                #bad_points = bad_cos_parallaxs | bad_depths1 | bad_depths2 | bad_chis2_1 | bad_chis2_2 | bad_scale_consistency                                                           
                #bad_points = bad_cos_parallaxs
                #print("bad points={}".format(bad_points))
                bad_points = [False] * (points3d.shape[0])

            img_coords = np.rint(kf1.kps[idxs1]).astype(np.intp)
            delta = 1
            patch_extension = 1 + 2*delta
            img_pts_start = img_coords - delta
            img_pts_end = img_coords + delta
            img_ranges = np.linspace(img_pts_start, img_pts_end, patch_extension, dtype=np.intp)[:,:].T

            def img_range_elem(ranges, i):
                return ranges[:, i]
            
            print("#### [map]: points3d n={}".format(len(points3d)))
            n_masked = 0
            n_badpt = 0
            for i, p in enumerate(points3d):
                if not mask_pts3d[i]:
                    n_masked += 1
                    continue
                
                idx1_i = idxs1[i]
                idx2_i = idxs2[i]

                if do_check:
                    if bad_points[i]:
                        n_badpt += 1
                        continue
                try:  
                    # get color of the point
                    img_range = img_range_elem(img_ranges, i)
                    color_patch = img1[img_range[1][:, np.newaxis], img_range[0]]
                    color = cv2.mean(color_patch)[:3]
                except IndexError:
                    color = (255, 0, 0)
                
                # add points to this map
                mp = MapPoint(p[0:3], color, kf2, idx2_i)
                self.add_point(mp)
                mp.add_observation(kf1, idx1_i)
                mp.add_observation(kf2, idx2_i)
                mp.update_info()
                out_mask_pts3d[i] = True
                added_points.append(mp)
            
            print("#### [map]: masked n={}, bad={}, added={}".format(n_masked, n_badpt, len(added_points)))
            return len(added_points), out_mask_pts3d, added_points

    # Remove points which have a big reprojection error 
    def remove_points_with_big_reproj_err(self, points): 
        with self._lock:             
            with self.update_lock: 
                #print('map points: ', sorted([p.id for p in self.points]))
                #print('points: ', sorted([p.id for p in points]))           
                culled_pt_count = 0
                for p in points:
                    # compute reprojection error
                    chi2s = []
                    for f, idx in p.observations():
                        uv = f.kpsu[idx]
                        proj,_ = f.project_map_point(p)
                        invSigma2 = Frame.tracker.inv_level_sigmas2[f.octaves[idx]]
                        err = (proj-uv)
                        chi2s.append(np.inner(err,err)*invSigma2)
                    # cull
                    mean_chi2 = np.mean(chi2s)
                    if np.mean(chi2s) > 5.991:  # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                        culled_pt_count += 1
                        #print('removing point: ',p.id, 'from frames: ', [f.id for f in p.keyframes])
                        self.remove_point(p)

    # BA considering all keyframes
    def optimize(self, local_window=20, verbose=False, rounds=10, use_robust_kernel=False, do_cull_points = False):
        err = optimizer.bundle_adjustment(self.get_keyframes(), self.get_points(), local_window = local_window, verbose = verbose, rounds = rounds, use_robust_kernel=False)        
        if do_cull_points: 
            self.remove_points_with_big_reproj_err(self.get_points())
        return err


    # local BA: only local keyframes and local points are adjusted
    def locally_optimize(self, kf_ref, verbose = False, rounds=10):
        keyframes, points, ref_keyframes = self.local_map.update(kf_ref)
        print('####[locally_optimize] local optimization window: ', sorted([kf.id for kf in keyframes]))        
        print('                     refs: ', sorted([kf.id for kf in ref_keyframes]))
        print('                   #points: ', len(points))
        err, ratio_bad_observations = optimizer_g2o.local_bundle_adjustment(keyframes, points, ref_keyframes, False, verbose, rounds, map_lock=self.update_lock)
        print('####[locally_optimize] local optimization - perc bad observations: %.2f %%' % (ratio_bad_observations*100) )              
        return err 



class MapPoint:
    global_lock = RLock()
    _id = 0
    _id_lock = RLock()
    def __init__(self, position, color, keyframe=None, idxf=None, id=None):
        if id is not None:
            self.id = id
        else:
            with MapPoint._id_lock:
                self.id = MapPoint._id
                MapPoint._id += 1
        
        self._lock_pos = RLock()
        self._lock_features = RLock()

        self.map = None
        self._observations = dict() # keyframe observations
        self._frame_views = dict()  # frame observations

        self._is_bad = False        # map point is bad when its num_observation < 2 (cannot be used for BA)
        self._num_observations = 0  # num of keyframe observations
        self.num_times_visible = 1  # num of times the point is visible in the camera
        self.num_times_found = 1    # num of times the point was actually matched and not rejected as outlier
        self.last_frame_id_seen = -1    # last frame id in which the point was seen

        self.replacement = None     # replaceing point

        self._pt = np.array(position)
        self.color = color
        self.des = None             # best descriptor (continuously updated)
        self._min_distance, self._max_distance = 0, float('inf')    # depth info
        self.normal = np.array([0, 0, 1])       # default 3D vector

        self.kf_ref = keyframe
        self.first_kid = -1         # furst ibservation keyframe id

        if keyframe is not None:
            self.first_kid = keyframe.kid
            self.des = keyframe.des[idxf]

        self.num_observations_on_last_update_des = 1       # must be 1!    
        self.num_observations_on_last_update_normals = 1   # must be 1!       

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return (isinstance(rhs, MapPoint) and  self.id == rhs.id)
    
    def __lt__(self, rhs):
        return self.id < rhs.id
    
    def __le__(self, rhs):
        return self.id <= rhs.id

    # return a copy of the dictionary’s list of (key, value) pairs
    def observations(self):
        with self._lock_features:
            return list(self._observations.items())   # https://www.python.org/dev/peps/pep-0469/
        
    # return an iterator of the dictionary’s list of (key, value) pairs
    # NOT thread-safe 
    def observations_iter(self):
        return iter(self._observations.items())       # https://www.python.org/dev/peps/pep-0469/

    # return a copy of the dictionary’s list of keys
    def keyframes(self):
        with self._lock_features:
            return list(self._observations.keys())
    
    def keyframes_iter(self):
            return iter(self._observations.keys())      
        
    def is_in_keyframe(self, keyframe):
        assert(keyframe.is_keyframe)
        with self._lock_features:
            return (keyframe in self._observations)      

      
    def get_observation_idx(self, keyframe):   
        assert(keyframe.is_keyframe)
        with self._lock_features:
            return self._observations[keyframe]     

    def add_observation(self, keyframe, idx):
        assert(keyframe.is_keyframe)
        with self._lock_features:
            if keyframe not in self._observations:
                keyframe.set_point_match(self, idx)
                self._observations[keyframe] = idx
                self._num_observations += 1

    def remove_observation(self, keyframe, idx=None):
        pass         

    @property  
    def pt(self):
        with self._lock_pos:   
            return self._pt  
        
    def homogeneous(self):
        with self._lock_pos:         
            #return add_ones(self._pt)
            return np.concatenate([self._pt,np.array([1.0])], axis=0)        
    @property
    def max_distance(self):
        with self._lock_pos:           
            return 1.2 * self._max_distance  
    
    @property
    def min_distance(self):
        with self._lock_pos:            
            return 1.2 * self._min_distance     

    def update_position(self, position):
        with self.global_lock:           
            with self._lock_pos:   
                self._pt = position 

    def update_info(self):
        with self._lock_features:
            with self._lock_pos:            
                #self.update_normal_and_depth()
                self.update_best_descriptor()

    # return a copy of the dictionary’s list of (key, value) pairs
    def frame_views(self):
        with self._lock_features:
            return list(self._frame_views.items())
        
    # return an iterator of the dictionary’s list of (key, value) pairs
    # NOT thread-safe         
    def frame_views_iter(self):
            return iter(self._frame_views.items())    
        
    # return a copy of the dictionary’s list of keys
    def frames(self):
        with self._lock_features:
            return list(self._frame_views.keys())
        
    # return an iterator of the dictionary’s list of keys
    # NOT thread-safe         
    def frames_iter(self):
            return iter(self._frame_views.keys())            

         
    @property
    def is_bad(self):
        with self._lock_features:
            with self._lock_pos:    
                return self._is_bad                

    def get_normal(self): 
        with self._lock_pos:
            return self.normal                    

    def increase_visible(self, num_times=1):
        with self._lock_features:
            self.num_times_visible += num_times

    @property
    def num_observations(self):
        with self._lock_features:
            return self._num_observations

    def increase_found(self, num_times=1):
        with self._lock_features:
            self.num_times_found += num_times

    def get_found_ratio(self):
        with self._lock_features:       
            return self.num_times_found/self.num_times_visible

    def update_position(self, position):
        with self.global_lock:           
            with self._lock_pos:   
                self._pt = position 

    def get_replacement(self): 
        with self._lock_features:
            with self._lock_pos:
                return self.replacement    

    def add_frame_view(self, frame, idx):
        assert(not frame.is_keyframe)
        with self._lock_features:
            if frame not in self._frame_views:  # do not allow a point to be matched to diffent keypoints 
                #print("####[map.add_frame_view] idx={}".format(idx))
                frame.set_point_match(self, idx)            
                self._frame_views[frame] = idx
                return True 
            else:
                print("####[map.add_frame_view] add frame view skip, idx={}".format(idx))
                return False     

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
                        
    def remove_frame_view(self, frame, idx=None): 
        assert(not frame.is_keyframe)        
        with self._lock_features:
            # remove point from frame     
            if idx is not None:
                #if __debug__: 
                #    assert(self == frame.get_point_match(idx))                     
                frame.remove_point_match(idx)   
                #if __debug__:
                #    assert(not self in frame.get_points())   # checking there are no multiple instances 
            else: 
                frame.remove_point(self)  # remove all match instances                                    
            try:
                del self._frame_views[frame]
            except KeyError:
                pass    
                
    # minimum distance between input descriptor and map point corresponding descriptors 
    def min_des_distance(self, descriptor):
        with self._lock_features:             
            #return min([Frame.descriptor_distance(d, descriptor) for d in self.descriptors()])
            return Frame.descriptor_distance(self.des, descriptor)

    def update_best_descriptor(self,force=False):
        skip = False 
        with self._lock_features:
            if self._is_bad:
                return                          
            if self._num_observations > self.num_observations_on_last_update_des or force:    # implicit if self._num_observations > 1   
                self.num_observations_on_last_update_des = self._num_observations      
                observations = list(self._observations.items())
            else: 
                skip = True 
        if skip:
            return 
        descriptors = [kf.des[idx] for kf,idx in observations if not kf.is_bad]
        N = len(descriptors)
        if N > 2:
            median_distances = [ np.median(Frame.descriptor_distances(descriptors[i], descriptors)) for i in range(N)]
            with self._lock_features:
                self.des = descriptors[np.argmin(median_distances)].copy()


    # replace this point with map point p 
    def replace_with(self, p):         
        if p.id == self.id: 
            return 
        observations, num_times_visible, num_times_found = None, 0, 0 
        with self._lock_features:
            with self._lock_pos:   
                observations = list(self._observations.items()) 
                self._observations.clear()     
                num_times_visible = self.num_times_visible
                num_times_found = self.num_times_found      
                self._is_bad = True  
                self._num_observations = 0  
                self.replacement = p                 
            
        # replace point observations in keyframes
        for kf, kidx in observations: # we have kf.get_point_match(kidx) = self 
            if p.add_observation(kf,kidx): 
                kf.replace_point_match(p,kidx)                  
            else:                
                kf.remove_point_match(kidx)

        p.increase_visible(num_times_visible)
        p.increase_found(num_times_found)        
        p.update_best_descriptor(force=True)    
                     
        self.map.remove_point(self)     


    def update_normal_and_depth(self, frame=None, idxf=None,force=False):
        skip = False  
        with self._lock_features:
            with self._lock_pos:   
                if self._is_bad:
                    return                 
                if self._num_observations > self.num_observations_on_last_update_normals or force:   # implicit if self._num_observations > 1          
                    self.num_observations_on_last_update_normals = self._num_observations 
                    observations = list(self._observations.items())
                    kf_ref = self.kf_ref 
                    idx_ref = self._observations[kf_ref]
                    position = self._pt.copy() 
                else: 
                    skip = True 
        if skip or len(observations)==0:
            return 
                     
        normals = np.array([normalize_vector2(position-kf.Ow) for kf,idx in observations])
        normal = normalize_vector2(np.mean(normals,axis=0))
  
        level = kf_ref.octaves[idx_ref]
        level_scale_factor = Frame.tracker.scale_factors[level]
        dist = np.linalg.norm(position-kf_ref.Ow)
        
        with self._lock_pos:
            self._max_distance = dist * level_scale_factor
            self._min_distance = self._max_distance / Frame.tracker.scale_factors[Frame.tracker.num_levels-1]            
            self.normal = normal

    def delete(self):                
        with self._lock_features:
            with self._lock_pos:
                self._is_bad = True 
                self._num_observations = 0                   
                observations = list(self._observations.items()) 
                self._observations.clear()        
        for kf,idx in observations:
            kf.remove_point_match(idx)         
        del self  # delete if self is the last reference 


# Local map base class 
class LocalMapBase(object):
    def __init__(self, map=None):
        self._lock = RLock()          
        self.map = map   
        self.keyframes     = OrderedSet() # collection of local keyframes 
        self.points        = set() # points visible in 'keyframes'  
        self.ref_keyframes = set() # collection of 'covisible' keyframes not in self.keyframes that see at least one point in self.points   

    @property    
    def lock(self):  
        return self._lock 
    
    def is_empty(self):
        with self._lock:           
            return len(self.keyframes)==0 
    
    def get_points(self): 
        with self._lock:       
            return self.points.copy()  
        
    def num_points(self):
        with self._lock: 
            return len(self.points)                
                
    def get_keyframes(self): 
        with self._lock:       
            return self.keyframes.copy()       
    
    def num_keyframes(self):
        with self._lock: 
            return len(self.keyframes)  
            
    # given some input local keyframes, get all the viewed points and all the reference keyframes (that see the viewed points but are not in the local keyframes)
    def update_from_keyframes(self, local_keyframes):
        local_keyframes = set([kf for kf in local_keyframes if not kf.is_bad])   # remove possible bad keyframes                         
        ref_keyframes = set()   # reference keyframes: keyframes not in local_keyframes that see points observed in local_keyframes      

        good_points = set([p for kf in local_keyframes for p in kf.get_matched_good_points()])  # all good points in local_keyframes (only one instance per point)
        for p in good_points:     
            # get the keyframes viewing p but not in local_keyframes      
            for kf_viewing_p in p.keyframes():          
                if (not kf_viewing_p.is_bad) and (not kf_viewing_p in local_keyframes):
                    ref_keyframes.add(kf_viewing_p)      

        with self.lock: 
            #local_keyframes = sorted(local_keyframes, key=lambda x:x.id)              
            #ref_keyframes = sorted(ref_keyframes, key=lambda x:x.id)              
            self.keyframes = local_keyframes
            self.points = good_points 
            self.ref_keyframes = ref_keyframes                                                  
        return local_keyframes, good_points, ref_keyframes   
         

    # from a given input frame compute: 
    # - the reference keyframe (the keyframe that sees most map points of the frame)
    # - the local keyframes 
    # - the local points  
    def get_frame_covisibles(self, frame):
        kMaxNumOfKeyframesInLocalMap = 80
        points = frame.get_matched_good_points()
        if len(points) == 0:
            print('####[get_frame_covisibles] - frame with not points')
        
        # for all map points in frame check in which other keyframes are they seen
        # increase counter for those keyframes
        print("####[get_frame_covisibles] points len={}".format(len(points)))
        viewing_keyframes = [kf for p in points for kf in p.keyframes()] # if not kf.is_bad]
        viewing_keyframes = Counter(viewing_keyframes)
        print("####[get_frame_covisibles] viewing_keyframes.most_common: len={}".format(len(viewing_keyframes.most_common(1))))
        kf_ref = viewing_keyframes.most_common(1)[0][0]          
    
        # include also some not-already-included keyframes that are neighbors to already-included keyframes
        for kf in list(viewing_keyframes.keys()):
            second_neighbors = kf.get_best_covisible_keyframes(10)
            viewing_keyframes.update(second_neighbors)
            children = kf.get_children()
            viewing_keyframes.update(children)        
            if len(viewing_keyframes) >= kMaxNumOfKeyframesInLocalMap:
                break                 
        
        local_keyframes_counts = viewing_keyframes.most_common(kMaxNumOfKeyframesInLocalMap)
        local_points = set()
        local_keyframes = []
        for kf,c in local_keyframes_counts:
            local_points.update(kf.get_matched_points())
            local_keyframes.append(kf)
        return kf_ref, local_keyframes, local_points  


# Local map from covisibility graph
class LocalCovisibilityMap(LocalMapBase):
    def __init__(self, map=None):
        super().__init__(map)
                
    def update_keyframes(self, kf_ref): 
        with self._lock:         
            assert(kf_ref is not None)
            self.keyframes = OrderedSet()
            self.keyframes.add(kf_ref)
            neighbor_kfs = [kf for kf in kf_ref.get_covisible_keyframes() if not kf.is_bad]
            self.keyframes.update(neighbor_kfs)
            return self.keyframes
        
    def get_best_neighbors(self, kf_ref, N=10): 
        return kf_ref.get_best_covisible_keyframes(N)               
    
    # update the local keyframes, the viewed points and the reference keyframes (that see the viewed points but are not in the local keyframes)
    def update(self, kf_ref):
        self.update_keyframes(kf_ref)
        return self.update_from_keyframes(self.keyframes)


# search by projection matches between {input map points} and {unmatched keypoints of frame f_cur}, (access frame from tracking thread, no need to lock)
def search_map_by_projection(points, f_cur, 
                             max_reproj_distance=3, 
                             max_descriptor_distance=0,
                             ratio_test=0.8):
    Ow = f_cur.Ow 
    
    found_pts_count = 0
    found_pts_fidxs = []   # idx of matched points in current frame 
    
    #reproj_dists = []
    
    if len(points) == 0:
        return 0 
            
    # check if points are visible 
    visible_pts, projs, depths, dists = f_cur.are_visible(points)
    
    predicted_levels = predict_detection_levels(points, dists) 
    kp_scale_factors = Frame.tracker.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    kd_idxs = f_cur.kd.query_ball_point(projs, radiuses)

    for i, p in enumerate(points):
        if not visible_pts[i] or p.is_bad:     # point not visible in frame or is bad 
            continue        
        if p.last_frame_id_seen == f_cur.id:   # we already matched this map point to current frame or it was outlier 
            continue
        
        p.increase_visible()
        
        predicted_level = predicted_levels[i]        
                       
        best_dist = math.inf 
        best_dist2 = math.inf
        best_level = -1 
        best_level2 = -1               
        best_k_idx = -1  

        # find closest map points of f_cur
        for kd_idx in kd_idxs[i]:
     
            p_f = f_cur.points[kd_idx]
            # check there is not already a match               
            if  p_f is not None:
                if p_f.num_observations > 0:
                    continue 
                
            # check detection level     
            kp_level = f_cur.octaves[kd_idx]    
            if (kp_level < predicted_level - 1) or (kp_level > predicted_level):
                continue

            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])
  
            if descriptor_dist < best_dist:                                      
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_k_idx = kd_idx    
            else: 
                if descriptor_dist < best_dist2:  
                    best_dist2 = descriptor_dist
                    best_level2 = kp_level                                        
                                                       
        if best_dist < max_descriptor_distance:            
            # apply match distance ratio test only if the best and second are in the same scale level 
            # if (best_level2 == best_level) and (best_dist > best_dist2 * ratio_test): 
            #     continue 
            if p.add_frame_view(f_cur, best_k_idx):
                found_pts_count += 1  
                found_pts_fidxs.append(best_k_idx)  
            
    reproj_dist_sigma = max_descriptor_distance
                       
    return found_pts_count, reproj_dist_sigma, found_pts_fidxs   

# predict detection levels from map point distances    
def predict_detection_levels(points, dists):    
    assert(len(points)==len(dists)) 
    max_distances = np.array([p._max_distance for p in points])  
    ratios = max_distances/dists
    levels = np.ceil(np.log(ratios)/Frame.tracker.log_scale_factor).astype(np.intp)
    levels = np.clip(levels,0,Frame.tracker.num_levels-1)
    return levels    
