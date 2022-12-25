# Monocular VO and SLAM sample

Main Scripts:
* `test_vo.py` estimates current camera pose with respect to the previous one. The norm of translation vector is 1 because groundtruth is not input and the trajectory is not in actual scale.

* `test_slam.py` [Under debugging] estimates current camera poase and 3D feature points. It includes components of visual SLAM: initialization, tracking, local mapping and bundle adjustment. 

* `calibration.py` calibrate camera intrinsic parameters using chessboard pattern.

* `test_pangolin.py` is the sample code to detect AR marker and the 3D camera coordinate using Pangolin.


# Dependencies

* Python >3.6.9
* Numpy >1.18.2
* OpenCV3
* [Pangolin](https://github.com/uoip/pangolin)