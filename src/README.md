C++ pre-processing
==================

What is in there?
-----------------

* `resampling`: resample data using T2W information.
* `flip_gt`: flip up-side-down ground-truth.
* `reg_dce`: registration of DCE volume.

### `resampling`

* `resampling_dce_from_t2w.cxx`: ITK script to resample each DCE volume using the T2W information.

### `flip_gt`

* `flip_gt.cxx`: ITK script to flip up-side-down the GT generated through MATLAB. It was an error between the original space of the T2W and the resaved DCE using ITK. This script should not have to be reused but we keep it just in case.

### `reg_dce`

* `reg_dce.cxx`: ITK script to register each serie in the DCE modality. The metric used is the Mattes mutual information optimized through a regular step gradient descent. Only a rigid registration is performed.
* `reg_gt.cxx`: ITK script to register the ground-truth of T2W and DCE and apply the transform found on the DCE series. The metric used is the the mean squared metric optimized through a regular step gradient descent. We used a rigid registration follow by a coarse and fine bspline deformable registration.

