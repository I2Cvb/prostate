Script to apply C++ script
==========================

What is in there?
-----------------

* `resampling_original_data`: resample data using T2W information.
* `flip_gt`: flip up-side-down ground-truth.
* `reg_dce`: registration of DCE volume.

### `resampling`

* `resampling_dce_from_t2w.sh`: Resample all experimental data.

### `flip_gt`

* `flip_gt.sh`: Flip all the different GT.

### `reg_dce`

* `reg_dce.sh`: Register all experimental data. This is for intra-patient motion correction.
* `reg_gt.sh`: Register all experimental data. This for inter-modalities motion correction.

