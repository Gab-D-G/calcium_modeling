## What's computed with calcium_modeling.py

These scripts are designed to conduct signal correction for wide-field calcium imaging 2D timeseries. Additionally, connectivity analyses are made available, and diagnostic reports are generated along the cleaned timeseries to allow for data exploration and quality control. This is an adaptation of existing scripts part of the RABIES fMRI pipeline https://github.com/CoBrALab/RABIES.
<br>
Multiple correction are made available (the design of which is customized with the CLI arguments as shown below), and the main steps are executed in the following order (if selected):

1. Frame censoring (with FD, DVARS and/or GS)
2. Detrending
3. Highpass/lowpass filters (missing timepoints from censoring are handled through simulations as documented in the RABIES pipeline)
4. Removal of X first PCA components from the signal
5. Regression of the global signal
6. Image scaling
7. Spatial smoothing

## Execution

The script ```calcium_modeling.py``` can process one target scan at a time and should be directly executed to call the CLI parser to handle inputs (see --help below). The ```calcium_modeling.py``` script must share the same folder as supporting scripts from this repository. 
<br>

**Example syntax:**
```sh
python /path_to_script/calcium_modeling.py \
--calcium_file /path_to_file/calcium_file.nii.gz \
--confounds_file /path_to_file/confounds.csv \
--FD_file /path_to_file/FD_file.csv \
--output_dir /path_to_folder/ \
--brain_mask_file /path_to_file/mask_2d.nii.gz \
--calcium_TR 0.1 --BOLD_TR 1.0 --timeseries_interval 200,3000 \
--frame_censoring FD_censoring=false,FD_threshold=0.05,DVARS_censoring=true,GS_censoring=true,minimum_timepoint=3 \
--n_pca_regress 1 --highpass 0.01 --edge_cutoff 30 --apply_GSR \
--image_scaling voxelwise_standardization --smoothing_filter 0.3 \
--IC_file /path_to_file/melodic_IC.nii.gz --IC_network_idx 13 2 14 \
--optimize_NPR apply=true,window_size=5,min_prior_corr=0.5,diff_thresh=0.05,max_iter=20,compute_max=false
```

## Outputs generated

Outputs will be generated in the provided `--output_dir` folder and include:

* *_cleaned.nii.gz: the corrected calcium timeseries
* *_frame_censoring_mask.csv: label as True timepoints included after censoring operations were conducted.
* *_pca.png: displays the spatial map for the first 10 PCA components computed on the cleaned timeseries.
* *_temporal_diagnosis.png: contains a set of temporal features generated after correction of timeseries
* *_spatial_diagnosis.png: contains a set of spatial features generated after correction of timeseries
* *_analysis_dict.pkl: pickle file containing a dictionary with analysis outputs in numpy format
* mean.nii.gz: if no template_file was provided, this is the mean of the timeseries used for display.
* *_NPR_optimize.png: NPR report (only if NPR was conducted)

## CLI --help:

```sh
$ python calcium_modeling.py --help

usage: calcium_modeling.py [-h] [--calcium_file CALCIUM_FILE] [--output_dir OUTPUT_DIR] [--overwrite] [--brain_mask_file BRAIN_MASK_FILE]
                           [--template_file TEMPLATE_FILE] [--confounds_file CONFOUNDS_FILE] [--FD_file FD_FILE] [--calcium_TR CALCIUM_TR] [--BOLD_TR BOLD_TR]
                           [--crop_BOLD_upsampled CROP_BOLD_UPSAMPLED] [--timeseries_interval TIMESERIES_INTERVAL]
                           [--image_scaling {None,global_variance,voxelwise_standardization,grand_mean_scaling,voxelwise_mean}]
                           [--detrending_order {linear,quadratic}] [--frame_censoring FRAME_CENSORING] [--n_pca_regress N_PCA_REGRESS] [--highpass HIGHPASS]
                           [--lowpass LOWPASS] [--edge_cutoff EDGE_CUTOFF] [--apply_GSR] [--smoothing_filter SMOOTHING_FILTER] [--seed_file SEED_FILE]
                           [--IC_file IC_FILE] [--IC_network_idx [IC_NETWORK_IDX ...]] [--optimize_NPR OPTIMIZE_NPR]

Handles corrections for mesoscale calcium scans.

optional arguments:
  -h, --help            show this help message and exit
  --calcium_file CALCIUM_FILE
                        Full path to the Nifti file container 2D calcium timeseries.
  --output_dir OUTPUT_DIR
                        Output folder.
  --overwrite           Overwrite old outputs if present. 
                        (default: False)
                        
  --brain_mask_file BRAIN_MASK_FILE
                        Full path to the brain mask.
  --template_file TEMPLATE_FILE
                        Provide the path to an overlapping 2D anatomical template. Used for display purpose only.
                        If none is provided, the mean will be computed from the calcium file as replacement.
                        (default: None)
                        
  --confounds_file CONFOUNDS_FILE
                        Provide a CSV file with the 6 motion parameters computed from a BOLD image, following the output format from RABIES.
                        The parameters are not displayed in the diagnosis report if not provided. 
                        (default: None)
                        
  --FD_file FD_FILE     Provide a CSV file with the framewise displacement estimate computed from a BOLD image, following the output format from RABIES.
                        FD is not displayed in the diagnosis report if not provided, and censoring cannot be applied based on FD. 
                        (default: None)
                        
  --calcium_TR CALCIUM_TR
                        Specify repetition time (TR) in seconds for the calcium file.
                        (default: 0.1)
                        
  --BOLD_TR BOLD_TR     Specify repetition time (TR) in seconds for the BOLD image. This is used to estimate upsampling of the 
                        motion parameters inherited from BOLD to match the calcium TR. 
                        (default: 1.0)
                        
  --crop_BOLD_upsampled CROP_BOLD_UPSAMPLED
                        With this option, can crop the motion parameters after upsampling if they don't match the calcium dimensions. 
                        e.g. '0,80' for timepoint 0 to 80.
                        (default: None)
                        

Correction options:
  Options for correcting the calcium signal. 

  --timeseries_interval TIMESERIES_INTERVAL
                        Before confound correction, can crop the timeseries within a specific interval. Can be used for instance to remove photobleach in the first frames. 
                        e.g. '0,80' for timepoint 0 to 80.
                        (default: all)
                        
  --image_scaling {None,global_variance,voxelwise_standardization,grand_mean_scaling,voxelwise_mean}
                        Image scaling options inherited from RABIES. 
                        (default: None)
                        
  --detrending_order {linear,quadratic}
                        Select between linear or quadratic (second-order) detrending of voxel timeseries.
                        (default: linear)
                        
  --frame_censoring FRAME_CENSORING
                        Censor frames that are highly corrupted (i.e. 'scrubbing'). Operates as documented in RABIES, with the addition of the GS censoring option. 
                        (default: FD_censoring=false,FD_threshold=0.05,DVARS_censoring=false,GS_censoring=false,minimum_timepoint=3)
                        
  --n_pca_regress N_PCA_REGRESS
                        Number of PCA components to regress.
                        (default: 0)
                        
  --highpass HIGHPASS   Specify highpass filter frequency.
                        (default: None)
                        
  --lowpass LOWPASS     Specify lowpass filter frequency.
                        (default: None)
                        
  --edge_cutoff EDGE_CUTOFF
                        Specify the number of seconds to cut at beginning and end of acquisition if applying a
                        frequency filter. Highpass filters generate edge effects at begining and end of the
                        timeseries. We recommend to cut those timepoints (around 30sec at both end for 0.01Hz 
                        highpass.).
                        (default: 0)
                        
  --apply_GSR           Whether to apply global signal regression. 
                        (default: False)
                        
  --smoothing_filter SMOOTHING_FILTER
                        Specify filter size in mm for spatial smoothing. Will apply nilearn's function 
                        https://nilearn.github.io/modules/generated/nilearn.image.smooth_img.html
                        (default: None)
                        

Analysis:
  Manage connectivity analysis. 

  --seed_file SEED_FILE
                        Provide a mask defining a seed region. If provided, seed connectivity is computed.
                        (default: None)
                        
  --IC_file IC_FILE     Provide an ICA decomposition. Dual regression is computed if provided, and this is needed for NPR.
                        (default: None)
                        
  --IC_network_idx [IC_NETWORK_IDX ...]
                        Specify the indices corresponding to networks to analyze from the --IC_file. 
                        IMPORTANT: index counting starts at 0 (i.e. the first component is selected with 0, not 1) 
                        (default: [])
                        
  --optimize_NPR OPTIMIZE_NPR
                        This option handles NPR, as documented on the RABIES pipeline. See RABIES documentation for details. 
                        (default: apply=false,window_size=5,min_prior_corr=0.5,diff_thresh=0.03,max_iter=20,compute_max=false)
```
