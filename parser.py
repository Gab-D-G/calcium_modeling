
import argparse
def get_parser():
    """Build parser object"""
    parser = argparse.ArgumentParser(
        description=
            "Handles corrections for mesoscale calcium scans.",
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        '--calcium_file', action='store', type=str,
        help=
            "Full path to the Nifti file container 2D calcium timeseries."
        )
    parser.add_argument(
        '--output_dir', action='store', type=str,
        help=
            "Output folder."
        )
    parser.add_argument(
        '--overwrite', dest='overwrite', action='store_true',
        help=
            "Overwrite old outputs if present. \n"
            "(default: %(default)s)\n"
            "\n"
        )
    parser.add_argument(
        '--brain_mask_file', action='store', type=str,
        help=
            "Full path to the brain mask."
        )
    parser.add_argument(
        '--template_file', action='store', type=str, default=None,
        help=
            "Provide the path to an overlapping 2D anatomical template. Used for display purpose only.\n"
            "If none is provided, the mean will be computed from the calcium file as replacement.\n"
            "(default: %(default)s)\n"
            "\n"
        )
    parser.add_argument(
        '--confounds_file', action='store', type=str, default=None,
        help=
            "Provide a CSV file with the 6 motion parameters computed from a BOLD image, following the output format from RABIES.\n"
            "The parameters are not displayed in the diagnosis report if not provided. \n"
            "(default: %(default)s)\n"
            "\n"
        )
    parser.add_argument(
        '--FD_file', action='store', type=str, default=None,
        help=
            "Provide a CSV file with the framewise displacement estimate computed from a BOLD image, following the output format from RABIES.\n"
            "FD is not displayed in the diagnosis report if not provided, and censoring cannot be applied based on FD. \n"
            "(default: %(default)s)\n"
            "\n"
        )
       
    parser.add_argument(
        '--calcium_TR', type=float, default=0.1,
        help=
            "Specify repetition time (TR) in seconds for the calcium file.\n"
            "(default: %(default)s)\n"
            "\n"
        )
    parser.add_argument(
        '--BOLD_TR', type=float, default=1.0,
        help=
            "Specify repetition time (TR) in seconds for the BOLD image. This is used to estimate upsampling of the \n"
            "motion parameters inherited from BOLD to match the calcium TR. \n"
            "(default: %(default)s)\n"
            "\n"
        )
    parser.add_argument(
        '--crop_BOLD_upsampled', type=str, default=None,
        help=
            "With this option, can crop the motion parameters after upsampling if they don't match the calcium dimensions. \n"
            "e.g. '0,80' for timepoint 0 to 80.\n"
            "(default: %(default)s)\n"
            "\n"
        )
     
    confound_correction = parser.add_argument_group(
        title='Correction options', 
        description=
            "Options for correcting the calcium signal. \n"
        )
    confound_correction.add_argument(
        '--timeseries_interval', type=str, default='all',
        help=
            "Before confound correction, can crop the timeseries within a specific interval. Can be used for instance to remove photobleach in the first frames. \n"
            "e.g. '0,80' for timepoint 0 to 80.\n"
            "(default: %(default)s)\n"
            "\n"
        )


    confound_correction.add_argument(
        '--image_scaling', type=str,
        default="None",
        choices=["None", "global_variance", "voxelwise_standardization", 
                 "grand_mean_scaling", "voxelwise_mean"],
        help=
            "Image scaling options inherited from RABIES. \n"
            "(default: %(default)s)\n"
            "\n"
        )
    confound_correction.add_argument(
        '--detrending_order', type=str,
        default="linear",
        choices=["linear", "quadratic"],
        help=
            "Select between linear or quadratic (second-order) detrending of voxel timeseries.\n"
            "(default: %(default)s)\n"
            "\n"
        )

    confound_correction.add_argument(
        '--frame_censoring', type=str, default='FD_censoring=false,FD_threshold=0.05,DVARS_censoring=false,GS_censoring=false,minimum_timepoint=3',
        help=
            "Censor frames that are highly corrupted (i.e. 'scrubbing'). Operates as documented in RABIES, with the addition of the GS censoring option. \n"
            "(default: %(default)s)\n"
            "\n"
        )
    confound_correction.add_argument(
        '--n_pca_regress', type=int, default=0,
        help=
            "Number of PCA components to regress.\n"
            "(default: %(default)s)\n"
            "\n"
        )

    confound_correction.add_argument(
        '--highpass', type=float, default=None,
        help=
            "Specify highpass filter frequency.\n"
            "(default: %(default)s)\n"
            "\n"
        )
    confound_correction.add_argument(
        '--lowpass', type=float, default=None,
        help=
            "Specify lowpass filter frequency.\n"
            "(default: %(default)s)\n"
            "\n"
        )
    confound_correction.add_argument(
        '--edge_cutoff', type=float, default=0,
        help=
            "Specify the number of seconds to cut at beginning and end of acquisition if applying a\n"
            "frequency filter. Highpass filters generate edge effects at begining and end of the\n" 
            "timeseries. We recommend to cut those timepoints (around 30sec at both end for 0.01Hz \n" 
            "highpass.).\n"
            "(default: %(default)s)\n"
            "\n"
        )
    confound_correction.add_argument(
        '--apply_GSR', dest='apply_GSR', action='store_true',
        help=
            "Whether to apply global signal regression. \n"
            "(default: %(default)s)\n"
            "\n"
        )
    confound_correction.add_argument(
        '--smoothing_filter', type=float, default=None,
        help=
            "Specify filter size in mm for spatial smoothing. Will apply nilearn's function \n"
            "https://nilearn.github.io/modules/generated/nilearn.image.smooth_img.html\n"
            "(default: %(default)s)\n"
            "\n"
        )



    analysis = parser.add_argument_group(
        title='Analysis', 
        description=
            "Manage connectivity analysis. \n"
        )
    analysis.add_argument(
        '--seed_file', action='store', type=str, default=None,
        help=
            "Provide a mask defining a seed region. If provided, seed connectivity is computed.\n"
            "(default: %(default)s)\n"
            "\n"
        )
    analysis.add_argument(
        '--IC_file', action='store', type=str, default=None,
        help=
            "Provide an ICA decomposition. Dual regression is computed if provided, and this is needed for NPR.\n"
            "(default: %(default)s)\n"
            "\n"
        )
    analysis.add_argument(
        '--IC_network_idx', type=int,
        nargs="*",  # 0 or more values expected => creates a list
        default=[],
        help=
            "Specify the indices corresponding to networks to analyze from the --IC_file. \n"
            "IMPORTANT: index counting starts at 0 (i.e. the first component is selected with 0, not 1) \n"
            "(default: %(default)s)\n"
            "\n"
        )

    analysis.add_argument(
        '--optimize_NPR', type=str,
        default='apply=false,window_size=5,min_prior_corr=0.5,diff_thresh=0.03,max_iter=20,compute_max=false',
        help=
            "This option handles NPR, as documented on the RABIES pipeline. See RABIES documentation for details. \n"
            "(default: %(default)s)\n"
            "\n"
        )

    return parser

