
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
            "e.g. '0,80' for timepoint 0 to 80. NOTE THAT THE FRAMES ARE DENOTED BASED ON THE CALCIUM RESOLUTION, NOT BOLD.\n"
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
        '--CPCA_temporal_comp', type=int, default=-1,
        help=
            "Option for performing Complementary Principal Component Analysis (CPCA). Specify with this option how many extra \n"
            "subject-specific sources will be computed to account for non-prior confounds. This options \n"
            "specifies the number of temporal components to compute. After computing \n"
            "these sources, CPCA will provide a fit for each prior in --prior_maps indexed by --prior_bold_idx.\n"
            "Specify at least 0 extra sources to run CPCA.\n"
            "(default: %(default)s)\n"
            "\n"
        )
    analysis.add_argument(
        '--CPCA_spatial_comp', type=int, default=-1,
        help=
            "Same as --CPCA_temporal_comp, but specify how many spatial components to compute (which are \n"
            "additioned to the temporal components).\n"
            "(default: %(default)s)\n"
            "\n"
        )
    analysis.add_argument(
        '--optimize_CPCA', type=str,
        default='apply=false,min_prior_corr=0.5,diff_thresh_t=0.03,diff_thresh_s=0.03',
        help=
            "This option handles the automated dimensionality estimation when carrying out CPCA. The number of \n"
            "components specified with --CPCA_temporal_comp and --CPCA_spatial_comp will be first derived, and \n"
            "then an ideal dimensionality will be selected for temporal components and then for spatial \n"
            "components. A convergence report is generated to visualize the results across iterations. \n"
            "\n"
            "Convergence criterion 1: A The correlation between the fitted network component \n"
            "and the prior must reach a minimum.\n"
            "\n"
            "Convergence criterion 2: The last CPCA component which generated a sufficient difference in the output \n"
            "fitted network is selected. \n"
            "\n"
            "When multiple priors are fitted, the minimum dimensionality for all networks to respect the convergence \n"
            "criteria is selected.\n"
            "\n"
            "* apply: select 'true' to apply this option.\n"
            "*** Specify 'true' or 'false'. \n"
            "* min_prior_corr: Threshold for criterion 1. \n"
            "*** Must provide a float. \n"
            "* diff_thresh_t: Threshold for criterion 2 for temporal components. \n"
            "*** Must provide a float. \n"
            "* diff_thresh_s: Threshold for criterion 2 for spatial components. \n"
            "*** Must provide a float. \n"
            "(default: %(default)s)\n"
            "\n"
        )
    return parser

