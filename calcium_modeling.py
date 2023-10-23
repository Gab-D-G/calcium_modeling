import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from rabies.utils import recover_3D
import pickle
import pathlib

import pandas as pd

from parser import get_parser
from modeling_functions import *

####################################### INPUTS

parser = get_parser()
opts = parser.parse_args()

args = 'CLI INPUTS: \n'
for arg in vars(opts):
    input = f'-> {arg} = {getattr(opts, arg)} \n'
    args += input
print(args)

out_dir = opts.output_dir
if not os.path.isdir(out_dir):
    raise ValueError(f"{out_dir} from --output_dir does not exist.")

for f,arg in zip([opts.calcium_file,opts.confounds_file,opts.FD_file,opts.brain_mask_file,opts.template_file,opts.seed_file,opts.IC_file,],
                 ['--calcium_file','--confounds_file','--FD_file','--brain_mask_file','--template_file','--seed_file','--IC_file',]):
    if not f is None:
        if not os.path.isfile(f):
            raise ValueError(f"{f} from {arg} does not exist.")

from rabies.parser import parse_argument
frame_censoring = parse_argument(opt=opts.frame_censoring, 
    key_value_pairs = {'FD_censoring':['true', 'false'], 'FD_threshold':float, 'DVARS_censoring':['true', 'false'],
        'GS_censoring':['true', 'false'], 'minimum_timepoint':int},
    name='frame_censoring')

optimize_NPR = parse_argument(opt=opts.optimize_NPR, 
    key_value_pairs = {'apply':['true', 'false'], 'window_size':int, 'min_prior_corr':float,
                        'diff_thresh':float, 'max_iter':int, 'compute_max':['true', 'false']},
    name='optimize_NPR')

#######################################

import pathlib  # Better path manipulation
filename_split = pathlib.Path(opts.calcium_file).name.rsplit(".nii")

fig_pca_filename = f'{out_dir}/{filename_split[0]}_pca.png'
fig_temporal_filename = f'{out_dir}/{filename_split[0]}_temporal_diagnosis.png'
fig_spatial_filename = f'{out_dir}/{filename_split[0]}_spatial_diagnosis.png'
analysis_dict_file=f'{out_dir}/{filename_split[0]}_analysis_dict.pkl'
cleaned_path=f'{out_dir}/{filename_split[0]}_cleaned.nii.gz'

if os.path.isfile(fig_pca_filename):
    if os.path.isfile(fig_temporal_filename):
        if os.path.isfile(fig_spatial_filename):
            if os.path.isfile(analysis_dict_file):
                if os.path.isfile(cleaned_path):
                    print('All output files already exist.')
                    if not opts.overwrite:
                        print('Exiting.')
                        sys.exit()
                    else:
                        print('Previous outputs will be overwritten.')



mask_img = sitk.ReadImage(opts.brain_mask_file)
volume_idx = sitk.GetArrayFromImage(mask_img).astype(bool)

calcium_img = sitk.ReadImage(opts.calcium_file, sitk.sitkFloat32)
data_array = sitk.GetArrayFromImage(calcium_img)
num_volumes = data_array.shape[0]
timeseries = np.zeros([num_volumes, volume_idx.sum()])
for i in range(num_volumes):
    timeseries[i, :] = (data_array[i, :, :, :])[volume_idx]

if opts.template_file is None:
    mean = timeseries.mean(axis=0)
    mean_img = recover_3D(
        opts.brain_mask_file, mean)
    template_file = f'{out_dir}/mean.nii.gz'
    sitk.WriteImage(mean_img, template_file)
else:
    template_file = opts.template_file

# select the subset of timeseries specified
if not opts.timeseries_interval == 'all':
    lowcut = int(opts.timeseries_interval.split(',')[0])
    highcut = int(opts.timeseries_interval.split(',')[1])
    time_range = range(lowcut,highcut)
else:
    time_range = range(num_volumes)

# select the subset of timeseries specified
if not opts.crop_BOLD_upsampled is None:
    lowcut = int(opts.crop_BOLD_upsampled.split(',')[0])
    highcut = int(opts.crop_BOLD_upsampled.split(',')[1])
    crop_BOLD_upsampled = range(lowcut,highcut)
else:
    crop_BOLD_upsampled = None

'''
Up-sample the fMRI-derived motion estimates to match the calcium
'''
if (not opts.confounds_file is None) and (not opts.FD_file is None):

    from rabies.confound_correction_pkg.utils import select_motion_regressors
    confounds_array_mot6_BOLD = select_motion_regressors(conf_list=['mot_6'],motion_params_csv=opts.confounds_file)

    num_frame_BOLD = confounds_array_mot6_BOLD.shape[0]
    num_frame_calcium = calcium_img.GetSize()[3]

    upsampling_ratio = int(opts.BOLD_TR/opts.calcium_TR)

    print(f"The BOLD-derived motion timecourse will be upsampled by {upsampling_ratio} from {num_frame_BOLD} frames to {num_frame_BOLD*upsampling_ratio} frames to match \
    the calcium modality, which contains {num_frame_calcium} frames.")


    if (not num_frame_calcium==num_frame_BOLD*upsampling_ratio) and (crop_BOLD_upsampled is None):
        raise ValueError("The length of the calcium and upsampled BOLD timecourses do not match. Consider using crop_BOLD_upsampled to manually crop the BOLD timeseries.")

    confounds_array_mot6 = np.repeat(confounds_array_mot6_BOLD,upsampling_ratio, axis=0)

    FD_trace_BOLD = pd.read_csv(opts.FD_file).get('Mean').to_numpy()
    FD_trace = np.repeat(FD_trace_BOLD,upsampling_ratio)

    if not crop_BOLD_upsampled is None:
        FD_trace = FD_trace[crop_BOLD_upsampled]
        confounds_array_mot6 = confounds_array_mot6[crop_BOLD_upsampled,:]    
        if not len(FD_trace)==num_frame_calcium:
            raise ValueError(f"The length of the calcium ({num_frame_calcium} frames) and upsampled BOLD timecourses ({len(FD_trace)} frames) do not match, even after applying crop_BOLD_upsampled.")
else:
    FD_trace = None
    confounds_array_mot6 = None
    
'''
Denoising
'''

timeseries,VE_spatial,temporal_std,predicted_std,CR_data_dict =  regress_calcium(
    timeseries, opts.calcium_file, opts.brain_mask_file, cleaned_path, FD_trace,confounds_array_mot6,opts.calcium_TR,time_range,
    image_scaling=opts.image_scaling,FD_censoring=frame_censoring['FD_censoring'], FD_threshold=frame_censoring['FD_threshold'], DVARS_censoring=frame_censoring['DVARS_censoring'], GS_censoring=frame_censoring['GS_censoring'],minimum_timepoint=frame_censoring['minimum_timepoint'],match_number_timepoints=False,
    n_pca_regress=opts.n_pca_regress,highpass=opts.highpass,lowpass=opts.lowpass,apply_GSR=opts.apply_GSR,edge_cutoff=opts.edge_cutoff,detrending_order=opts.detrending_order, smoothing_filter=opts.smoothing_filter)

'''
Generate PCA report after denoising was applied
'''

from sklearn.decomposition import PCA

n=10

pca = PCA(n_components=n)
transformed = pca.fit_transform(timeseries)

fig_pca, axes = plt.subplots(nrows=2, ncols=7, figsize=(4*7, 4*2))
ax_extra = fig_pca.add_subplot(1,4,4)
ax_extra.plot(pca.explained_variance_ratio_)
ax_extra.set_ylabel('Variance explained', fontsize=20)
ax_extra.set_ylabel('Component number', fontsize=20)
plt.setp(ax_extra.get_xticklabels(), fontsize=15)
plt.setp(ax_extra.get_yticklabels(), fontsize=15)
plt.tight_layout()

axes[0,5].axis('off')
axes[0,6].axis('off')
axes[1,5].axis('off')
axes[1,6].axis('off')

for i in range(n):
    sitk_img = recover_3D(
        opts.brain_mask_file, pca.components_[i,:])
    
    ax = axes[int(i/5),i%5]
    
    vector = pca.components_[i,:].flatten()
    vector.sort()
    vmax = vector[int(len(vector)*0.95)]
    cbar = plot_2d(ax,sitk_img,fig_pca,vmin=-vmax,vmax=vmax,cmap='cold_hot', alpha=1, cbar=True, threshold=None)


'''
Compute connectivity
'''

from rabies.analysis_pkg.analysis_math import dual_regression,vcorrcoef

analysis_dict={}

if not opts.seed_file is None:
    roi_mask = sitk.GetArrayFromImage(sitk.ReadImage(opts.seed_file))[volume_idx].astype(bool)

    # extract the voxel timeseries within the mask, and take the mean ROI timeseries
    seed_timeseries = timeseries[:,roi_mask].mean(axis=1)
    seed_timeseries /= np.sqrt((seed_timeseries ** 2).sum(axis=0)) # the temporal domain is variance-normalized so that the weights are contained in the spatial maps
    seed_corrs = vcorrcoef(timeseries.T, seed_timeseries)
    seed_corrs[np.isnan(seed_corrs)] = 0

    analysis_dict['SBC']={}
    analysis_dict['SBC']['signal_trace'] = seed_timeseries.reshape(-1,1)
    analysis_dict['SBC']['FC_maps'] = seed_corrs.reshape(1,-1)

if not opts.IC_file is None:
    if len(opts.IC_network_idx)==0:
        raise ValueError(f"--IC_network_idx must be specified if a IC_file is provided.")
    calcium_ICs = sitk.GetArrayFromImage(sitk.ReadImage(opts.IC_file))[:,volume_idx]
    IC_vectors = calcium_ICs

    DR_full = dual_regression(IC_vectors, timeseries)

    analysis_dict['DR']={}
    analysis_dict['DR']['signal_trace'] = DR_full['W'][:,opts.IC_network_idx]
    analysis_dict['DR']['FC_maps'] = (DR_full['C'][:,opts.IC_network_idx]*DR_full['S'][opts.IC_network_idx]).T


    if optimize_NPR['apply']:
        from rabies.analysis_pkg.analysis_functions import spatiotemporal_fit_converge
        C_prior = calcium_ICs[opts.IC_network_idx,:].T
        modeling,_,_,_,optimize_report_fig = spatiotemporal_fit_converge(timeseries,C_prior,
                                window_size=optimize_NPR['window_size'],
                                min_prior_corr=optimize_NPR['min_prior_corr'],
                                diff_thresh=optimize_NPR['diff_thresh'],
                                max_iter=optimize_NPR['max_iter'], 
                                compute_max=optimize_NPR['compute_max'], 
                                gen_report=True)
        
        optimize_report_file = os.path.abspath(f'{out_dir}/{filename_split[0]}_NPR_optimize.png')
        optimize_report_fig.savefig(optimize_report_file, bbox_inches='tight')

        C_fit = modeling['C_fitted_prior']*modeling['S_fitted_prior']
        analysis_dict['NPR']={}
        analysis_dict['NPR']['signal_trace'] = modeling['W_fitted_prior']
        analysis_dict['NPR']['FC_maps'] = C_fit.T

'''
spatiotemporal diagnosis
'''

temporal_info,spatial_info = process_data(timeseries,VE_spatial,temporal_std,predicted_std,CR_data_dict)    

fig,fig2 = scan_diagnosis_calcium(timeseries, opts.brain_mask_file, template_file, CR_data_dict,spatial_info,analysis_dict)


'''
Save outputs
'''


fig_pca.savefig(fig_pca_filename, bbox_inches='tight')
fig.savefig(fig_temporal_filename, bbox_inches='tight')
fig2.savefig(fig_spatial_filename, bbox_inches='tight')

# save censored frame mask as CSV
frame_mask_file = out_dir+'/'+filename_split[0]+'_frame_censoring_mask.csv'
pd.DataFrame(CR_data_dict['frame_mask']).to_csv(frame_mask_file, index=False, header=['False = Masked Frames'])

analysis_dict['CR_data_dict'] = CR_data_dict

with open(analysis_dict_file, 'wb') as handle:
    pickle.dump(analysis_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        