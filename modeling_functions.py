import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from rabies.utils import recover_3D,recover_4D


from sklearn.decomposition import PCA
def pca_regress(timeseries, n_components):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(timeseries)
    regressed = timeseries - transformed.dot(pca.components_)
    return regressed

def regress_calcium(timeseries, calcium_file, brain_mask_file, cleaned_path, FD_trace,confounds_array_mot6,TR,time_range,
            image_scaling='None',FD_censoring=False, FD_threshold=0.05, DVARS_censoring=False, GS_censoring=False, minimum_timepoint=3,match_number_timepoints=False,
            n_pca_regress=0,highpass=None,lowpass=None,apply_GSR=False,edge_cutoff=0,detrending_order='linear', smoothing_filter=None):
    from rabies.confound_correction_pkg.utils import lombscargle_fill, butterworth, smooth_image, remove_trend
    from rabies.analysis_pkg.analysis_functions import closed_form

    num_volumes = timeseries.shape[0]
    FD_trace_none = False
    confounds_array_mot6_none = False
    if FD_trace is None:
        FD_trace = np.zeros(num_volumes)
        FD_trace_none = True
    if confounds_array_mot6 is None:
        confounds_array_mot6 = np.repeat(np.zeros(num_volumes).reshape(-1,1),6,axis=1)
        confounds_array_mot6_none = True


    # apply time_range    
    timeseries = timeseries[time_range,:]
    confounds_array_mot6 = confounds_array_mot6[time_range,:]
    FD_trace = FD_trace[time_range]
    

    '''
    #1 - Compute and apply frame censoring mask (from FD and/or DVARS thresholds)
    '''
    frame_mask,FD_trace,DVARS = temporal_censoring(timeseries, FD_trace, 
            FD_censoring, FD_threshold, DVARS_censoring, GS_censoring, minimum_timepoint)
    
    '''
    #2 - If --match_number_timepoints is selected, each scan is matched to the defined minimum_timepoint number of frames.
    '''
    if match_number_timepoints:
        if (not highpass is None) or (not lowpass is None):
            # if frequency filtering is applied, avoid selecting timepoints that would be removed with --edge_cutoff
            num_cut = int(edge_cutoff/TR)
            if not num_cut==0:
                frame_mask[:num_cut]=0
                frame_mask[-num_cut:]=0

                if frame_mask.sum()<int(minimum_timepoint):
                    return 

        # randomly shuffle indices that haven't been censored, then remove an extra subset above --minimum_timepoint
        num_timepoints = len(frame_mask)
        time_idx=np.array(range(num_timepoints))
        perm = np.random.permutation(time_idx[frame_mask])
        # selecting the subset of extra timepoints, and censoring them
        subset_idx = perm[minimum_timepoint:]
        frame_mask[subset_idx]=0
        # keep track of the original number of timepoints for tDOF estimation, to evaluate latter if the correction was succesful
        number_extra_timepoints = len(subset_idx)
    else:
        number_extra_timepoints = 0

    timeseries = timeseries[frame_mask]
    confounds_array_mot6 = confounds_array_mot6[frame_mask]

    '''
    #3 - Linear/Quadratic detrending of fMRI timeseries and nuisance regressors
    '''
    # apply detrending, after censoring
    if detrending_order=='linear':
        second_order=False
    elif detrending_order=='quadratic':
        second_order=True
    else:
        raise ValueError(f"--detrending_order must be 'linear' or 'quadratic', not {detrending_order}")

    # save grand mean prior to detrending
    timeseries_ = remove_trend(timeseries, frame_mask, second_order=second_order, keep_intercept=True)
    grand_mean = timeseries_.mean()
    voxelwise_mean = timeseries_.mean(axis=0)

    timeseries = remove_trend(timeseries, frame_mask, second_order=second_order, keep_intercept=False)
    confounds_array_mot6 = remove_trend(confounds_array_mot6, frame_mask, second_order=second_order, keep_intercept=False)    

    if (not highpass is None) or (not lowpass is None):
        '''
        #5 - If frequency filtering and frame censoring are applied, simulate data in censored timepoints using the Lomb-Scargle periodogram, 
            as suggested in Power et al. (2014, Neuroimage), for both the fMRI timeseries and nuisance regressors prior to filtering.
        '''
        timeseries_filled = lombscargle_fill(x=timeseries,time_step=TR,time_mask=frame_mask)
        confounds_filled = lombscargle_fill(x=confounds_array_mot6,time_step=TR,time_mask=frame_mask)

        '''
        #6 - As recommended in Lindquist et al. (2019, Human brain mapping), make the nuisance regressors orthogonal
            to the temporal filter.
        '''
        confounds_filtered = butterworth(confounds_filled, TR=TR,
                                high_pass=highpass, low_pass=lowpass)

        '''
        #7 - Apply highpass and/or lowpass filtering on the fMRI timeseries (with simulated timepoints).
        '''

        timeseries_filtered = butterworth(timeseries_filled, TR=TR,
                                high_pass=highpass, low_pass=lowpass)

        # correct for edge effects of the filters
        num_cut = int(edge_cutoff/TR)
        if len(frame_mask)<2*num_cut:
            raise ValueError(f"The timeseries are too short to remove {edge_cutoff}sec of data at each edge.")

        if not num_cut==0:
            frame_mask[:num_cut]=0
            frame_mask[-num_cut:]=0


        '''
        #8 - Re-apply the frame censoring mask onto filtered fMRI timeseries and nuisance regressors, taking out the
            simulated timepoints. Edge artefacts from frequency filtering can also be removed as recommended in Power et al. (2014, Neuroimage).
        '''
        # re-apply the masks to take out simulated data points, and take off the edges
        timeseries = timeseries_filtered[frame_mask]
        confounds_array_mot6 = confounds_filtered[frame_mask]

    if frame_mask.sum()<int(minimum_timepoint):
        return 


    # voxels that have a NaN value are set to 0
    nan_voxels = np.isnan(timeseries).sum(axis=0)>1
    timeseries[:,nan_voxels] = 0
    
    '''
    ### - Regress principal components
    '''
    if n_pca_regress>0:
        timeseries = pca_regress(timeseries, n_components=n_pca_regress)

    '''
    #9 - Apply confound regression using the selected nuisance regressors.
    '''    

    # always apply GSR for CR estimates
    confounds_array=timeseries.mean(axis=1).reshape(-1,1)
    X=confounds_array
    Y=timeseries
    try:
        predicted = X.dot(closed_form(X,Y))
        res = Y-predicted
    except:
        return 

    VE_spatial = 1-(res.var(axis=0)/Y.var(axis=0))
    VE_temporal = 1-(res.var(axis=1)/Y.var(axis=1))

    if apply_GSR:
        # if confound regression is applied
        timeseries = res

    '''
    #10 - Scaling of timeseries variance.
    '''
    if image_scaling=='global_variance':
        scaling_factor = timeseries.std()
        timeseries = timeseries/scaling_factor
        # we scale also the variance estimates from CR
        predicted = predicted/scaling_factor
    elif image_scaling=='grand_mean_scaling':
        scaling_factor = grand_mean
        timeseries = timeseries/scaling_factor
        timeseries *= 100 # we scale BOLD in % fluctuations
        # we scale also the variance estimates from CR
        predicted = predicted/scaling_factor
    elif image_scaling=='voxelwise_standardization':
        # each voxel is scaled according to its STD
        temporal_std = timeseries.std(axis=0) 
        timeseries = timeseries/temporal_std
        nan_voxels = np.isnan(timeseries).sum(axis=0)>1
        timeseries[:,nan_voxels] = 0
        # we scale also the variance estimates from CR
        predicted = predicted/temporal_std
        nan_voxels = np.isnan(predicted).sum(axis=0)>1
        predicted[:,nan_voxels] = 0
    elif image_scaling=='voxelwise_mean':
        # each voxel is scaled according to its mean
        timeseries = timeseries/voxelwise_mean
        nan_voxels = np.isnan(timeseries).sum(axis=0)>1
        timeseries[:,nan_voxels] = 0
        timeseries *= 100 # we scale BOLD in % fluctuations

        # we scale also the variance estimates from CR
        predicted = predicted/voxelwise_mean
        nan_voxels = np.isnan(predicted).sum(axis=0)>1
        predicted[:,nan_voxels] = 0

    # after variance scaling, compute the variability estimates
    temporal_std = timeseries.std(axis=0)
    predicted_std = predicted.std(axis=0)
    predicted_time = np.sqrt((predicted.T**2).mean(axis=0))
    predicted_global_std = predicted.std()

    # apply the frame mask to FD trace/DVARS
    DVARS = DVARS[frame_mask]
    FD_trace = FD_trace[frame_mask]

    # calculate temporal degrees of freedom left after confound correction
    num_timepoints = frame_mask.sum()
    num_regressors = confounds_array.shape[1]
    tDOF = num_timepoints - num_regressors + number_extra_timepoints

    data_dict = {'FD_trace':FD_trace, 'DVARS':DVARS, 'time_range':time_range, 'frame_mask':frame_mask, 'confounds_array_mot6':confounds_array_mot6, 'VE_temporal':VE_temporal, 'predicted_time':predicted_time, 'tDOF':tDOF, 'CR_global_std':predicted_global_std}
    if FD_trace_none:
        data_dict['FD_trace'] = None
    if confounds_array_mot6_none:
        data_dict['confounds_array_mot6'] = None
    
    timeseries_img = recover_4D(brain_mask_file, timeseries, calcium_file)
    
    if smoothing_filter is not None:
        '''
        #12 - Apply Gaussian spatial smoothing.
        '''

        import nibabel as nb
        affine = nb.load(calcium_file).affine[:3,:3] # still not sure how to match nibabel's affine reliably
        mask_img = sitk.ReadImage(brain_mask_file, sitk.sitkFloat32)
        timeseries_img = smooth_image(timeseries_img, affine, smoothing_filter, mask_img)
    
    sitk.WriteImage(timeseries_img, cleaned_path)

    return timeseries,VE_spatial,temporal_std,predicted_std,data_dict


def temporal_censoring(timeseries, FD_trace, 
        FD_censoring, FD_threshold, DVARS_censoring, GS_censoring, minimum_timepoint):
    from rabies.confound_correction_pkg.utils import gen_FD_mask

    # compute the DVARS before denoising
    derivative=np.concatenate((np.empty([1,timeseries.shape[1]]),timeseries[1:,:]-timeseries[:-1,:]))
    DVARS=np.sqrt((derivative**2).mean(axis=1))
    global_signal = timeseries.mean(axis=1)

    # apply the temporal censoring
    frame_mask = np.ones(timeseries.shape[0]).astype(bool)
    if FD_censoring:
        FD_mask = gen_FD_mask(FD_trace, FD_threshold)
        frame_mask*=FD_mask
        
    for trace,censoring in zip([global_signal,DVARS],[GS_censoring,DVARS_censoring]):
        if censoring:
            # create a distribution where no timepoint falls more than 2.5 STD away from the mean
            mask1=np.zeros(len(trace)).astype(bool)
            mask2=np.ones(len(trace)).astype(bool)
            mask2[0]=False # remove the first timepoint, which is always 0
            while ((mask2!=mask1).sum()>0):
                mask1=mask2
                mean=trace[mask1].mean()
                std=trace[mask1].std()
                norm=(trace-mean)/std
                mask2=np.abs(norm)<2.5
            frame_mask*=mask2
    if frame_mask.sum()<int(minimum_timepoint):
        from nipype import logging
        log = logging.getLogger('nipype.workflow')
        log.warning(f"FD/DVARS CENSORING LEFT LESS THAN {str(minimum_timepoint)} VOLUMES. THIS SCAN WILL BE REMOVED FROM FURTHER PROCESSING.")
        return None,None,None

    return frame_mask,FD_trace,DVARS


def process_data(timeseries,VE_spatial,temporal_std,predicted_std,CR_data_dict):
    temporal_info = {}
    spatial_info = {}


    '''Temporal Features'''
    '''
    DR_W = np.array(pd.read_csv(analysis_dict['dual_regression_timecourse_csv'], header=None))
    DR_array = sitk.GetArrayFromImage(
        sitk.ReadImage(analysis_dict['dual_regression_nii']))
    DR_C = np.zeros([DR_array.shape[0], volume_indices.sum()])
    for i in range(DR_array.shape[0]):
        DR_C[i, :] = (DR_array[i, :, :, :])[volume_indices]

    temporal_info['DR_all'] = DR_W

    signal_trace = np.abs(DR_W[:, prior_bold_idx]).mean(axis=1)
    noise_trace = np.abs(DR_W[:, prior_confound_idx]).mean(axis=1)
    temporal_info['signal_trace'] = signal_trace
    temporal_info['noise_trace'] = noise_trace
    '''

    '''Spatial Features'''
    global_signal = timeseries.mean(axis=1)
    GS_cov = (global_signal.reshape(-1,1)*timeseries).mean(axis=0) # calculate the covariance between global signal and each voxel

    '''
    prior_fit_out = {'C': [], 'W': []}
    if (NPR_temporal_comp>-1) or (NPR_spatial_comp>-1):
        prior_fit_out['W'] = np.array(pd.read_csv(analysis_dict['NPR_prior_timecourse_csv'], header=None))
        C_array = sitk.GetArrayFromImage(
            sitk.ReadImage(analysis_dict['NPR_prior_filename']))
        C = np.zeros([C_array.shape[0], volume_indices.sum()])
        for i in range(C_array.shape[0]):
            C[i, :] = (C_array[i, :, :, :])[volume_indices]
        prior_fit_out['C'] = C


    spatial_info['prior_maps'] = data_dict['prior_map_vectors'][prior_bold_idx]
    spatial_info['DR_BOLD'] = DR_C[prior_bold_idx]
    spatial_info['DR_all'] = DR_C

    spatial_info['NPR_maps'] = prior_fit_out['C']
    temporal_info['NPR_time'] = prior_fit_out['W']
    '''

    spatial_info['VE_spatial'] = VE_spatial
    spatial_info['temporal_std'] = temporal_std
    spatial_info['predicted_std'] = predicted_std
    spatial_info['GS_cov'] = GS_cov
    
    spatial_info['DR_BOLD'] = np.array([])
    spatial_info['NPR_maps'] = np.array([])

    '''
    if len(prior_fit_out['W'])>0:
        NPR_prior_W = np.array(pd.read_csv(analysis_dict['NPR_prior_timecourse_csv'], header=None))
        NPR_extra_W = np.array(pd.read_csv(analysis_dict['NPR_extra_timecourse_csv'], header=None))
        temporal_info['NPR_prior_trace'] = np.abs(NPR_prior_W).mean(axis=1)
        temporal_info['NPR_noise_trace'] = np.abs(NPR_extra_W).mean(axis=1)
    '''

    return temporal_info,spatial_info




import nilearn.plotting
from rabies.visualization import plot_3d,otsu_scaling

def plot_2d(ax,sitk_img,fig,vmin=0,vmax=1,cmap='gray', alpha=1, cbar=False, threshold=None):
    physical_dimensions = (np.array(sitk_img.GetSpacing())*np.array(sitk_img.GetSize()))[::-1] # invert because the array is inverted indices
    array=sitk.GetArrayFromImage(sitk_img)

    array[array==0]=None # set 0 values to be empty

    if not threshold is None:
        array[np.abs(array)<threshold]=None

    slice=array[0,:,:]
    pos = ax.imshow(slice, extent=[0,physical_dimensions[2],0,physical_dimensions[1]], vmin=vmin, vmax=vmax,cmap=cmap, alpha=alpha, interpolation='none')
    ax.axis('off')
    if cbar:
        cbar = fig.colorbar(pos, ax=ax)
    return cbar

from rabies.analysis_pkg.diagnosis_pkg.diagnosis_functions import grayplot

def scan_diagnosis_calcium(timeseries, mask_file, template_file, CR_data_dict, spatial_info,analysis_dict):
    
    global_signal = timeseries.mean(axis=1)
    
    
    fig = plt.figure(figsize=(6, 18))
    #fig.suptitle(name, fontsize=30, color='white')
    
    ax0 = fig.add_subplot(3,1,1)
    ax1 = fig.add_subplot(12,1,5)
    ax1_ = fig.add_subplot(12,1,6)
    ax2 = fig.add_subplot(6,1,4)
    ax3 = fig.add_subplot(6,1,5)
    ax4 = fig.add_subplot(6,1,6)

    im = grayplot(timeseries, ax0)

    ax0.set_ylabel('Voxels', fontsize=20)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.axes.get_yaxis().set_ticks([])
    plt.setp(ax0.get_xticklabels(), fontsize=15)
    plt.setp(ax1.get_xticklabels(), visible=False)

    num_timepoints = timeseries.shape[0]
    x = range(num_timepoints)
    ax0.set_xlim([0, num_timepoints-1])
    ax1.set_xlim([0, num_timepoints-1])
    ax1_.set_xlim([0, num_timepoints-1])
    ax2.set_xlim([0, num_timepoints-1])
    ax3.set_xlim([0, num_timepoints-1])
    ax4.set_xlim([0, num_timepoints-1])

    if not CR_data_dict['confounds_array_mot6'] is None:
        # plot the motion timecourses
        confounds_array_mot6 = CR_data_dict['confounds_array_mot6']
        # take proper subset of timepoints
        ax1.plot(x,confounds_array_mot6[:,0])
        ax1.plot(x,confounds_array_mot6[:,1])
        ax1.plot(x,confounds_array_mot6[:,2])

        ax1_.plot(x,confounds_array_mot6[:,3])
        ax1_.plot(x,confounds_array_mot6[:,4])
        ax1_.plot(x,confounds_array_mot6[:,5])

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.setp(ax1_.get_xticklabels(), visible=False)
    ax1_.spines['right'].set_visible(False)
    ax1_.spines['top'].set_visible(False)

    if not CR_data_dict['FD_trace'] is None:
        y = CR_data_dict['FD_trace']
        ax2.plot(x,y, 'r')
        ax2.set_ylabel('FD in mm', fontsize=20)
        ax2.legend(['Framewise \nDisplacement (FD)'
                    ], loc='center left', fontsize=15, bbox_to_anchor=(1.15, 0.7))

    DVARS = CR_data_dict['DVARS']
    DVARS[0] = None
    ax2_ = ax2.twinx()
    y2 = DVARS
    ax2_.plot(x,y2, 'b')
    ax2_.set_ylabel('DVARS', fontsize=20)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2_.spines['top'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2_.get_xticklabels(), visible=False)
    ax2_.legend(['DVARS'
                ], loc='center left', fontsize=15, bbox_to_anchor=(1.15, 0.3))

    ax3.plot(x,global_signal)
    #ax3.plot(x,CR_data_dict['predicted_time'])
    ax3.set_ylabel('Global signal and CR variance', fontsize=20)
    #ax3_ = ax3.twinx()
    #ax3_.plot(x,CR_data_dict['VE_temporal'], 'darkviolet')
    #ax3_.set_ylabel('CR $\mathregular{R^2}$', fontsize=20)
    #ax3_.spines['right'].set_visible(False)
    #ax3_.spines['top'].set_visible(False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    #plt.setp(ax3_.get_xticklabels(), visible=False)
    ax3.legend(['Global signal', 'CR prediction'
                ], loc='center left', fontsize=15, bbox_to_anchor=(1.15, 0.7))
    #ax3_.legend(['CR $\mathregular{R^2}$'
    #            ], loc='center left', fontsize=15, bbox_to_anchor=(1.15, 0.2))
    #ax3_.set_ylim([0,1])

    analyses = list(analysis_dict.keys())
    FC_legend=[]
    for analysis in analyses:
        y = np.abs(analysis_dict[analysis]['signal_trace']).mean(axis=1)        
        ax4.plot(x,y)
        FC_legend.append(analysis)

    ax4.legend(FC_legend,
                loc='center left', fontsize=15, bbox_to_anchor=(1.15, 0.5))
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_xlabel('Timepoint', fontsize=25)
    ax4.set_ylabel('Abs. Beta \ncoefficients (Avg.)', fontsize=20)
    plt.setp(ax4.get_xticklabels(), fontsize=15)

    plt.setp(ax1.get_yticklabels(), fontsize=15)
    plt.setp(ax1_.get_yticklabels(), fontsize=15)
    plt.setp(ax2.get_yticklabels(), fontsize=15)
    plt.setp(ax2_.get_yticklabels(), fontsize=15)
    plt.setp(ax3.get_yticklabels(), fontsize=15)
    #plt.setp(ax3_.get_yticklabels(), fontsize=15)
    plt.setp(ax4.get_yticklabels(), fontsize=15)


    analyses = list(analysis_dict.keys())
    FC_legend=[]
    FC_map_list=[]
    for analysis in analyses:
        FC_maps = analysis_dict[analysis]['FC_maps']
        for i in range(FC_maps.shape[0]):
            FC_map_list.append(FC_maps[i,:])
            FC_legend.append(f'{analysis} {i}')
    

    nrows = 4+len(FC_map_list)

    fig2, axes2 = plt.subplots(nrows=nrows, ncols=1, figsize=(4, 4*nrows))
    plt.tight_layout()

    scaled = otsu_scaling(template_file)
    
    ax = axes2[0]
    cbar = plot_2d(ax,scaled,fig2,vmin=0,vmax=1.2,cmap='gray', alpha=1, cbar=False, threshold=None)

    temporal_std = spatial_info['temporal_std']
    sitk_img = recover_3D(
        mask_file, temporal_std)

    # select vmax at 95th percentile value
    vector = temporal_std.flatten()
    vector.sort()
    vmax = vector[int(len(vector)*0.95)]
    cbar = plot_2d(ax,sitk_img,fig2,vmin=0,vmax=vmax,cmap='inferno', alpha=1, cbar=True, threshold=None)
    
    cbar.ax.get_yaxis().labelpad = 35
    cbar.set_label('Standard \n Deviation', fontsize=17, rotation=270, color='white')
    cbar.ax.tick_params(labelsize=15)
    ax.set_title('$\mathregular{BOLD_{SD}}$', fontsize=30, color='white')


    ax = axes2[1]
    cbar = plot_2d(ax,scaled,fig2,vmin=0,vmax=1.2,cmap='gray', alpha=1, cbar=False, threshold=None)

    predicted_std = spatial_info['predicted_std']
    sitk_img = recover_3D(
        mask_file, predicted_std)

    # select vmax at 95th percentile value
    vector = predicted_std.flatten()
    vector.sort()
    vmax = vector[int(len(vector)*0.95)]
    cbar = plot_2d(ax,sitk_img,fig2,vmin=0,vmax=vmax,cmap='inferno', alpha=1, cbar=True, threshold=None)
    cbar.ax.get_yaxis().labelpad = 35
    cbar.set_label('Standard \n Deviation', fontsize=17, rotation=270, color='white')
    cbar.ax.tick_params(labelsize=15)
    ax.set_title('$\mathregular{CR_{SD}}$', fontsize=30, color='white')


    ax = axes2[2]
    cbar = plot_2d(ax,scaled,fig2,vmin=0,vmax=1.2,cmap='gray', alpha=1, cbar=False, threshold=None)
    
    sitk_img = recover_3D(
        mask_file, spatial_info['VE_spatial'])
    cbar = plot_2d(ax,sitk_img,fig2,vmin=0,vmax=1,cmap='inferno', alpha=1, cbar=True, threshold=0.1)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.set_label('$\mathregular{R^2}$', fontsize=17, rotation=270, color='white')
    cbar.ax.tick_params(labelsize=15)
    ax.set_title('CR $\mathregular{R^2}$', fontsize=30, color='white')

    ax = axes2[3]
    cbar = plot_2d(ax,scaled,fig2,vmin=0,vmax=1.2,cmap='gray', alpha=1, cbar=False, threshold=None)
    
    sitk_img = recover_3D(
        mask_file, spatial_info['GS_cov'])
    # select vmax at 95th percentile value
    vector = spatial_info['GS_cov'].flatten()
    vector.sort()
    vmax = vector[int(len(vector)*0.95)]
    cbar = plot_2d(ax,sitk_img,fig2,vmin=-vmax,vmax=vmax,cmap='cold_hot', alpha=1, cbar=True, threshold=None)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.set_label("Covariance", fontsize=17, rotation=270, color='white')
    cbar.ax.tick_params(labelsize=15)
    ax.set_title('Global Signal Covariance', fontsize=30, color='white')

    for i,name in zip(range(len(FC_map_list)),FC_legend):
        FC_map = FC_map_list[i]
        ax = axes2[i+4]
        cbar = plot_2d(ax,scaled,fig2,vmin=0,vmax=1.2,cmap='gray', alpha=1, cbar=False, threshold=None)

        sitk_img = recover_3D(
            mask_file, FC_map)
        vector = FC_map.flatten()
        vector.sort()
        vmax = vector[int(len(vector)*0.95)]
        cbar = plot_2d(ax,sitk_img,fig2,vmin=-vmax,vmax=vmax,cmap='cold_hot', alpha=1, cbar=True, threshold=None)
        
        cbar.ax.get_yaxis().labelpad = 35
        cbar.set_label("Connectivity", fontsize=17, rotation=270, color='white')
        cbar.ax.tick_params(labelsize=15)
        ax.set_title(name, fontsize=30, color='white')

    return fig, fig2