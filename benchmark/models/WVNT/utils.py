import numpy as np
import scipy
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing

def extract_seiz_ranges(true_data):
    diff_data = np.diff(np.concatenate([[0],np.squeeze(true_data),[0]]))
    starts = np.where(diff_data == 1)[0]
    stops = np.where(diff_data == -1)[0]
    return list(zip(starts,stops))

def get_gaussianx_threshold(sz_prob,noise_floor='medianover',verbose=False,seed=100):
    all_gbounds = []
    all_chs = []
    all_max_means = []
    for i_ch in range(sz_prob.shape[1]):
        X = sz_prob.iloc[:,i_ch].to_numpy()
        X_f = X.reshape(-1,1)
        # X_f = X_f[X_f < np.percentile(X_f,99.99)].reshape(-1,1)
        bics = []
        for n in range(1,3):
            gmm = GaussianMixture(n_components=n,random_state=seed)
            gmm.fit(X_f)
            bics.append(gmm.bic(X_f))
        if bics[0]<bics[1]:
            all_gbounds.append(np.nan)
            all_chs.append(sz_prob.columns[i_ch])
            all_max_means.append(np.nan)
            if verbose:
                print(f'{sz_prob.columns[i_ch]}: unimodal channel')
            continue

        means = gmm.means_.flatten()
        mu1, mu2 = means
        
        sigma1, sigma2 = np.sqrt(gmm.covariances_.flatten())
        pi1, pi2 = gmm.weights_

        # Coefficients of the quadratic equation
        A = (1 / (2 * sigma1**2)) - (1 / (2 * sigma2**2))
        B = (mu2 / sigma2**2) - (mu1 / sigma1**2)
        C = ((mu1**2 / (2 * sigma1**2)) - (mu2**2 / (2 * sigma2**2))
            - np.log((pi1 * sigma2) / (pi2 * sigma1)))

        # Solve for x (intersection points)
        boundaries = np.roots([A, B, C])
        meets_criteria = boundaries[(boundaries > min(mu1,mu2)) & (boundaries < max(mu1,mu2))]
        if len(meets_criteria) < 1:
            all_gbounds.append(np.nan)
            all_max_means.append(np.nan)
            if verbose:
                print(f'{sz_prob.columns[i_ch]}: overlapping gaussians')
        else:
            boundary = meets_criteria[0]
            all_gbounds.append(boundary)
            all_max_means.append(max(means))
        all_chs.append(sz_prob.columns[i_ch])
    all_gbounds = np.array(all_gbounds)
    if noise_floor == 'meanover':
        threshold = np.nanmean(all_gbounds[all_gbounds > np.nanmean(all_gbounds)])
    elif noise_floor == 'medianover':
        threshold = np.nanmedian(all_gbounds[all_gbounds > np.nanmedian(all_gbounds)])
    else:
        threshold = np.nanmean(all_gbounds[all_gbounds > noise_floor])
    return threshold

def get_pred(sz_prob, threshold, filter_w=5, rwin_size=5, rwin_req=4, num_chan = 1):
    feat_settings = {'win':1, 'stride':0.5}
    filter_w_idx = int(np.floor((filter_w - feat_settings['win']) / feat_settings['stride'])) + 1
    rwin_size_idx = int(np.floor((rwin_size - feat_settings['win']) / feat_settings['stride'])) + 1
    rwin_req_idx = int(np.floor((rwin_req - feat_settings['win']) / feat_settings['stride'])) + 1

    sz_clf = sz_prob > threshold
    sz_clf = scipy.ndimage.median_filter(sz_clf, size=filter_w_idx, mode='nearest',axes=0,origin=0)
    
    rolling_sums = np.round(np.abs(scipy.signal.fftconvolve(sz_clf, np.ones([rwin_size_idx,1]), mode='full',axes=0)))
    rolling_sums[:rwin_size_idx-1,:] = rolling_sums[rwin_size_idx-1,:]
    rolling_sums = rolling_sums[:sz_clf.shape[0],:]
    sz_spread_idxs_all = rolling_sums >= rwin_req_idx
    
    pred = (np.sum(sz_spread_idxs_all, axis=1) >= num_chan).astype(int)

    return pred, sz_spread_idxs_all

def smooth_pred(pred):
    smoothed = binary_opening(pred, structure=np.ones(2))  # remove short 1s
    smoothed = binary_closing(smoothed, structure=np.ones(2))  # remove short 0s
    smoothed[0] = smoothed[1]
    smoothed[-1] = smoothed[-2]

    smoothed = binary_opening(pred, structure=np.ones(3))  # remove short 1s
    smoothed = binary_closing(smoothed, structure=np.ones(3))  # remove short 0s
    smoothed[0] = smoothed[1]
    smoothed[-1] = smoothed[-2]
    return smoothed.astype(int)


def get_events(smoothed_pred, gap_num = 2, min_event_num = 10):
    sz_events = np.array(extract_seiz_ranges(smoothed_pred))

    if sz_events.shape[0] == 0:
        return sz_events

    # Merge events that are close together
    start_times = sz_events[:, 0]
    end_times = sz_events[:, 1]

    merged_start = [start_times[0]]
    merged_end = [end_times[0]]

    for i in range(1, len(start_times)):
        if start_times[i] - merged_end[-1] <= gap_num:
            # Merge the events
            merged_end[-1] = end_times[i]
        else:
            # Start a new event
            merged_start.append(start_times[i])
            merged_end.append(end_times[i])

    sz_events = np.array(list(zip(merged_start, merged_end)))

    # Filter short events
    durations = sz_events[:, 1] - sz_events[:, 0]
    sz_events = sz_events[durations >= min_event_num]
    return sz_events

def get_event_smoothed(smoothed_pred, gap_num = 2, min_event_num = 10):
    new_pred = np.zeros_like(smoothed_pred)
    sz_events = np.array(extract_seiz_ranges(smoothed_pred))

    if sz_events.shape[0] == 0:
        return new_pred

    # Merge events that are close together
    start_times = sz_events[:, 0]
    end_times = sz_events[:, 1]

    merged_start = [start_times[0]]
    merged_end = [end_times[0]]

    for i in range(1, len(start_times)):
        if start_times[i] - merged_end[-1] <= gap_num:
            # Merge the events
            merged_end[-1] = end_times[i]
        else:
            # Start a new event
            merged_start.append(start_times[i])
            merged_end.append(end_times[i])

    sz_events = np.array(list(zip(merged_start, merged_end)))

    # Filter short events
    durations = sz_events[:, 1] - sz_events[:, 0]
    sz_events = sz_events[durations >= min_event_num]
    for start, end in sz_events:
        new_pred[start:end] = 1
    return new_pred

# def smooth_pred(pred, gap_num = 5, min_event_num = 20):
#     smoothed_pred = __smooth_pred__(pred)
#     new_pred = np.zeros_like(smoothed_pred)
#     sz_events = np.array(extract_seiz_ranges(smoothed_pred))

#     if sz_events.shape[0] == 0:
#         return sz_events

#     # Merge events that are close together
#     start_times = sz_events[:, 0]
#     end_times = sz_events[:, 1]

#     merged_start = [start_times[0]]
#     merged_end = [end_times[0]]

#     for i in range(1, len(start_times)):
#         if start_times[i] - merged_end[-1] <= gap_num:
#             # Merge the events
#             merged_end[-1] = end_times[i]
#         else:
#             # Start a new event
#             merged_start.append(start_times[i])
#             merged_end.append(end_times[i])

#     sz_events = np.array(list(zip(merged_start, merged_end)))

#     # Filter short events
#     durations = sz_events[:, 1] - sz_events[:, 0]
#     sz_events = sz_events[durations >= min_event_num]
#     for start, end in sz_events:
#         new_pred[start:end] = 1

#     return new_pred

# def set_plot_params():
#     plt.rcParams['image.cmap'] = 'magma'
#     plt.rcParams['xtick.labelsize'] = 14
#     plt.rcParams['ytick.labelsize'] = 14
#     plt.rcParams['axes.linewidth'] = 2
#     plt.rcParams['axes.titlesize'] = 16
#     plt.rcParams['axes.labelsize'] = 14
#     plt.rcParams['lines.linewidth'] = 2

#     plt.rcParams['xtick.major.size'] = 5  # Change to your desired major tick size
#     plt.rcParams['ytick.major.size'] = 5  # Change to your desired major tick size
#     plt.rcParams['xtick.minor.size'] = 3   # Change to your desired minor tick size
#     plt.rcParams['ytick.minor.size'] = 3   # Change to your desired minor tick size

#     plt.rcParams['xtick.major.width'] = 2  # Change to your desired major tick width
#     plt.rcParams['ytick.major.width'] = 2  # Change to your desired major tick width
#     plt.rcParams['xtick.minor.width'] = 1  # Change to your desired minor tick width
#     plt.rcParams['ytick.minor.width'] = 1  # Change to your desired minor tick width
