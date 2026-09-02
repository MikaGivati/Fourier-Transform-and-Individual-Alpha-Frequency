import os
import time
import re

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view 
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.fft import fft

from edf import edfread
from args_parser import parse_args

## ====================================================================
## ============================ Parameters ============================
## ====================================================================

fs = 256       # units in [Hz]

args = parse_args()
window_size = args.window_size * fs # convert from seconds to samples
overlap     = args.overlap     * fs # convert from seconds to samples

EC_COLOR = 'deepskyblue'
EO_COLOR = 'deeppink'

ALPHA_BAND = [8, 12]
INTERESTED_RANGE = [6, 14]

## ====================================================================
## ======================= Q1: Data Handling ==========================
## ====================================================================

DATA_DIR = args.DATA_DIR if args.DATA_DIR != '' else 'data'

#match .edf filename contains a number and either EO or EC and capture them as group 
file_pattern = re.compile(r".*?(\d+).*?(EO|EC).*?\.edf$")

#dict for all organized data, structure {id:{EO:data, EC:data}}
data_dict = {}
#loop over the main folder
for subfolder in os.listdir(DATA_DIR):
    subfolder_path = os.path.join(DATA_DIR, subfolder)

    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)

            if os.path.isfile(file_path):
                match = file_pattern.search(filename)

                if match:
                    sub_num = int(match.group(1))
                    task_type = match.group(2)

                    if sub_num not in data_dict:
                        data_dict[sub_num] = {}

                    hdr, record = edfread(file_path)
                    #extract the Pz electrode - channel 19
                    signal = record[args.elec_num, :]
                    data_dict[sub_num][task_type] = signal
                    
                
## ====================================================================
## ======================= Q2: Power Spectra ==========================
## ====================================================================

## Plot raw signal
def plot_raw_signal(data_dict, fs, eo_color, ec_color):
    num_samples_10s = 10*fs
    time_axis = np.arange(num_samples_10s)/fs
    subjects = sorted(list(data_dict.keys()))
    fig, axes = plt.subplots(3,1,figsize=(15,10), sharex=True)
    fig.suptitle("Raw EEG signals - first 10 sec", fontsize=16)

    for idx, sub_num in enumerate(subjects):
        ax = axes[idx]

        if 'EO' in data_dict[sub_num]:
            ax.plot(time_axis, data_dict[sub_num]['EO'][:num_samples_10s], color=eo_color, label = 'Eyes Open - EO', alpha=0.8)

        if 'EC' in data_dict[sub_num]:
            ax.plot(time_axis, data_dict[sub_num]['EC'][:num_samples_10s], color=ec_color, label = 'Eyes Closed - EC', alpha=0.8)

        ax.set_title(f"Subject {sub_num}", fontsize=15)
        ax.set_ylabel("Amplitude", fontsize=15)
        ax.grid(True)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel("Time [sec]", fontsize=15)
    plt.tight_layout()
    plt.show()

plot_raw_signal(data_dict, fs, EO_COLOR, EC_COLOR)

## Define the functions

#compute PSD of a signal using fft
#input - 1d signal and sampling frequency, output - frequency axis and psd corresponding to each frequency 
def fft_psd(x, fs):
    n = len(x)
    #compute the fft for all samples
    fft_vals = fft(x)
    #convert fft to power spectrum ans normalize
    psd_two_sided = (np.abs(fft_vals)**2) / (n*fs)
    #keep only positive frequencies
    half_n = n//2 +1
    #extract one side spectrum
    p=psd_two_sided[:half_n].copy()
    p[1:-1] *= 2
    f = np.linspace(0, fs/2, half_n)

    return f, p

#calc PSD using built in scipy welch method
def welch_psd(x, fs, window_size, overlap):
    f, p = welch(x, fs=fs, nperseg=window_size, noverlap=overlap)
    return f, p

#custom made welch method, optimized vectorization with FFT without loops
#compute 95% confident intervals
def welch_fft_psd(x, fs, window_size, overlap):
    step_size = window_size - overlap
    windows = sliding_window_view(x, window_shape=window_size)[::step_size]

    hanning_window = np.hanning(window_size)
    windows_windowed = windows * hanning_window

    fft_vals = fft(windows_windowed, axis=1)
    psd_two_sided = (np.abs(fft_vals)**2 / (window_size * fs))
    half_n = window_size//2 + 1
    p_segment = psd_two_sided[:,:half_n].copy()
    p_segment[:, 1:-1] *= 2

    f=np.fft.rfftfreq(window_size, d=1/fs)
    n_windows = p_segment.shape[0]
    p_mean = np.mean(p_segment, axis=0)
    p_std = np.std(p_segment, axis=0)
    p_ci = 1.96 * (p_std / np.sqrt(n_windows))
    return f, p_mean, p_ci

#custom made welch method, manually constructed discrete fourier transform.
#vandermonde mat (W) instead of FFT routines.
def welch_dft_psd(x, fs, window_size, overlap)  :

    step_size = window_size - overlap
    windows = sliding_window_view(x, window_shape=window_size)[::step_size]
    hanning_window = np.hanning(window_size)
    windows_windowed = windows * hanning_window
    k = np.arange(window_size).reshape(-1,1)
    n_indices = np.arange(window_size).reshape(1,-1)
    kn_mat = k @ n_indices
    W = np.exp(-2j*np.pi*kn_mat/window_size)
    fft_vals = windows_windowed @ W.T
    psd_two_sided = (np.abs(fft_vals)**2) / (window_size*fs)
    half_n = window_size//2+1
    p_segment = psd_two_sided[:,:half_n].copy()
    p_segment[:,1:-1] *=2
    f=np.fft.rfftfreq(window_size, d=1/fs)
    n_windows = p_segment.shape[0]
    p_mean = np.mean(p_segment, axis=0)
    p_std = np.std(p_segment, axis = 0)
    p_ci = 1.96 * (p_std / np.sqrt(n_windows))
    return f, p_mean, p_ci

# BONUS
#calc full signal DFT using vandermonde approach
def dft_psd(x, fs):
    n = len(x)
    half_n = n//2+1
    k_indices = np.arange(half_n)
    n_indices = np.arange(n)
    fft_vals_half = np.zeros(half_n, dtype=complex)
    chunk_size = 500
    for i in range(0, half_n, chunk_size):
        k_chunk = k_indices[i:i+chunk_size]
        kn_mat_chunk = np.outer(k_chunk, n_indices)
        W_chunk = np.exp(-2j * np.pi * kn_mat_chunk / n)
        fft_vals_half[i:i+chunk_size] = W_chunk @ x
    p=(np.abs(fft_vals_half)**2)/(n*fs)
    p[1:-1] *= 2
    f=np.linspace(0,fs/2, half_n)
    return f, p

## Calculate the power spectra for each (subject, task)
spectra_results = {}

#dict that store execution time for each method
runtime_results = {
    'Fast Fourier Transform (FFT)': [],
    'Welch method': [],
    'Custom-made Welch (FFT based)': [],
    'Custom-made Welch (DFT based)': [],
    'bonus - Discrete Fourier Transform (DFT)': []
    }

subjects = sorted(list(data_dict.keys()))
for sub_num in subjects:
    spectra_results[sub_num] = {method:{'EO':None, 'EC':None} for method in runtime_results.keys()}
    for task in ['EO', 'EC']:
        if task in data_dict[sub_num]:
            sig = data_dict[sub_num][task]

            t0=time.time()
            spectra_results[sub_num]['Fast Fourier Transform (FFT)'][task] = fft_psd(sig, fs)
            runtime_results['Fast Fourier Transform (FFT)'].append(time.time() - t0)

            t0=time.time()
            spectra_results[sub_num]['Welch method'][task] = welch_psd(sig, fs, window_size, overlap)
            runtime_results['Welch method'].append(time.time() - t0)
            
            t0 = time.time()
            spectra_results[sub_num]['Custom-made Welch (FFT based)'][task] = welch_fft_psd(sig, fs, window_size, overlap)
            runtime_results['Custom-made Welch (FFT based)'].append(time.time() - t0)
            
            t0=time.time()
            spectra_results[sub_num]['Custom-made Welch (DFT based)'][task] = welch_dft_psd(sig, fs, window_size, overlap)
            runtime_results['Custom-made Welch (DFT based)'].append(time.time() - t0)

            t0=time.time()            
            spectra_results[sub_num]['bonus - Discrete Fourier Transform (DFT)'][task] = dft_psd(sig, fs)
            runtime_results['bonus - Discrete Fourier Transform (DFT)'].append(time.time() - t0)

## Plot power spectra
def plot_subject_spectra(sub_num, methods_data, interested_range=INTERESTED_RANGE):
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    fig.suptitle(f"Subject {sub_num} - Power Spectra Compresion", fontsize=15, fontweight='bold')
    method_names = list(methods_data.keys())
    for idx , method_name in enumerate(method_names):
        if idx ==0: plot_idx=0
        elif idx ==1: plot_idx=1
        elif idx==2: plot_idx=2
        elif idx==3: plot_idx=5
        elif idx==4: plot_idx=3
        
        ax=axes_flat[plot_idx]
        
        tasks = methods_data[method_name]

        if tasks['EO'] is not None:
            if len(tasks['EO']) == 3:
                f, p_mean, p_ci = tasks['EO']
                ax.plot(f, p_mean, color=EO_COLOR, label='Eyes Open (EO)')
                ax.fill_between(f, p_mean-p_ci, p_mean+p_ci, color=EO_COLOR, alpha=0.2)
            else:
                f, p = tasks['EO']
                ax.plot(f, p, color=EO_COLOR, label='Eyes Open (EO)')

        if tasks['EC'] is not None:
            if len(tasks['EC']) == 3:
                f, p_mean, p_ci = tasks['EC']
                ax.plot(f, p_mean, color=EC_COLOR, label='Eyes Closed (EC)')
                ax.fill_between(f, p_mean-p_ci, p_mean+p_ci, color=EC_COLOR, alpha=0.2)
            else:
                f, p = tasks['EC']
                ax.plot(f, p, color=EC_COLOR, label='Eyes Closed (EC)')

        ax.set_title(method_name, fontsize=15, loc='left')
        ax.set_ylabel("PSD", fontsize=15)
        ax.set_xlim(interested_range)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set_xlabel("Frequency [Hz]", fontsize=15)

    fig.delaxes(axes_flat[4])
    plt.tight_layout()
    plt.show()

for sub_num in subjects:
    plot_subject_spectra(sub_num, spectra_results[sub_num], INTERESTED_RANGE)

## ====================================================================
## ====================== Q3: Finding the IAF =========================
## ====================================================================

#calc IAF by locating the frequency with max power in alpha band during EC
#compute power diff (EC - EO) at peak frequency
def calculate_and_plot_IAF(spectra_results, intrested_range=[6,14],alpha_band=[8,12]):
    subject_list = sorted(list(spectra_results.keys()))
    IAF_results = {}
    for subject_number in subject_list:
        methods_data = spectra_results[subject_number]
        method_names_list = list(methods_data.keys())

        fig, axes_matrix = plt.subplots(2,3,figsize=(15,10))
        axes_flat_list = axes_matrix.flatten()
        fig.suptitle(f"Subject {subject_number} - IAF Calculation per Method", fontsize=15)
        IAF_results[subject_number] = {}

        for method_index, current_method_name in enumerate(method_names_list):
            if 'Fast Fourier Transform' in current_method_name:
                plot_idx = 0
            elif 'Welch method' in current_method_name:
                plot_idx = 1
            elif 'Welch (FFT based)' in current_method_name:
                plot_idx = 2
            elif 'bonus - Discrete Fourier Transform' in current_method_name:
                plot_idx = 3
            elif 'Welch (DFT based)' in current_method_name:
                plot_idx = 5
            else:
                plot_idx = method_index
            
            current_axis = axes_flat_list[plot_idx]
            EC_data = methods_data[current_method_name]['EC']
            EO_data = methods_data[current_method_name]['EO']

            if len(EC_data) == 3:
                frequencies, power_EC, _ = EC_data
                _, power_EO, _ = EO_data
            else:
                frequencies, power_EC = EC_data
                _, power_EO = EO_data

            power_difference = power_EC - power_EO

            alpha_band_mask = (frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1])
            frequencies_in_alpha = frequencies[alpha_band_mask]
            power_diff_in_alpha = power_difference[alpha_band_mask]

            max_power_index = np.argmax(power_diff_in_alpha)
            IAF_value = frequencies_in_alpha[max_power_index]

            IAF_results[subject_number][current_method_name] = IAF_value

            intrested_range_mask = (frequencies >= intrested_range[0]) & (frequencies <= intrested_range[1])

            current_axis.plot(frequencies[intrested_range_mask], power_difference[intrested_range_mask], label='EC - EO Difference')
            current_axis.axvline(x=IAF_value, linestyle='--', alpha=0.7, label=f'IAF: {IAF_value:.2F} [Hz]')
            max_graph_y_val = np.max(power_difference[intrested_range_mask])
            current_axis.text(IAF_value + 0.1, max_graph_y_val*0.9, f'{IAF_value:.3f}', fontsize=15)

            current_axis.set_title(current_method_name, fontsize=15)
            current_axis.set_ylabel("Power Difference [AU]", fontsize=15)
            current_axis.set_xlabel("Frequency [Hz]", fontsize=15)
            current_axis.set_xlim(intrested_range)
            current_axis.grid(True, linestyle='--', alpha=0.5)
            current_axis.legend(loc='upper right')

        fig.delaxes(axes_flat_list[4])
        plt.tight_layout()
        plt.show()
    return IAF_results
    
IAF_results = calculate_and_plot_IAF(spectra_results, INTERESTED_RANGE, ALPHA_BAND)
## ====================================================================
## =================== Q4: Comparison of Methods ======================
## ====================================================================

#print table to sum individual alpha across all subjects and computation methods
print("\n" + "="*10)
print("Q4a IAF results table")
print("="*10)
methods_list = list(runtime_results.keys())
header_str = f"{'Computation Method':<40} | " + " | ".join([f"Sub {sub:<4}" for sub in subjects])
print(header_str)
print("-" * len(header_str))

for method in methods_list:
    row_val = []
    for sub in subjects:
        IAF_val = IAF_results[sub][method]
        row_val.append(f"{IAF_val:<8.3F} Hz")

    print(f"{method:<40} | " + " | ".join(row_val))
print("="*10 + "\n")

## PART B

#analyzed time complexity for each method
method_names =  list(runtime_results.keys())
means = []
sems = []

for method in method_names:
    times = np.array(runtime_results[method])

    mean_time = np.mean(times)
    means.append(mean_time)

    sem_time = np.std(times, ddof=1) / np.sqrt(len(times))
    sems.append(sem_time)

sorted_idx = np.argsort(means)
methods_sorted = [method_names[i] for i in sorted_idx]
means_sorted = [means[i] for i in sorted_idx]
sems_sorted = [sems[i] for i in sorted_idx]


# Create the bar plot
fig, ax = plt.subplots(figsize = (15, 10))
bars = ax.barh(methods_sorted, means_sorted, xerr=sems_sorted)
ax.set_xscale('log')
ax.bar_label(bars, fmt=' %.3f sec',padding=5, fontsize=15)
ax.set_title("Run-Time Performance Comparison Across Methods (Log Scale)", fontsize=15)
ax.set_xlabel("Mean Execution Time [Sec]", fontsize=15)
ax.set_ylabel("Computation Method", fontsize=15)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()