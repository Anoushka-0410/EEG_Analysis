# %%
# Importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import autoreject

# %%
# Loading raw data from subjects in folders
def get_sub_folders(fol_path):
    folders=[]
    folders_needed=[]
    for root, dirs, _ in os.walk(fol_path):
        folders.extend([os.path.join(root, d) for d in dirs])
        break
    for i in range(0,1):
        folders_needed.append(folders[i])
    return folders_needed

# %%
# Function to preprocess data
def preprocess(raw_data_path):
    rd= mne.io.read_raw_eeglab(raw_data_path, preload=True,verbose=False, montage_units='mm') #load data
    raw_plot=rd.plot()
    channels_to_drop = ['FT9','FT10','TP9','TP10','Resp'] #channels to drop
    available_channels = rd.ch_names
    channels_to_drop_existing = [ch for ch in channels_to_drop if ch in available_channels]
    rd.drop_channels(channels_to_drop_existing) #drop channels
    
    rd.set_eeg_reference(ref_channels='average') #rereference to average
    new_sampling_freq = 256 #new sampling frequency
    rd.resample(new_sampling_freq)  #resample

    rd= rd.copy().filter(l_freq=0.1, h_freq=None) #highpass filter

    rd.info['bads'] = ['Cz'] #exlude Cz channel from ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=50, max_iter=800) #perform ICA
    ica.fit(rd)  #fit ICA
    after_ica = rd.plot()
    events,event_dict = mne.events_from_annotations(rd)  #{'S  1': 1, 'S  2': 2, 'S  3': 3}
    epochs= mne.Epochs(rd, events, tmin=-1, tmax=2.5, event_id=event_dict, preload=True) #epoching
    
    del rd #delete raw data as only epochs needed further.
    
   
    ar= autoreject.AutoReject(n_interpolate=[1,2,3,4],random_state=11,n_jobs=1,verbose=True) #perform autoreject to remove bad epochs
    ar.fit(epochs[:10])
    epochs_arr, reject_log = ar.transform(epochs, return_log=True)
    epochs_arr.info['bads'].remove('Cz') #remove Cz channel from bad channels
    epochs_arr.interpolate_bads() #interpolate bad channels exlcuding Cz
    


    return epochs_arr




# %%
# Pipline for preprocessing

def pipe_pre(in_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    subjects = get_sub_folders(in_path)

    for subject in subjects:
        print("Preprocessing subject:", subject)

        # Construct paths for EEG and output folders
        eeg_folder = os.path.join(subject, 'eeg')  # Assuming eeg folder is located within each subject folder
        out_folder = os.path.join(out_path, subject[-7:])

        # Create the output folder if it doesn't exist
        os.makedirs(out_folder, exist_ok=True)

        # Process .set files in the eeg_folder
        set_files = [f for f in os.listdir(eeg_folder) if f.endswith('.set')]
        for set_file in set_files:
            raw_data_path = os.path.join(eeg_folder, set_file)
            epochs_arr = preprocess(raw_data_path)
            out_file = os.path.join(out_folder, "OUT_" + set_file.replace('.set', '-epo.fif'))
            epochs_arr.save(out_file, overwrite=True)

        # Construct electrode positions file name
            electrode_file_name = f"{subject[-7:]}_task-Oddball_electrodes.tsv"
            montage_file = os.path.join(eeg_folder, electrode_file_name)
            if os.path.exists(montage_file):
                montage_data = pd.read_csv(montage_file, sep='\s+|\t+', header=0,engine='python')
                # Set the electrode names as the index
                montage_data = montage_data.set_index('name',drop=True)
                # Rescaling the electrode positions to fit the head model
                scaling_fac= 0.095
                montage_data.loc[:,['x','y','z']]*=scaling_fac
                # Convert the dataframe to a dictionary
                map=montage_data.T.to_dict('list')
                # Create montage object
                montage= mne.channels.make_dig_montage(ch_pos=map,coord_frame='head')
                rd = mne.io.read_raw_eeglab(raw_data_path, preload=True, verbose=False)
                # Set the montage
                rd.set_montage(montage,on_missing='ignore')
                rd.plot_sensors(kind='topomap')
            else:
                print(f"Electrode positions file not found for subject: {subject}")

        print("Preprocessing done for subject:", subject, "\n")




# %%





