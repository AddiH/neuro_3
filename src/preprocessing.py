# this file preprocesses the data, and outputs X and y for the machine learning
import mne
import numpy as np
from os.path import join
from tqdm import tqdm
import sys
import io
import warnings
import pickle

# Redirect standard output to a null device (suppress output messages)
#original_stdout = sys.stdout
#sys.stdout = io.StringIO()

# Filter and suppress the specific MNE runtime warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning, message="This filename .* does not conform to MNE naming conventions")

##############################################################################################################
##############################################################################################################
############################################### Functions ####################################################
##############################################################################################################
##############################################################################################################

# This function returns the data (X) and labels (y) for sensor space
def preprocess_sensor_space_data(subject, date, raw_path,
                                h_freq=40,
                                tmin=-0.200, tmax=2.000, baseline=(None, 0),
                                reject=None, decim=7):
    epochs_list = list()
    # looping over all the files
    for recording_index, recording_name in enumerate(recording_names):
        # removes the first 4 characters of the name
        fif_fname = recording_name[4:]
        # joins the path to the file
        full_path = join(raw_path, subject, date, 'MEG', recording_name,
                        'files', fif_fname + '.fif')
        # load in the data
        raw = mne.io.read_raw(full_path, preload=True)
        # filter the data - cut all frequencies above h_freq
        raw.filter(l_freq=None, h_freq=h_freq, n_jobs=3)
        # find the triggers
        events = mne.find_events(raw, min_duration=0.002)
        # create a dictionary of event codes
        if 'self' in recording_name:
            event_id = dict(self_positive=11, self_negative=12,
                            button_press=23)
        elif 'other' in recording_name: 
            event_id = dict(other_positive=21, other_negative=22,
                            button_press=23)
        else:
            raise NameError('Event codes are not coded for file')
        # create the epochs
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline,
                            preload=True, decim=decim)
        # remove eeg (keep only magnetometers and gradiometers)
        epochs.pick_types(meg=True)
        # add the epochs (remeber we are in a loop)
        epochs_list.append(epochs)
        # split the epocs into data (X) and labels (y)
        if recording_index == 0:
            X = epochs.get_data()
            y = epochs.events[:, 2]
        else:
            X = np.concatenate((X, epochs.get_data()), axis=0)
            y = np.concatenate((y, epochs.events[:, 2]))
    
    return epochs_list, X, y

# This function returns the data (X) and labels (y) for source space
def preprocess_source_space_data(subject, 
                                subjects_dir,
                                epochs_list,
                                method='MNE', lambda2=1, pick_ori='normal',
                                label=None):
    # initialize the y (empty array)
    y = np.zeros(0)
    # loop over the epochs
    for epochs in epochs_list: # get y
        # notice that this is the same as the end of the function above
        y = np.concatenate((y, epochs.events[:, 2]))
    
    if label is not None:
        # if a label is added (labeling the brain is a science in itself, freesurfer has automatically labelled the brainregions of our participants MR data)
        # goes to MEG_workshop/data/freesurfer/0108/label
        label_path = join(subjects_dir, subject, 'label', label)
        # mne loads the label (so hopefully freesurfer identified the correct brainregion, and is using the same coordinate system as our MEG data)
        # might be good to sanity check this
        label = mne.read_label(label_path)
        
    # list of folder names    
    recording_names = ['001.self_block1',  '002.other_block1',
                        '003.self_block2',  '004.other_block2',
                        '005.self_block3',  '006.other_block3']
    
    # loop over the recordings (every epoch)
    for epochs_index, epochs in enumerate(epochs_list): ## get X
        fwd_fname = recording_names[epochs_index][4:] + '-oct-6-src-' + \
                    '5120-fwd.fif'
        # read the forward solution (BEM)
        fwd = mne.read_forward_solution(join(subjects_dir,
                                            subject, 'bem', fwd_fname))
        # find covariance between sensors
        noise_cov = mne.compute_covariance(epochs, tmax=0.000)
        inv = mne.minimum_norm.make_inverse_operator(epochs.info,
                                                    fwd, noise_cov)
        # make forward solution
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2,
                                                    method, label,
                                                    pick_ori=pick_ori)
        # loop over the stcs (every epoch) to get the data
        for stc_index, stc in enumerate(stcs):
            this_data = stc.data
            if epochs_index == 0 and stc_index == 0:
                n_trials = len(stcs)
                n_vertices, n_samples = this_data.shape
                this_X = np.zeros(shape=(n_trials, n_vertices, n_samples))
            this_X[stc_index, :, :] = this_data
            
        if epochs_index == 0:
            X = this_X
        else:
            X = np.concatenate((X, this_X))
    return X, y

##############################################################################################################
##############################################################################################################
############################################### Variables ####################################################
##############################################################################################################
##############################################################################################################

# names of the files
recording_names = [
                '001.self_block1',  '002.other_block1',
                '003.self_block2',  '004.other_block2',
                '005.self_block3',  '006.other_block3']

subjects = [
            #'0108', 
            #'0109', 
            #'0110',
            #'0111', 
            #'0112', 
            #'0113',
            #'0114', 
            '0115'
            ]

dates = [
        #'20230928_000000',
        #'20230926_000000', 
        #'20230926_000000',
        #'20230926_000000', 
        #'20230927_000000', 
        #'20230927_000000',
        #'20230927_000000', 
        '20230928_000000'
        ]

subjects_dir = '../data/freesurfer'
raw_path = '../data/MEG_data/'

##############################################################################################################
##############################################################################################################
########################################## Running functions #################################################
##############################################################################################################
##############################################################################################################

# looping over all the subjects
for subject, date in tqdm(zip(subjects, dates), total=len(subjects)):
    # preprocessing the sensor data 
    epochs_list, X_sensor, y_sensor = preprocess_sensor_space_data(
                                                subject = subject, 
                                                date = date, 
                                                raw_path = raw_path,
                                                h_freq = 40, # remove high frequencies
                                                tmin = -0.200, 
                                                tmax = 2.000, # for getting more inner speech
                                                baseline = (None, 0),
                                                reject = None, 
                                                decim = 7) # for less data (easier to compute)
    
    # save the data
    np.save('../data/sensor_data/X_' + subject, X_sensor)
    np.save('../data/sensor_data/y_' + subject, y_sensor)
    times = epochs_list[0].times
    np.save('../data/sensor_data/times_' + subject, times)
    # Save the epochs_list using pickle
    with open('../data/sensor_data/epochs_list_' + subject + '.pkl', 'wb') as file:
        pickle.dump(epochs_list, file)

    # preprocessing the source data
    X_source, y_source = preprocess_source_space_data(
                                                subject = subject, 
                                                subjects_dir = subjects_dir,
                                                epochs_list = epochs_list)
    # save the data
    np.save('../data/source_data/X_' + subject, X_source)
    np.save('../data/source_data/y_' + subject, y_source)

# Restore the original standard output
#sys.stdout = original_stdout