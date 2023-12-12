# this returns output of the calssification and saves a plot
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
from tqdm import tqdm

##############################################################################################################
##############################################################################################################
############################################### Functions ####################################################
##############################################################################################################
##############################################################################################################

# finds which index in the trigger array corresponds to the trigger of interest. Returns the indices
def get_indices(y, triggers):
    indices = list()
    for trigger_index, trigger in enumerate(y):
        if trigger in triggers:
            indices.append(trigger_index)
            
    return indices

# returns the data (X) and labels (y), with equal number of trials for each condition
def equalize_number_of_indices(X,y): 
    keys, counts = np.unique(y, return_counts = True) # get unique conditions and the number of trials for each condition
    
    min_trials = counts.min() # get the minimum number of trials for a condition
    keep_index = []

    for key in keys: # loop over evry condition
        index = np.where(np.array(y) == key) # get the index of the trials for the current condition
        random_trials = np.random.choice(index[0], size = min_trials, replace=False) # randomly select trials to keep
        keep_index.extend(random_trials) # add the randomly selected trials to the list of trials to keep
    
    X_equal = X[keep_index, :, :] # slice it up
    y_equal = y[keep_index]

    return X_equal, y_equal

# classifyer
def simple_classification(X, y, triggers):

    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import permutation_test_score, StratifiedKFold
    from sklearn.inspection import permutation_importance
    from tqdm import tqdm

    n_features = X.shape[1] 
    n_samples = X.shape[2] 

    indices = get_indices(y, triggers)
    X = X[indices, :, :]
    y = y[indices] 

    X, y = equalize_number_of_indices(X, y)  # Equalize the number of trials for each condition in X and y

    gnb = GaussianNB()  # Create a Gaussian Naive Bayes classifier
    sc = StandardScaler()  # Create a feature scaler
    cv = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)  # Create a 5-fold cross-validation object

    # empty arrrays to hold the results:
    mean_scores = np.zeros(n_samples)  
    feature_importance = np.zeros((n_features, n_samples)) 
    permutation_scores = np.zeros((n_samples, 100))  
    pvalues = np.zeros(n_samples)  

    for sample_index in tqdm(range(n_samples)):  # Loop through each sample
        this_X = X[:, :, sample_index]  # Extract the feature data for the current sample
        sc.fit(this_X)  # Fit the feature scaler to the current sample
        this_X_std = sc.transform(this_X)  # Standardize the features for the current sample

        scores, permutation_score, pvalue = permutation_test_score(gnb, this_X_std, y, cv=cv)  # Perform permutation testing and get scores, permutation scores, and p-values
        gnb.fit(this_X_std, y)  # Fit the Gaussian Naive Bayes classifier to the standardized features and labels
        importances = permutation_importance(gnb, this_X_std, y)  # Compute feature importances using permutation importance

        feature_importance[:, sample_index] = importances.importances_mean  # Store the feature importances for the current sample
        mean_scores[sample_index] = np.mean(scores)  # Calculate and store the mean score for the current sample
        permutation_scores[sample_index, :] = permutation_score  # Store the permutation scores for the current sample
        pvalues[sample_index] = pvalue  # Store the p-value for the current sample

    return mean_scores, permutation_scores, pvalues, feature_importance 


def plot_classification(times, mean_scores, title=None):

    plt.figure()
    plt.plot(times, mean_scores)
    plt.hlines(0.50, times[0], times[-1], linestyle='dashed', color='k')
    plt.ylabel('Proportion classified correctly')
    plt.xlabel('Time (s)')
    if title is None:
        pass
    else:
        plt.title(title)
    plt.show()


##############################################################################################################
##############################################################################################################
############################################### Variables ####################################################
##############################################################################################################
##############################################################################################################

# load data
data_folder = '../data/'

#s_data = data_folder + 'sensor_data/'
#s_class = data_folder + 'sensor_class/'

s_data = data_folder + 'source_data/'
s_class = data_folder + 'source_class/'

subjects = [
            '0108', 
            '0109', '0110',
            '0111', '0112', '0113',
            '0114', '0115'
            ]

##############################################################################################################
##############################################################################################################
################################################## RUN #######################################################
##############################################################################################################
##############################################################################################################

for subject in tqdm(subjects):
    # load the data
    X = np.load(s_data + 'X_' + subject + '.npy')
    y = np.load(s_data + 'y_' + subject + '.npy')

    print(X.shape)

    # collapse the conditions
    y_copy=copy.copy(y)
    y_copy[y_copy==11]=16
    y_copy[y_copy==12]=17
    y_copy[y_copy==21]=16
    y_copy[y_copy==22]=17

    # run classification
    sensor_pos_neg_scores, pos_permutation_scores, pos_pvalues, pos_feat_importance = simple_classification(X, y_copy, triggers=[16, 17])

    # Data to be saved
    data_to_save = [sensor_pos_neg_scores, pos_permutation_scores, pos_pvalues, pos_feat_importance]

    # Save the data 
    with open(s_class + subject + '.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)