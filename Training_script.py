# %%
# Importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import seaborn as sns
data_path = r'C:\Users\hp\Desktop\EEG_preprocessed'
    

# %%
temp = {'mean_amp':[],'variance_amp':[], 'avg_psd':[],'Age':[], 'subject':[], 'Label':[]}
data = pd.DataFrame(temp) # Create an empty dataframe with the columns we want

# %%
# Read the participants.tsv file
participants = pd.read_csv(r'C:\Users\hp\Desktop\EEG\participants.tsv', sep= '\s+|\t+', engine='python')
data['subject'] = participants['participant_id'].copy()
data.set_index('subject', inplace=True, drop=True)


# %%
# Copy values from the 'AGE' and 'GROUP' columns to the 'Age' and 'Label' columns
data['Age']=participants['AGE'].values
data['MOCA']=participants['MOCA'].values
data['Label'] = participants['GROUP'].values

# %%
X = []
y = []

# Loop through subjects (sub-001 to sub-146)
for subject_index in range(1, 147):
    subject_folder = os.path.join(data_path, f'sub-{subject_index:03d}')
    fif_file_path = os.path.join(subject_folder, f'OUT_sub-{subject_index:03d}_task-Oddball_eeg-epo.fif')
    
    if os.path.exists(fif_file_path):
        epochs = mne.read_epochs(fif_file_path, preload=True)

        # Calculate Power Spectral Density (PSD)
        psd,freqs=mne.time_frequency.psd_array_multitaper(epochs.get_data(), fmin=1, fmax=10, n_jobs=1, verbose=None, sfreq=256)
        avg_psd = np.mean(np.mean(psd,axis=1), axis=1)  # Average across channels
        
        # Calculate mean and variance of the EEG data
        mean_amplitude = np.mean(epochs.get_data(), axis=2)
        variance_amplitude = np.var(epochs.get_data(), axis=2)

        # Update the 'mean_amp', 'mean_power', and 'variance_amp' columns
        data.loc[f'sub-{subject_index:03d}', 'mean_amp'] = mean_amplitude.mean()  # Update with appropriate index
        data.loc[f'sub-{subject_index:03d}', 'avg_psd'] = avg_psd.mean()  # Update with appropriate index
        data.loc[f'sub-{subject_index:03d}', 'variance_amp'] = variance_amplitude.mean()  # Update with appropriate index
print(data.head())

# %%
# Categorize the labels into 'Control', 'PD', 'PDMCI', and 'PDD'
def categorize_label(row):
    if row['Label'] == 'PD':
        if row['MOCA'] < 22:
            return 'PDD'
        elif row['MOCA'] >= 22 and row['MOCA'] <= 26:
            return 'PDMCI'
        else:
            return 'PD'
    else:
        return row['Label']

data['Label'] = data.apply(categorize_label, axis=1)

# %%
print(data[data['Label'] == 'Control'].shape[0]) # Checking the number of control subjects

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Define the feature columns and target variable
feature_cols = ['mean_amp', 'avg_psd','Age','variance_amp']
target_col = 'Label'

# Extract the features and target variable
X = data[feature_cols].values
y = data[target_col].values
nan_indices = np.isnan(X).any(axis=1)

# Filter out rows with NaN values
X_cleaned = X[~nan_indices]
y_cleaned = y[~nan_indices]

# Filter rows for 'PD' and 'PDD' labels
valid_indices = np.logical_or(y_cleaned == 'PDD', y_cleaned == 'PD') # Change these labels for different classification tasks
X_filtered = X_cleaned[valid_indices]
y_filtered = y_cleaned[valid_indices]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Standardize features (scale the data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#####################   SVM MODEL  #####################
# Train an SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Evaluate the SVM model
accuracy = svm_model.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)

# %%

# Perform cross-validation
scores = cross_val_score(svm_model, X_filtered, y_filtered, cv=5)  # X is your feature matrix, y is your target variable

# Print the average accuracy across folds
print("SVM Cross-Validation Accuracy: {:.2f}".format(scores.mean()))
print("SVM Cross-Validation Accuracy(stdev):  {:.2f}".format(scores.std()))

# %%
##################   RANDOM FOREST MODEL  ##################
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Perform cross-validation
scores = cross_val_score(rf_model, X_filtered, y_filtered, cv=5)  # X is your feature matrix, y is your target variable

# Print the average accuracy across folds
print("Random Forest Cross-Validation Accuracy: {:.2f}".format(scores.mean()))
print("RANDOM FOREST Cross-Validation Accuracy(stdev):  {:.2f}".format(scores.std()))


# %%
##################   LOGISTIC REGRESSION MODEL  ##################
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Perform cross-validation (e.g., 5-fold)
scores = cross_val_score(log_reg, X_filtered, y_filtered, cv=5)  # X is your feature matrix, y is your target variable

# Print the average accuracy across folds
print("Logistic Regression Cross-Validation Accuracy: {:.2f}".format(scores.mean()))


