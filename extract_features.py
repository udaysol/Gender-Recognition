import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# Directory and metadata file paths
data_dir = 'dataset'
metadata_file = 'other.tsv'
output_file = 'extracted_features_with_labels.csv'

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = np.hstack([
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1),
            np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1),
            np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Load metadata
metadata = pd.read_csv(metadata_file, sep='\t')

# List all files in the data directory
all_files = [f for f in os.listdir(data_dir) if f.endswith('.mp3')]
print(f"Total files in directory: {len(all_files)}")

# Extract features for files listed in metadata
features_list = []
file_names = []
for file_name in tqdm(metadata['path'], desc="Extracting features", unit='files'):
    file_path = os.path.join(data_dir, file_name)
    if file_name in all_files:
        features = extract_features(file_path)
        if features is not None:
            features_list.append(features)
            file_names.append(file_name)
    else:
        print(f"File not found in directory: {file_name}")

# Create DataFrame from features
features_df = pd.DataFrame(features_list)

# Add file names and gender to the DataFrame
features_df['file_path'] = file_names
metadata = metadata[['path', 'gender']]
metadata.rename(columns={'path': 'file_path'}, inplace=True)
merged_df = pd.merge(features_df, metadata, on='file_path')

# Save to CSV
merged_df.to_csv(output_file, index=False)

print(f"Feature extraction and labeling completed. Output saved to {output_file}.")
