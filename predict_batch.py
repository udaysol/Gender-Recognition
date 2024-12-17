import os
import joblib
import librosa
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load the trained model and scaler
model = joblib.load('svm_gender.pkl')
scaler = joblib.load('scaler.pkl')

# Extract features from the audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = np.hstack([
        np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1),
        np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1),
        np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    ])
    
    # Create a DataFrame with these features and drop column names
    features_df = pd.DataFrame([features])
    return features_df

def main():
    folder_path = input('Enter the path of the folder containing .wav files: ')
    
    # Verify folder path exists
    if not os.path.isdir(folder_path):
        print("Invalid folder path. Please ensure the path exists.")
        return
    
    predictions = []
    
    # Process each .wav file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            features_df = extract_features(file_path)
            
            # Ensure scaler gets data without column names
            features_df.columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else range(features_df.shape[1])
            
            # Apply the scaler
            features_scaled = scaler.transform(features_df)
            
            # Make prediction
            prediction = model.predict(features_scaled)
            
            # Map the prediction to gender
            gender_map = {
                1: 'Male',
                0: 'Female'
            }
            gender = gender_map.get(prediction[0], 'Unknown')
            
            # Append file name, prediction, and features to the list
            features_with_metadata = {
                'file_name': file_name,
                'predicted_gender': gender,
            }
            
            # Add each feature as a separate column
            # for i, feature_value in enumerate(features_df.values[0]):
            #     features_with_metadata[f'feature_{i+1}'] = feature_value
            
            predictions.append(features_with_metadata)

    # Convert predictions to a DataFrame and save to CSV
    predictions_df = pd.DataFrame(predictions)
    output_file = os.path.join(folder_path, 'gender_predictions_with_features.csv')
    predictions_df.to_csv(output_file, index=False)
    
    print(f"Predictions with features saved to {output_file}")

if __name__ == "__main__":
    main()
