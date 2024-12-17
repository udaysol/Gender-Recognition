import joblib
import librosa
import numpy as np
import pandas as pd

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
    return features
def main():
    audio = input('Enter the path of audio file: ')
    features = extract_features(audio)
    
    # Ensure that features are in the same format as when training
    features_df = pd.DataFrame([features])

    # Apply the scaler
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    # Print raw prediction
    print(f"Raw Prediction: {prediction}")

    # Map the prediction to gender
    gender_map = {
        1: 'Male',
        0: 'Female'
    }    

    print(f"Predicted Gender: {gender_map.get(prediction[0], 'Unknown')}")

if __name__ == "__main__":
    main()
