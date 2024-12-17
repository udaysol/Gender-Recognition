from tkinter import * 
from tkinter import filedialog, font, Button

root = Tk()
root.title('Open File')

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

def open():
    global img, input_audio
    input_audio = filedialog.askopenfilename(
        defaultextension=".wav", 
        filetypes=[("WAV files", "*.wav")]
    )

    mylabel = Label(root, text=input_audio).pack()

# Add a heading at the top
heading_font = font.Font(family="Helvetica", size=16, weight="bold")
heading = Label(
    root,
    text="G-Recognition GUI",
    font=heading_font,
    fg="black",
    bg="#f0f0f0",  # Background color for the header
    pady=10        # Padding to add space below the heading
)
heading.pack(pady=10, ipadx=15)


# butt = Button(root, text='Open File', command=open).pack(padx= 40, pady=20)

button_font = font.Font(family="Helvetica", size=12, weight="bold")

# Style and create the button
butt = Button(
    root,
    text="Open File",
    command=open,
    font=button_font,
    bg="#4485CF",      # Modern flat color for background
    fg="white",        # Text color
    activebackground="#45a049",  # Color when button is pressed
    activeforeground="white",    # Text color when button is pressed
    bd=0,              # Remove border for a flat look
    padx=20,           # Add padding for a larger button
    pady=10
)
butt.pack(padx=40, pady=20)


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
    audio = input_audio 
    features_df = extract_features(audio)
    
    # Ensure scaler gets data without column names
    features_df.columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else range(features_df.shape[1])
    
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
    return gender_map.get(prediction[0], 'Unknown')


def show_gender():
    gender = main()

    if gender == 'Male':
        color = 'blue'
    else:
        color = '#AA336A'

    # gen_label = Label(root, text=gender, fg=color).pack()

    label_font = font.Font(family="Helvetica", size=14, weight="bold")

    # Style and create the label
    gen_label = Label(
        root,
        text=gender,
        fg=color,
        font=label_font,
        bg="#f0f0f0",       # Light gray background for a boxed look
        padx=10,            # Horizontal padding inside the box
        pady=5,             # Vertical padding inside the box
    )
    gen_label.pack(padx=20, pady=20)

# butt2 = Button(root, text='Predict gender', command=show_gender).pack(padx= 40, pady=20)

butt = Button(
    root,
    text="Predict Gender",
    command=show_gender,
    font=button_font,
    bg="#4CAF60",      # flat color for background
    fg="white",        # Text color
    activebackground="#45a049",  # Color when button is pressed
    activeforeground="white",    # Text color when button is pressed
    bd=0,              # Remove border for a flat look
    padx=20,           # Add padding for a larger button
    pady=10
)
butt.pack(padx=40, pady=20)


root.mainloop()
