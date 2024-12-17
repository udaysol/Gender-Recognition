# G-recognition through Voice

**G-recognition through Voice** is a machine learning-based project designed to predict a speaker's gender based on audio input. This system is built using Python and supports both single and batch predictions. It includes tools for feature extraction, data preparation, and model training, allowing users to either use the pre-trained model or train the model themselves.

> **Note**: This project is a work in progress and not yet fully complete.

---

## Features

- Predict speaker gender (Male/Female) using a trained SVM model.
- Support for single-file predictions via a GUI.
- Batch mode processing for analyzing multiple audio files and saving results to a CSV file.
- Tools for custom model training, including data preparation and feature extraction.

---

## Folder Structure for Training

To train the model, organize your dataset and files as follows:

```
g-recognition-voice/
├── dataset/
│   ├── <audio_file_1>.wav
│   ├── <audio_file_2>.wav
│   ├── ...
│   ├── other.tsv  # Metadata file containing file paths and gender labels
├── extract_features.py
├── train_test.py
├── model.py
├── gui.py
├── predict_batch.py
├── requirements.txt
```

- **`dataset/`**: This folder should contain all your `.wav` audio files and the metadata file (`other.tsv`) with columns specifying file paths and gender labels.

---

## Prerequisites

1. **Hardware**:
   - Multi-core CPU
   - Minimum 8GB RAM
2. **Software**:
   - Python 3.x
   - Required libraries:
     - `numpy`
     - `pandas`
     - `librosa`
     - `joblib`
     - `tqdm`
     - `scikit-learn`
     - `tkinter`

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/g-recognition-voice.git
   cd g-recognition-voice
   ```

2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your audio files are in `.wav` format and organized in a dataset directory.

---

## How to Use

### Pre-trained Model
1. **Single File Prediction**:
   - Run the `gui.py` script:
     ```bash
     python gui.py
     ```
   - Use the GUI to upload an audio file and predict gender.
2. **Batch Processing**:
   - Run the `predict_batch.py` script:
     ```bash
     python predict_batch.py
     ```
   - Specify the folder containing `.wav` files. Results will be saved as a CSV file in the same folder.

---

### Training the Model

To train the model yourself, follow these steps:

1. **Extract Features**:
   - Place your dataset (audio files and metadata) in the `dataset/` folder.
   - Run the `extract_features.py` script:
     ```bash
     python extract_features.py
     ```
   - This will create `extracted_features_with_labels.csv` containing audio features and labels.

2. **Prepare Train/Test Data**:
   - Run the `train_test.py` script:
     ```bash
     python train_test.py
     ```
   - This will split the data into training and testing sets and save them as `x_train.csv`, `x_test.csv`, `y_train.csv`, and `y_test.csv`.

3. **Train the Model**:
   - Run the `model.py` script:
     ```bash
     python model.py
     ```
   - This trains an SVM model and saves the scaler (`scaler.pkl`) and model (`svm_gender.pkl`).

4. **Use the Trained Model**:
   - Place the `svm_gender.pkl` and `scaler.pkl` files in the project directory to use them for predictions.

---

## Limitations

- Binary classification only (Male/Female).
- Sensitive to noise and low-quality audio inputs.
- Does not include non-binary or gender-nonconforming categories.
- Further testing is needed to ensure scalability and robustness.

---

## Future Enhancements

- Add support for non-binary gender classification.
- Improve noise robustness and scalability.
- Incorporate real-time processing capabilities.

---

## Acknowledgments

- Mozilla Common Voice dataset for providing diverse and inclusive audio samples.
- Python and its extensive libraries, including Librosa and Scikit-learn, for making audio processing and machine learning accessible.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Let me know if any further details need to be added or adjusted!

