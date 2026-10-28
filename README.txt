# Histopathology Streamlit app

## Files to place in this folder

Copy these files from your training notebook's working folder into the same folder as `app.py`:

- `custom_cnn_final.keras`
- `mobilenetv2_final.keras`
- `class_mapping.csv`

# Histopathology Image Classification Using Deep Learning

This project classifies histopathology images into five categories related to lung and colon tissue using two deep learning models: a custom Convolutional Neural Network (CNN) and MobileNetV2.

The project includes exploratory data analysis, image preprocessing, model training, model comparison, and a Streamlit web application for image classification.

**Technologies:** Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Pillow, and Streamlit.

The app uses the class order from `class_mapping.csv` when it is present. Otherwise, it uses:
`colon_aca`, `colon_n`, `lung_aca`, `lung_n`, `lung_scc`.

## Run locally (Windows)

1. Open Command Prompt or the terminal in this folder.
2. Create a virtual environment (recommended):

   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Streamlit will show a local URL, usually `http://localhost:8501`.

## Important compatibility note

The app expects the saved models from the provided notebook:
- CNN expects RGB images resized to 224x224 and scaled to [0, 1].
- MobileNetV2 expects RGB images resized to 224x224, then `preprocess_input` is applied.

If your saved model filenames differ, update `MODEL_OPTIONS` in `app.py`.

## Testing requirement

Test one representative image from each of the five class folders. Record the actual folder label, predicted class, and confidence score. Do not change or fabricate results. Confidence is not a medical diagnosis.
