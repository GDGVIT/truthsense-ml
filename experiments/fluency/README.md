
# Fluency Detection Model Experiments

This folder contains our experiments for building a robust audio fluency detection model using a variety of audio features such as the pitch, zero-crossing rate, RMS energy, etc.

## Models Explored

We experimented with several machine learning models:

- **LightGBM**
- **Random Forest**
- **XGBoost**
- **Voting Ensemble** (combining the above)
- **Stacking Ensemble** (with Linear Regression as the meta-model)

All training scripts for these models are located in the `train` folder.

## Data

You can find the dataset we used [on HuggingFace](https://github.com/KaminiSabu/ReadingConfidenceDataset/tree/main). (Let us know if you have trouble accessing it!)

## Preprocessing

To prepare the data, use the `preprocess.py` script. This will process the entire dataset and save the resulting inputs and outputs in a new `outputs` folder.

## Model Weights

Trained model weights are available in the `weights` folder, on the same dataset that we mention under [Data](#data).

## Testing

We tested all our models on both our own speaking samples and the full [Tim Urban TED talk](https://youtu.be/arj7oStGLkU?si=JnqCOG9LV-CtCS--). You can run your own tests using `testing.ipynb` — just update the paths to your `.wav` files.

The testing file assumes for now that the samples are stored under a `samples` folder in the root project directory, change that path in order to test the models on your data.
