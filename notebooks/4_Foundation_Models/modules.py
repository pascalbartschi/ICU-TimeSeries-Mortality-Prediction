import numpy as np
import pandas as pd
from sklearn.linear_model import (LogisticRegression, LinearRegression)
from sklearn.ensemble import RandomForestClassifier
import torch


from ollama import chat
from ollama import embeddings
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)
# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader, TensorDataset
# from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
# import torchmetrics

import os
from glob import glob
from pathlib import Path
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib import pyplot as plt

from modules import *

# globals
device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_features(df):
    # df = df.copy()
    df['Time'] = df['Time'].str.replace(':00', '').astype(float)  # Convert '00:00' to 0.0, etc.
    features = []

    for patient_id, group in tqdm(df.groupby("RecordID")):
        row = {'RecordID': patient_id}

        # Time series variables
        ts_vars = [col for col in df.columns if col not in ['RecordID', 'Time']]

        for var in ts_vars:
            times = group['Time']
            vals = group[var].dropna()
            row[f'{var}_mean'] = vals.mean() if not vals.empty else np.nan
            row[f'{var}_max'] = vals.max() if not vals.empty else np.nan
            row[f'{var}_last'] = vals.iloc[-1] if not vals.empty else np.nan
            
            row[f'{var}_std'] = vals.std() if not vals.empty else np.nan
            row[f'{var}_missing_frac'] = vals.isna().sum() / len(vals)
            if vals.count() >= 2:
                # Fit linear regression for slope
                x = times[vals.notnull()].values.reshape(-1, 1)
                y = vals.values.reshape(-1, 1)
                model = LinearRegression()
                model.fit(x, y)
                row[f'{var}_slope'] = model.coef_[0][0]
            else:
                row[f'{var}_slope'] = np.nan


        features.append(row)

    return pd.DataFrame(features)

def generate_patient_summary(row):
    return (
        f"age {int(row['Age_max'])} years, "
        f"last GCS score {int(row['GCS_last'])}, "
        f"mean GCS score {int(row['GCS_mean'])}, "
        f"mean urine output {round(row['Urine_mean'])} ml, "
        f"last BUN level {round(row['BUN_last'], 1)} mg/dL, "
        f"last WBC count {round(row['WBC_last'], 1)} ×10^9/L, "
        f"mean heart rate {int(row['HR_mean'])} bpm, "
        f"mean FiO₂ {int(row['FiO2_mean'])}%, "
        f"variation in pH {round(row['pH_std'], 2)}, "
        f"mean bicarbonate {int(row['HCO3_mean'])} mmol/L"
    )
def build_few_shot_prompt(df, label_col="In-hospital_death", max_examples=3, scoring = False):
    assert len(df) > max_examples, "Not enough patients for few-shot + test split."

    if scoring: 
         force_model_output = "Respond with exactly one digit between 1-10 indicating the probability of survival, where 10 is most likely survival and 1 least likely."
    else: 
        force_model_output = "Respond with exactly one word: either 'died' or 'survived'."
        
    prompt = f"You are a medical AI assistant. Based on ICU data summaries, predict patient outcomes. {force_model_output} Do not explain your answer.\n\n"


    # Few-shot examples
    for i in range(max_examples):
        row = df.iloc[i]
        summary = generate_patient_summary(row)
        outcome = "died" if row[label_col] == 1 else "survived"
        prompt += f"Example {i+1}:\nPatient summary: {summary}.\nOutcome: {outcome}\n\n"

    # Add test cases
    test_cases = []
    outcomes = []
    for i in range(max_examples, len(df)):
        row = df.iloc[i]
        summary = generate_patient_summary(row)
        test_cases.append(f"Patient summary: {summary}.\nOutcome:")
        outcomes.append(r"died" if row[label_col] == 1 else "survived")

    return prompt, test_cases, outcomes

def query_llm(prompt, test_case, model='gemma2:2b'):
    messages = [{'role': 'user', 'content': f"{prompt}\n\n{test_case}"}]
    stream = chat(model=model, messages=messages, stream=True)

    output = ""
    for chunk in stream:
        output += chunk['message']['content']
    return output.strip()

def evaluate_llm_predictions(prompt, test_cases, outcomes, model="chronos", max_cases=None, verbose = False):

    max_cases = min(len(test_cases), max_cases)
    y_true = []
    y_pred = []
    y_scores = []  # for optional numeric scoring later

    for i, test_case in tqdm(enumerate(test_cases[:max_cases]), desc="Patient Cases", total = max_cases):
        
        prediction_raw = query_llm(prompt, test_case, model=model).lower().strip()
       
        truth = 1 if outcomes[i] == "died" else 0

        # Parse prediction
        if "died" in prediction_raw:
            pred = 1
            prob = 0.0
        elif "survived" in prediction_raw:
            pred = 0
            prob = 1.0
        elif prediction_raw.isdigit():
            # If using scoring (e.g., 1–10), treat >5 as survival
            score = int(prediction_raw)
            pred = 0 if score >= 6 else 1
            prob = score / 10
        else:
            print("⚠️ Unexpected output format, treating as incorrect.")
            pred = -1
            prob = 0.5

        if verbose: 
            print(f"\n--- Test Case {i+1} ---")
            print(f"Raw LLM Output: {prediction_raw}")
            print(f"Prediction: {'died' if pred == 1 else 'survived'}")
            print(f"Truth: {'died' if truth == 1 else 'survived'}")

        y_true.append(truth)
        y_pred.append(pred)
        y_scores.append(prob)

    # Filter valid predictions
    valid_idx = [i for i, p in enumerate(y_pred) if p in [0, 1]]
    y_true = [y_true[i] for i in valid_idx]
    y_pred = [y_pred[i] for i in valid_idx]
    y_scores = [y_scores[i] for i in valid_idx]

    # Metrics
    print("\n📊 Confusion Matrix:")
    # print(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("LLM Predictions: Confusion Matrix")
    plt.show()

    if len(set(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        print(f"AUROC : {auroc:.4f}")
        print(f"AUPRC : {auprc:.4f}")

        RocCurveDisplay.from_predictions(y_true, y_scores)
        plt.title("ROC Curve")
        plt.show()

        PrecisionRecallDisplay.from_predictions(y_true, y_scores)
        plt.title("Precision-Recall Curve")
        plt.show()
    else:
        print("⚠️ Not enough class diversity to compute AUROC/AUPRC.")

    return y_true, y_pred, y_scores

def get_chronos_embeddings(X_tensor, num_features, pipeline, return_stack = False):

    X_tensor = X_tensor.to(device)
    embedding_sum = None
    embedding_list = []
    num_valid = 0
    
    
    for feature_idx in tqdm(range(num_features)):
    
        # slice one feature vector
        feature_data = X_tensor[:, :, feature_idx].to(pipeline.model.device)
    
        # get embeddings
        with torch.no_grad():
            embeddings, _ = pipeline.embed(feature_data)
            embeddings = embeddings.cpu()
            embedding_list.append(embeddings)
    
        # compute the mean incrementally to save memory (instead of stacking with torhc)
        if embedding_sum is None: 
            embedding_sum = embeddings.clone()
        else:
            embedding_sum +=embeddings
    
        num_valid += 1
    
    final_embeddings = embedding_sum / num_valid
    
    # verify correct shape
    assert final_embeddings.shape == embeddings.shape, "Mean was taken across the wrong dimension"

    if return_stack: 
        return torch.stack(embedding_list, dim=1).cpu()
    else: 
        return final_embeddings

