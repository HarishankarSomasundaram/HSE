from datetime import timedelta

from django.shortcuts import render
from django.http import JsonResponse
import time  # Simulate training delay
import random
from django.http import HttpResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,precision_score, recall_score, f1_score
from numpy import mean, std
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import os
import joblib
import io
import base64
import plotly.express as px
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent.parent

def predict_strain(request):
    # return HttpResponse("Hello world!")
    return render(request, 'hospital_strain.html')

# Load CSV file into pandas DataFrame
def load_csv_data():
    # Adjust the path to your CSV file
    file_path = BASE_DIR / 'predict_strain/static/trolleys_strain.csv'
    return pd.read_csv(file_path)

def create_chart():
    # Load CSV data
    df = load_csv_data()

    # Example: Create a bar chart of hospital strain data
    plt.figure(figsize=(10, 6))
    df['hospital'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Hospital Strain')
    plt.xlabel('hospital')
    plt.ylabel('Strain Level')

    # Save chart as PNG
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')  # Convert to base64 string
    buf.close()
    return img_str


def create_plotly_chart():
    df = load_csv_data()
    fig = px.bar(df, x='hospital', title="Hospital Strain Analysis")
    fig.write_html('hospital_strain_plot.html')  # Save the plot as HTML or render it to Django template


def visualization(request):
    # Generate chart
    chart_img = create_chart()

    return render(request, 'hospital_strain.html', {
        'chart_img': chart_img
    })



def predict_output(request):
    if request.method == 'POST':
        model_name = request.POST.get('model')
        region = request.POST.get('region')
        hospital = request.POST.get('hospital')
        date = request.POST.get('date')
        surge_capacity = int(request.POST.get('surge_capacity', 0))
        delayed_transfers = int(request.POST.get('delayed_transfers', 0))
        waiting_24hrs = int(request.POST.get('waiting_24hrs', 0))
        waiting_75y_24hrs = int(request.POST.get('waiting_75y_24hrs', 0))
        new_data = {
            'region': [region],
            'hospital': [hospital],
            'date': [date],
            'Surge Capacity in Use (Full report @14:00)': [surge_capacity],
            'Delayed Transfers of Care (As of Midnight)': [delayed_transfers],
            'No of Total Waiting >24hrs': [waiting_24hrs],
            'No of >75+yrs Waiting >24hrs':[waiting_75y_24hrs]
        }
        file_path = BASE_DIR / 'predict_strain/static/HSE.trolleys.csv'
        df = pd.read_csv(file_path)

        df = df[
            (df['region'] == region) &
            (df['hospital'] == hospital)
            ].sort_values('date')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load( BASE_DIR / 'predict_strain/static/models/LSTM/lstm_model.pth', map_location=device, weights_only=False)
        model = LSTMRegressor(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        scaler = joblib.load( BASE_DIR / 'predict_strain/static/models/LSTM/scaler.pkl')

        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        last_date = df['date'].max()
        # forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

        features = ['ED Trolleys', 'Ward Trolleys', 'Surge Capacity in Use (Full report @14:00)',
                    'Delayed Transfers of Care (As of Midnight)', 'No of >75+yrs Waiting >24hrs']

        df[features] = scaler.transform(df[features])
        last_seq = df[features].values[-7:].copy()

        predictions = []
        for _ in range(7):
            input_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(input_tensor).item()
            predictions.append(pred)
            next_input = np.append(last_seq[1:], [[pred] * last_seq.shape[1]], axis=0)
            last_seq = next_input

        forecast_data = [
            {
                "date": (last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d'),
                "day": (last_date + timedelta(days=i + 1)).strftime('%A'),
                "value": float(round(pred, 1))  # Rounds to 1 decimal place
            }
            for i, pred in enumerate(predictions)
        ]

        response =''
        # if predictions[0] == 'Low':
        #     response = '0-5'
        #     value = 2.5
        # elif predictions[0] == 'Moderate':
        #     response = '6-16'
        #     value = 9
        # elif predictions[0] == 'High':
        #     response = '17+'
        #     value = 22
        # return JsonResponse({f"Predicted strain level: {predictions[0]} ({response} trolleys approximately)"},safe=False)
        return JsonResponse({
            "prediction": f"Using LSTM Model",
            # "value": forecast_data,
            "hospital": hospital,
            "date":date,
            "forecast": forecast_data
        })
    return JsonResponse({'error': 'Invalid request'}, status=400)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out.squeeze()
def train_model(request):
    if request.method == 'POST':
        model_name = request.POST.get('model').replace('_', ' ').title()

        mc_runs = int(request.POST.get('mc_runs', 10))

        # 'hospital_strain_prediction / predict_strain / static / HSE.trolleys.csv'
        df = pd.read_csv(BASE_DIR / 'predict_strain/static/HSE.trolleys.csv')  # could be removed
        # print(df.head())
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        df['Strain Level'] = df['Total Trolleys'].apply(
            lambda x: 'Low' if x <= 5 else 'Moderate' if x <= 16 else 'High')

        df.to_csv(BASE_DIR / 'predict_strain/static/trolleys_strain.csv', index=False)

        df.drop('_id', axis=1, inplace=True)
        df.drop('date', axis=1, inplace=True)
        column_order = ['region', 'hospital', 'year', 'month', 'day_of_week',
                        'ED Trolleys', 'Ward Trolleys', 'Total Trolleys',
                        'Surge Capacity in Use (Full report @14:00)',
                        'Delayed Transfers of Care (As of Midnight)',
                        'No of Total Waiting >24hrs', 'No of >75+yrs Waiting >24hrs','Strain Level']
        df = df[column_order]
        # df['Strain Level'] = df['Total Trolleys'].apply(assign_strain_level)


        df = pd.get_dummies(df, columns=['region', 'hospital', 'day_of_week', 'month', 'year'], dtype=int)
        # Encode the target variable 'Strain Level' (still using LabelEncoder for target)
        le_strain = LabelEncoder()
        df['Strain Level Encoded'] = le_strain.fit_transform(df['Strain Level'])
        # Define features (all columns except 'Total Trolleys' and target)
        features = [col for col in df.columns if
                    col not in ['ED Trolleys','Ward Trolleys','Total Trolleys', 'Strain Level', 'Strain Level Encoded']]
        X = df[features]
        y = df['Strain Level Encoded']

        # y = df['Strain Level']
        # X = df.drop('Strain Level', axis=1)
        # Apply SMOTE
        # undersampler = RandomUnderSampler(random_state=42)
        # X_resampled, y_resampled = undersampler.fit_resample(X, y)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        # Check class distribution after SMOTE
        # print("\nClass Distribution After SMOTE (Encoded):")
        # print(pd.Series(y_resampled).value_counts())


        # Step 4: Scale the Features
        scaler = StandardScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)
        X_resampled_scaled_df = pd.DataFrame(X_resampled_scaled, columns=features)

        joblib.dump(scaler, BASE_DIR / 'predict_strain/static/models/scaler.pkl')
        joblib.dump(le_strain, BASE_DIR / 'predict_strain/static/models/le_strain.pkl')
        joblib.dump(features, BASE_DIR / 'predict_strain/static/models/feature_names.pkl')

        model = get_models(model_name)
        results = evaluate_model(model_name, model, mc_runs, X_resampled_scaled_df, y_resampled)
        print("Model and preprocessors saved successfully.")
        result = {
            'result': f'{model_name} trained successfully',
            'metrics': {
                'accuracy': round(results['accuracy_mean']*100, 2),
                'precision': round(results['precision_mean']*100, 2),
                'recall': round(results['recall_mean']*100, 2),
                'f1_score': round(results['f1_mean']*100, 2)
            }
        }

        # Simulating training process
        # time.sleep(2)  # Simulate processing delay

        # # Dummy accuracy scores for each model
        # results = {
        #     'random_forest': round(random.uniform(85, 95), 2),
        #     'logistic_regression': round(random.uniform(75, 85), 2),
        #     'svm': round(random.uniform(80, 90), 2)
        # }
        #
        # accuracy = results.get(model, "Unknown Model")

        # return JsonResponse({'result': f"{model_name} trained with {mc_runs} MC runs. Accuracy: {accuracy*100:.2f}%. Standard Deviation: {std*100:.2f}%" })
        return JsonResponse(result)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def evaluate_model(model_name, model, mc_runs,  X, y):
    acc, prec, rec, f1 = [], [], [], []

    for i in range(mc_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)  # Split dataset with different seeds
        dt = model.fit(X_train, y_train)  # Fit the model
        y_pred = dt.predict(X_test)  # Predict
        a = accuracy_score(y_test, y_pred)  # Compute accuracy
        acc.append(a)  # Append accuracy
        acc.append(accuracy_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        rec.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1.append(f1_score(y_test, y_pred, average='macro', zero_division=0))


    if model_name == 'Random Forest':
        joblib.dump(model, BASE_DIR / 'predict_strain/static/models/rf_model.pkl')
    elif model_name == 'Logistic Regression':
        joblib.dump(model, BASE_DIR / 'predict_strain/static/models/lr_model.pkl')
    elif model_name == 'Support Vector Classification':
        joblib.dump(model, BASE_DIR / 'predict_strain/static/models/svc_model.pkl')


    return {
        'accuracy_mean': np.mean(acc),
        'accuracy_std': np.std(acc),
        'precision_mean': np.mean(prec),
        'recall_mean': np.mean(rec),
        'f1_mean': np.mean(f1)
    }

# Step 5: Define the models
def get_models(modelName):
    if modelName == 'Random Forest':
        return RandomForestClassifier(random_state=42)
    elif modelName == 'Logistic Regression':
        return LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    elif modelName == 'Support Vector Classification':
        return SVC(decision_function_shape='ovr', random_state=42)

# def assign_strain_level(total_trolleys):
#     if total_trolleys <= 10:
#         return 0
#     elif total_trolleys <= 20:
#         return 1
#     else:
#         return 2