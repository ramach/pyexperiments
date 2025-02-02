import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Preparation and Import
# Load the financial data from CSV file
df = pd.read_csv('balance_sheet_data.csv')

# Preprocessing numerical values
def preprocess_data(df):
    # Normalize numerical features using MinMax scaling
    scaled_df = df.copy()
    for col in ['current_assets', 'total_liabilities', 'working_capital', 'cash_flow']:
        scaled_df[col] = (scaled_df[col].values - np.min(scaled_df[col].values)) / \
                         (np.max(scaled_df[col].values) - np.min(scaled_df[col].values))
    return scaled_df

df = preprocess_data(df)

# Step 2: Neural Network Implementation
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class DatasetClassifier(Dataset):
    def __init__(self, df, label_col='is_high_risk'):
        self.df = df
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = self.df.iloc[idx]
        return torch.FloatTensor(X[:-1]), X.iloc[-1]

# Create training and test datasets
train_dataset = DatasetClassifier(df, label_col='is_high_risk')
test_dataset = DatasetClassifier(df, label_col='is_high_risk')

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define model and training parameters
input_dim = len(df.columns) - 1  # Assuming labels are last column
hidden_dim = 64
output_dim = 1

model = NeuralNetwork(input_dim, hidden_dim, output_dim)

# Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train_model(model, train_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_size
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

def predict_model(model, test_loader):
    predictions = []
    for X, y in test_loader:
        output = model(X)
        predictions.append(output)
    return predictions

# Training the model
train_model(model, train_loader)

# Generate test predictions
test_pred_probs = predict_model(test_loader)
test_labels = [x[1] for x in test_loader]

# Convert predictions to binary classifications (high risk or not)
test_pred_classes = np.round((test_pred_probs > 0.5).astype(int))

print(f"Number of high risks predicted: {sum(test_pred_classes)}")

# Step 3: LLM Integration
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Replace with your API key or use a local model if available
os.environ['PYTORCH_TRANSFORMERS_API_KEY'] = 'your_api_key'

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model_for_cls = AutoModelForSequenceClassification.from_prefixed_model_name(
    'sentence-transformers/all-mpnet-base-v2')

def generate_explanation(pred_class, pred_prob):
    tokenized_input = tokenizer(pred_class.astype(str), return_tensors='pt', max_length=128, padding=True)
    outputs = model_for_cls(**tokenized_input)
    explanation = "Liquidity Risk Explanation: " + outputs[0].modifiers_as_sentences()
    return explanation

# Create a sample prompt for demonstration
sample_prompt = f"Based on the financial data, explain why {test_pred_classes[0]} is considered high risk."

example_explanation = generate_explanation(test_pred_classes[0], test_pred_probs[0])

print(example_explanation)

# Step 4: Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='is_high_risk', y='yearly_profit', data=df)
plt.title('High Risk vs. Profit Analysis')
plt.xlabel('High Risk Classification')
plt.ylabel('Yearly Profit (Million USD)')
plt.show()
