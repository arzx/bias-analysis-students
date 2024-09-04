import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data/exams.csv')

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
X = data.drop(columns=['math score', 'reading score', 'writing score'])
y = data[['math score', 'reading score', 'writing score']].values

# Normalize the features
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Normalize the target variables
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Increased dropout rate to 20%

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the neural network
input_size = X.shape[1]
output_size = y.shape[1]
model = NeuralNet(input_size, output_size)

# Group data by ethnicity
ethnicities = data['race/ethnicity'].unique()

# Initialize lists to store the data for plotting
predicted_reading_scores = []
predicted_writing_scores = []
residual_reading_scores = []
residual_writing_scores = []
ethnicity_labels = []

# Loop over each ethnicity group and make predictions
for ethnicity in ethnicities:
    group_data = data[data['race/ethnicity'] == ethnicity]
    
    X_group = torch.tensor(scaler_X.transform(group_data.drop(columns=['math score', 'reading score', 'writing score'])), dtype=torch.float32)
    y_group_actual = group_data[['math score', 'reading score', 'writing score']].values
    
    # Predict the scores
    y_group_pred = model(X_group).detach().numpy()
    
    # Reverse normalization on predictions
    y_group_pred_original = scaler_y.inverse_transform(y_group_pred)
    
    # Calculate residuals
    residuals = y_group_actual - y_group_pred_original
    
    # Store the results for plotting
    predicted_reading_scores.extend(y_group_pred_original[:, 1])
    predicted_writing_scores.extend(y_group_pred_original[:, 2])
    residual_reading_scores.extend(residuals[:, 1])
    residual_writing_scores.extend(residuals[:, 2])
    ethnicity_labels.extend([ethnicity] * len(y_group_pred_original))

# Create DataFrame for plotting
plot_data = pd.DataFrame({
    'predicted_reading_score': predicted_reading_scores,
    'predicted_writing_score': predicted_writing_scores,
    'residual_reading_score': residual_reading_scores,
    'residual_writing_score': residual_writing_scores,
    'race/ethnicity': ethnicity_labels
})

# Plot predicted reading scores by race/ethnicity
sns.boxplot(x='race/ethnicity', y='predicted_reading_score', data=plot_data)
plt.title('Predicted Reading Scores by Race/Ethnicity')
plt.show()

# Plot residual reading scores by race/ethnicity
sns.boxplot(x='race/ethnicity', y='residual_reading_score', data=plot_data)
plt.title('Residual Reading Scores by Race/Ethnicity')
plt.show()

# Plot predicted writing scores by race/ethnicity
sns.boxplot(x='race/ethnicity', y='predicted_writing_score', data=plot_data)
plt.title('Predicted Writing Scores by Race/Ethnicity')
plt.show()

# Plot residual writing scores by race/ethnicity
sns.boxplot(x='race/ethnicity', y='residual_writing_score', data=plot_data)
plt.title('Residual Writing Scores by Race/Ethnicity')
plt.show()
sns.boxplot(x='gender', y='predicted_math_score', data=plot_data)
plt.title('Predicted Math Scores by Gender')
plt.show()

# Plot residuals by race/ethnicity
sns.boxplot(x='race/ethnicity', y='residual_math_score', data=plot_data)
plt.title('Residual Math Scores by Race/Ethnicity')
plt.show()