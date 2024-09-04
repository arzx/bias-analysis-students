import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Normalize the target variables
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

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
input_size = X_train.shape[1]
output_size = y_train.shape[1]
model = NeuralNet(input_size, output_size)

# Group data by gender
males = data[data['gender'] == label_encoders['gender'].transform(['male'])[0]]
females = data[data['gender'] == label_encoders['gender'].transform(['female'])[0]]

# Get predictions for males and females (using the test set for demonstration)
X_males = torch.tensor(scaler_X.transform(males.drop(columns=['math score', 'reading score', 'writing score'])), dtype=torch.float32)
y_males_pred = model(X_males).detach().numpy()

X_females = torch.tensor(scaler_X.transform(females.drop(columns=['math score', 'reading score', 'writing score'])), dtype=torch.float32)
y_females_pred = model(X_females).detach().numpy()

# Reverse normalization on predictions
y_males_pred_original = scaler_y.inverse_transform(y_males_pred)
y_females_pred_original = scaler_y.inverse_transform(y_females_pred)

# Calculate residuals (difference between actual and predicted scores)
residuals_males = males[['math score', 'reading score', 'writing score']].values - y_males_pred_original
residuals_females = females[['math score', 'reading score', 'writing score']].values - y_females_pred_original

# Create DataFrame for plotting gender analysis
gender_plot_data = pd.DataFrame({
    'gender': ['male'] * len(y_males_pred) + ['female'] * len(y_females_pred),
    'predicted_math_score': list(y_males_pred_original[:, 0]) + list(y_females_pred_original[:, 0]),
    'predicted_reading_score': list(y_males_pred_original[:, 1]) + list(y_females_pred_original[:, 1]),
    'predicted_writing_score': list(y_males_pred_original[:, 2]) + list(y_females_pred_original[:, 2]),
    'residual_math_score': list(residuals_males[:, 0]) + list(residuals_females[:, 0]),
    'residual_reading_score': list(residuals_males[:, 1]) + list(residuals_females[:, 1]),
    'residual_writing_score': list(residuals_males[:, 2]) + list(residuals_females[:, 2])
})

# Plot predicted scores by gender
sns.boxplot(x='gender', y='predicted_math_score', data=gender_plot_data)
plt.title('Predicted Math Scores by Gender')
plt.show()

sns.boxplot(x='gender', y='predicted_reading_score', data=gender_plot_data)
plt.title('Predicted Reading Scores by Gender')
plt.show()

sns.boxplot(x='gender', y='predicted_writing_score', data=gender_plot_data)
plt.title('Predicted Writing Scores by Gender')
plt.show()

# Plot residuals by gender
sns.boxplot(x='gender', y='residual_math_score', data=gender_plot_data)
plt.title('Residual Math Scores by Gender')
plt.show()

sns.boxplot(x='gender', y='residual_reading_score', data=gender_plot_data)
plt.title('Residual Reading Scores by Gender')
plt.show()

sns.boxplot(x='gender', y='residual_writing_score', data=gender_plot_data)
plt.title('Residual Writing Scores by Gender')
plt.show()

# Group data by ethnicity
ethnicities = data['race/ethnicity'].unique()

# Initialize lists to store the data for ethnicity plotting
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

# Create DataFrame for plotting ethnicity analysis
ethnicity_plot_data = pd.DataFrame({
    'predicted_reading_score': predicted_reading_scores,
    'predicted_writing_score': predicted_writing_scores,
    'residual_reading_score': residual_reading_scores,
    'residual_writing_score': residual_writing_scores,
    'race/ethnicity': ethnicity_labels
})

# Plot predicted reading scores by race/ethnicity
sns.boxplot(x='race/ethnicity', y='predicted_reading_score', data=ethnicity_plot_data)
plt.title('Predicted Reading Scores by Race/Ethnicity')
plt.show()

# Plot residual reading scores by race/ethnicity
sns.boxplot(x='race/ethnicity', y='residual_reading_score', data=ethnicity_plot_data)
plt.title('Residual Reading Scores by Race/Ethnicity')
plt.show()

# Plot predicted writing scores by race/ethnicity
sns.boxplot(x='race/ethnicity', y='predicted_writing_score', data=ethnicity_plot_data)
plt.title('Predicted Writing Scores by Race/Ethnicity')
plt.show()

# Plot residual writing scores by race/ethnicity
sns.boxplot(x='race/ethnicity', y='residual_writing_score', data=ethnicity_plot_data)
plt.title('Residual Writing Scores by Race/Ethnicity')
plt.show()