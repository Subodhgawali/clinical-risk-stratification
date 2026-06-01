import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

print("Generating 150,000 advanced clinical records (6 variables)...")
np.random.seed(42)
n_patients = 50000

# GROUP 1: Low Risk (Healthy across the board)
age_1, bmi_1, bp_1 = np.random.normal(28, 5, n_patients), np.random.normal(21, 2, n_patients), np.random.normal(110, 8, n_patients)
chol_1, rhr_1, fbs_1 = np.random.normal(170, 15, n_patients), np.random.normal(65, 5, n_patients), np.random.normal(85, 10, n_patients)

# GROUP 2: Moderate Risk (Borderline metrics)
age_2, bmi_2, bp_2 = np.random.normal(48, 7, n_patients), np.random.normal(27, 3, n_patients), np.random.normal(130, 10, n_patients)
chol_2, rhr_2, fbs_2 = np.random.normal(215, 20, n_patients), np.random.normal(75, 8, n_patients), np.random.normal(110, 15, n_patients)

# GROUP 3: High Risk (Hypertension, High Cholesterol, Pre-diabetic)
age_3, bmi_3, bp_3 = np.random.normal(68, 8, n_patients), np.random.normal(35, 4, n_patients), np.random.normal(155, 12, n_patients)
chol_3, rhr_3, fbs_3 = np.random.normal(260, 25, n_patients), np.random.normal(88, 10, n_patients), np.random.normal(145, 20, n_patients)

# Combine and build the massive DataFrame
df = pd.DataFrame({
    'Patient_ID': range(8000001, 8150001),
    'Age': np.round(np.clip(np.concatenate([age_1, age_2, age_3]), 18, 95)),
    'BMI': np.round(np.clip(np.concatenate([bmi_1, bmi_2, bmi_3]), 15, 50), 1),
    'Resting_Blood_Pressure': np.round(np.clip(np.concatenate([bp_1, bp_2, bp_3]), 80, 200)),
    'Cholesterol': np.round(np.clip(np.concatenate([chol_1, chol_2, chol_3]), 100, 400)),
    'Resting_Heart_Rate': np.round(np.clip(np.concatenate([rhr_1, rhr_2, rhr_3]), 40, 120)),
    'Fasting_Blood_Sugar': np.round(np.clip(np.concatenate([fbs_1, fbs_2, fbs_3]), 60, 250))
}).sample(frac=1).reset_index(drop=True)

# Save the raw data for your portfolio repo
df.to_csv('advanced_clinical_registry.csv', index=False)

print("Scaling all 6 variables and running K-Means Clustering...")
# We feed ALL 6 columns into the algorithm to make the AI super smart
features = ['Age', 'BMI', 'Resting_Blood_Pressure', 'Cholesterol', 'Resting_Heart_Rate', 'Fasting_Blood_Sugar']
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Risk_Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster_Name'] = 'Risk Group ' + df['Risk_Cluster'].astype(str)

print("Building sleek Dark Mode 3D visualization...")
df_sample = df.sample(n=3000, random_state=42)

# Notice we map Blood Sugar to 'size' to squeeze a 4th variable into the visual!
fig = px.scatter_3d(
    df_sample, x='Age', y='BMI', z='Resting_Blood_Pressure',
    color='Cluster_Name',
    size='Fasting_Blood_Sugar', # Larger dots = higher blood sugar
    size_max=12,
    title='Advanced Clinical Risk Stratification (Multi-Variable Clustering)',
    opacity=0.8,
    color_discrete_sequence=['#00e676', '#ff3d00', '#2979ff'],
    template='plotly_dark' 
)

fig.update_layout(
    paper_bgcolor='#0f0f0f', 
    scene=dict(
        xaxis=dict(showbackground=False, gridcolor='#333333'),
        yaxis=dict(showbackground=False, gridcolor='#333333'),
        zaxis=dict(showbackground=False, gridcolor='#333333')
    )
)
fig.update_traces(marker=dict(line=dict(width=0))) 

html_file = "advanced_patient_clusters_3d.html"
fig.write_html(html_file)
print(f"SUCCESS! Open '{html_file}' in your web browser.")
# Calculate the average metrics for each cluster to profile them
print(df.groupby('Cluster_Name').mean().round(1))
