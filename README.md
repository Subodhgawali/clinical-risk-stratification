# clinical-risk-stratification
Unsupervised Machine Learning model using K-Means Clustering to automatically stratify 150,000 synthetic patient records into cardiovascular risk tiers. Built with Python, Scikit-Learn, and Plotly 3D.
<img width="1918" height="967" alt="cluster-preview" src="https://github.com/user-attachments/assets/39a0c6f0-d996-46a2-8954-57924e0f15e5" />

**Live Interactive Model:** [https://Subodhgawali.github.io/clinical-risk-stratification/](https://Subodhgawali.github.io/clinical-risk-stratification/)

---

### The Business Problem
In enterprise healthcare, identifying at-risk patients early is critical for resource allocation and preventative care. However, when dealing with millions of records across multiple medical variables (Age, BMI, Blood Pressure, Cholesterol, etc.), manual risk-scoring becomes impossible. 

This project utilizes a K-Means clustering algorithm to automatically discover hidden patterns in patient data, allowing healthcare providers to instantly segment populations into "Low," "Moderate," and "High" risk tiers without requiring manual medical labeling.

### Technical Execution
* **Data Engineering:** Generated a highly realistic, synthetic cardiovascular registry of 150,000 patient records across 6 distinct medical variables using `numpy` and `pandas`.
* **Machine Learning:** Standardized the multidimensional data using `StandardScaler` to ensure unbiased distance calculations, then applied a `KMeans` algorithm to partition the data into 3 optimal clusters.
* **Data Visualization:** Reduced the dimensionality for the UI and mapped the output to an interactive 3D web environment using `plotly`. Added dynamic sizing based on Fasting Blood Sugar levels to squeeze a 4th dimension into the 3D space.

### Cluster Profiles (The Results)
By analyzing the cluster centroids, the algorithm successfully identified three distinct patient personas:
* **🔵 Cluster 0 (Low Risk):** Younger demographic (Avg Age: 28) with healthy BMI (~21) and normal blood pressure (~110). 
* **🟢 Cluster 1 (Moderate Risk):** Middle-aged (Avg Age: 48) with slightly elevated BMI (~27) and pre-hypertension (~130).
* **🔴 Cluster 2 (High Risk):** Older demographic (Avg Age: 68) showing clear markers for hypertension (BP ~155), high cholesterol, and elevated fasting blood sugar.

### Tools Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Plotly (Interactive 3D rendering)
