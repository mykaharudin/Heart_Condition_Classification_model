# Heart Condition Classification 

# Heart Condition Classification Project Using K-NN and Decision Tree

This project aims to classify heart conditions into two classes (Normal and Abnormal) using **K-Nearest Neighbors (K-NN)** and **Decision Tree** algorithms. The dataset used consists of heart data classified as **Normal** and **Abnormal**, with a total of 14,550 entries.

## Steps

### 1. Data Preparation

1. **Merging Data**:
   - The dataset consists of two files: `ptbdb_normal.csv` and `ptbdb_abnormal.csv`.
   - Merge these two datasets and add a `target` column with the value `0` for Normal and `1` for Abnormal.

2. **Balancing the Data**:
   - Since the data is imbalanced (more Abnormal data), take a subset of the Abnormal data to match the number of Normal data entries, creating a balanced dataset.

3. **Splitting the Data**:
   - Split the data into **80% training** and **20% testing** sets to train and evaluate the model.

4. **Handling Missing Values**:
   - Check for any missing values (`NaN`) in the data.
   - Remove columns with many missing values, then fill any remaining missing values with the median of each column.

### 2. Feature Extraction

To detect heart conditions, we extracted features from the heart signal using the following methods:

1. **Fourier Transform**:
   - Apply Fourier Transform to capture the frequency characteristics of the heart signal.
   - Extract features such as the mean (`mean_fft`), maximum value (`max_fft`), and standard deviation (`std_fft`).

2. **Basic Statistics** (as a substitute for GLCM, which was less suitable for signal data):
   - Since the signal data did not vary enough for GLCM to be effective, we used basic statistical features:
     - **Mean**: The average value of the signal.
     - **Standard Deviation (std_dev)**: The spread of data around the mean.
     - **Skewness**: The asymmetry of the data distribution.
     - **Kurtosis**: The "peakedness" or sharpness of the data distribution.

### 3. Model Building

We used two models for classification:

1. **K-Nearest Neighbors (K-NN)**:
   - Applied the K-NN algorithm with `K=3` and **Euclidean distance**.
   - The model considers the 3 closest neighbors to classify new samples.

2. **Decision Tree**:
   - Implemented a Decision Tree with a maximum depth of `5` to avoid overfitting.
   - The model works by splitting data based on the best feature at each node.

### 4. Model Evaluation

After training the models, we evaluated their performance using the following metrics:

1. **Confusion Matrix**:
   - Displays the count of correct and incorrect predictions for each class.
   
2. **Classification Report**:
   - Shows **Precision**, **Recall**, and **F1-Score** metrics for each class (Normal and Abnormal).

3. **Model Accuracy**:
   - K-NN achieved an accuracy of **73%**.
   - Decision Tree achieved an accuracy of **70%**.

### 5. Conclusion

- **K-NN** is more accurate for detecting **abnormal conditions**.
- **Decision Tree** provides results that are easier to interpret due to its "if-then" rule structure.
- The choice of model depends on the requirement:
  - **K-NN** for better abnormal condition detection.
  - **Decision Tree** if model interpretability is more important.

## How to Run the Project

1. Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas scikit-learn seaborn matplotlib scipy

References
Dataset sourced from [Kaggle Heart Disease Dataset](https://www.kaggle.com/).