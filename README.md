# SupportVectorMachine

Overview of Support Vector Machines (SVM)
SVM is a supervised machine learning algorithm used for classification and regression tasks. It works by finding a hyperplane that best separates different classes in the feature space. The key concepts include:

Hyperplane: A decision boundary that separates different classes.
Support Vectors: Data points that are closest to the hyperplane, which influence its position and orientation.
Kernel Trick: A technique that allows SVM to perform well in non-linear spaces by transforming the data into a higher-dimensional space.
Step-by-Step Process
1. Environment Setup
Make sure you have the necessary libraries installed. You can use libraries such as pandas, numpy, matplotlib, and scikit-learn.

bash
Copy code
pip install pandas numpy matplotlib scikit-learn
2. Load the Datasets
Start by loading the datasets using pandas.

python
Copy code
import pandas as pd

# Load datasets
salaries_data = pd.read_csv('Position_Salaries.csv')
parkinsons_data = pd.read_csv('parkinsons_new.csv')
3. Data Preprocessing
For Position_Salaries.csv:

Extract features (level) and target variable (salary).
Reshape the data as needed for SVM.
python
Copy code
# Features and target for regression
X = salaries_data.iloc[:, 1:2].values  # Position Level
y = salaries_data.iloc[:, 2].values     # Salary
For parkinsons_new.csv:

Identify features and target variable.
Handle any missing values or encode categorical variables if necessary.
python
Copy code
# Features and target for classification
X = parkinsons_data.drop(columns=['status']).values  # All features except 'status'
y = parkinsons_data['status'].values                  # Target variable (0 or 1)
4. Split the Data
Divide your data into training and testing sets to evaluate the performance of your models.

python
Copy code
from sklearn.model_selection import train_test_split

# For regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# For classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y, test_size=0.2, random_state=42)
5. Feature Scaling (if necessary)
SVM is sensitive to the scale of the data, so it's essential to standardize features.

python
Copy code
from sklearn.preprocessing import StandardScaler

# For regression
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# For classification
scaler_class = StandardScaler()
X_train_class = scaler_class.fit_transform(X_train_class)
X_test_class = scaler_class.transform(X_test_class)
6. Implement Support Vector Regression (SVR)
Using the regression data, fit an SVR model.

python
Copy code
from sklearn.svm import SVR

# Create the SVR model
svr_model = SVR(kernel='rbf')  # Using Radial Basis Function Kernel
svr_model.fit(X_train, y_train)
7. Predictions and Visualization (SVR)
Make predictions and visualize the results.

python
Copy code
import numpy as np
import matplotlib.pyplot as plt

# Predictions
y_pred = scaler_y.inverse_transform(svr_model.predict(X_test))

# Visualization
plt.scatter(X, y, color='red')  # Original data
plt.scatter(X_test, y_pred, color='blue')  # Predictions
plt.title('Support Vector Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
8. Implement Support Vector Classification (SVC)
Now, fit a SVC model on the classification data.

python
Copy code
from sklearn.svm import SVC

# Create the SVC model
svc_model = SVC(kernel='linear')  # Using Linear Kernel
svc_model.fit(X_train_class, y_train_class)
9. Predictions and Evaluation (SVC)
Make predictions and evaluate the performance of the classification model.

python
Copy code
from sklearn.metrics import confusion_matrix, classification_report

# Predictions
y_pred_class = svc_model.predict(X_test_class)

# Evaluation
cm = confusion_matrix(y_test_class, y_pred_class)
report = classification_report(y_test_class, y_pred_class)
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', report)
10. Hyperparameter Tuning (Optional)
Consider tuning hyperparameters (like the kernel type and C parameter) using techniques such as Grid Search or Random Search.

python
Copy code
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, scoring='accuracy')
grid_search.fit(X_train_class, y_train_class)

print('Best Parameters:', grid_search.best_params_)
Conclusion
In this guide, we covered the essential steps to implement SVM for regression and classification using two different datasets. You can further experiment with different kernels, tune hyperparameters, and visualize results to gain deeper insights into your models.



