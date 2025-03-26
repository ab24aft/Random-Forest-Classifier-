# Breast Cancer Prediction using Random Forest Classifier

Welcome to the **Breast Cancer Prediction** project! This repository demonstrates how to build a **Random Forest Classifier (RFC)** model to predict breast cancer malignancy based on various medical features. The dataset used for this model is the **Breast Cancer dataset** from the `sklearn.datasets`.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Evaluation](#model-evaluation)
- [Visualizations](#visualizations)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview

In this project, we utilize the **Random Forest Classifier** to predict whether a breast tumor is malignant or benign based on a variety of medical features. The model is trained using a dataset from `sklearn.datasets`, and the performance is evaluated using **accuracy**, **confusion matrix**, and **classification report** metrics. The model's feature importance is also visualized to better understand which features contribute most to the decision-making process.

## Dataset

The dataset used is the **Breast Cancer Dataset** from `sklearn.datasets`, which contains medical attributes such as:
- **Radius**
- **Texture**
- **Smoothness**
- **Compactness**
- And more...

The target variable is a binary classification (malignant: 0, benign: 1).

## Installation Instructions

To run this project locally, you'll need to set up your environment by installing the required dependencies.

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-rfc.git
   ```

2. Navigate to the project directory:
   ```bash
   cd breast-cancer-rfc
   ```

3. Install the necessary Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

   This will install the following libraries:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn

## Usage

After setting up your environment, you can execute the Python script to train the **Random Forest Classifier** model, evaluate its performance, and visualize the feature importances.

1. Run the script:
   ```bash
   python breast_cancer_rfc.py
   ```

   The script will output:
   - **Model evaluation metrics** such as **accuracy**, **confusion matrix**, and **classification report**.
   - **Feature importance visualization** showing which features contribute most to the model's predictions.

## Project Structure

The project directory contains the following files:

- **breast_cancer_rfc.py**: Python script that loads the dataset, trains the RFC model, evaluates performance, and visualizes results.
- **requirements.txt**: A list of required dependencies.
- **feature_importance.png**: A plot showing the importance of each feature used in the model.
- **confusion_matrix.png**: A plot showing the confusion matrix of the model.

## Model Evaluation

The model’s performance is evaluated using the following metrics:

1. **Accuracy**: Measures the percentage of correct predictions.
2. **Confusion Matrix**: Shows the breakdown of correct and incorrect predictions for both classes.
3. **Classification Report**: Provides precision, recall, and F1-score for both classes (malignant and benign).

## Visualizations

1. **Confusion Matrix**:
   - This heatmap visualizes the number of true positives, false positives, true negatives, and false negatives.

2. **Feature Importance**:
   - This bar plot shows the relative importance of each feature in making predictions.

## Conclusion

The **Random Forest Classifier** performed with an accuracy of **94%**, demonstrating its strong capability in distinguishing between malignant and benign tumors. The model's performance is further confirmed by its confusion matrix and classification report. The **feature importance visualization** highlights the most influential features, aiding in interpretability.

To improve the model further, you could:
- Tune the model’s hyperparameters for better performance.
- Try alternative models such as **Support Vector Machines** or **Logistic Regression**.
- Use ensemble methods like **boosting** to improve predictions.

Feel free to explore and adapt this code for your own projects or tutorials. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
