# Multiclass Classification with Perceptron and Logistic Regression

## Project Overview
This project demonstrates the implementation of multiclass classification using two machine learning algorithms: Perceptron and Logistic Regression. The aim is to understand and implement these algorithms from scratch, avoiding the use of built-in classifiers from libraries such as sklearn and numpy. This hands-on approach helps in gaining a deeper understanding of how these algorithms work.

## Technology and Implementation
### Perceptron
The Perceptron is a simple binary classifier that can be extended to handle multiple classes using techniques such as One-Versus-The-Rest (OvR) and One-Versus-One (OvO).

1. One-Versus-The-Rest (OvR): This technique involves training one classifier per class, with the samples of that class as positive samples and all other samples as negatives.
2. One-Versus-One (OvO): This technique involves training a classifier for every pair of classes. For a problem with k classes, k*(k-1)/2 classifiers are trained.

### Logistic Regression
Logistic Regression can also be extended for multiclass classification using the Softmax function, transforming model outputs into probability distributions over the classes.

### Dataset
- Generated a dataset with 2 coordinates (x1, x2), 4 clusters (4 classes), and 50 points per cluster.
- Dataset split into 80% training and 20% testing sets.

### Files
- **both_techniques.py:** Contains the implementation of Perceptron classifier using OvR and OvO techniques.
- **logistic_regression.py:** Contains the implementation of Logistic Regression for multiclass classification using the Softmax function.

## Results
Perceptron One-Versus-The-Rest

![One-Versus-The-Rest](https://github.com/user-attachments/assets/ef1ce8c7-bf95-4bd8-ac04-1e584140c0dd)

Perceptron One-Versus-One

![One-Versus-One](https://github.com/user-attachments/assets/b7dfafb3-0339-4510-89c2-266467f7070e)

Logistic Regression Multiclass Classification 

![Multiclass](https://github.com/user-attachments/assets/6981b8f2-dc11-4ccd-9f9b-9a55efe5d353)

## Conclusion
This project illustrates the implementation of multiclass classification using Perceptron and Logistic Regression from scratch. By avoiding built-in classifiers, the goal is to deepen the understanding of these fundamental algorithms in machine learning.

## Usage
1. Clone the Repository
``` bash
git clone https://github.com/yourusername/multiclass-classification.git
cd multiclass-classification
```

2. Run Perceptron Classifier
``` bash
python both_techniques.py
```

3. Run Logistic Regression Classifier
```bash
python logistic_regression.py
```

## What I Learned
**Data Generation:**
- How to generate synthetic datasets with specified means and standard deviations.
- Importance of dataset splitting for training and testing.

**Perceptron Algorithm:**
- Implementation of a basic binary Perceptron classifier.
- Extension of binary Perceptron to multiclass classification using OvR and OvO techniques.
- Handling class imbalance and converting multiclass labels to binary.

**Logistic Regression:**
- Building a Logistic Regression model from scratch.
- Application of the Softmax function for multiclass classification.
- Understanding gradient descent for parameter optimization.

**Visualization:**
- Plotting decision boundaries to visually assess the classifier performance.
- Importance of visual aids in understanding model behavior.

**Accuracy Calculation:**
- Methods to evaluate model accuracy on test data.
- Differences in performance metrics across different classification techniques.

**General Machine Learning Concepts:**
- The significance of not using built-in libraries for educational purposes.
- The trade-offs between different multiclass classification techniques (OvR vs OvO).
- The practical challenges in implementing machine learning algorithms from scratch.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
