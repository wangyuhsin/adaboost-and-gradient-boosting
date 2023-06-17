# Adaboost and Gradient Boosting

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains Python implementations of two popular boosting algorithms: AdaBoost and Gradient Boosting. The implementations are provided in separate Python files, `adaboost.py` and `gradient_boosting_mse.py`.

## Table of Contents

- [Introduction to AdaBoost](#introduction-to-adaboost)
- [Introduction to Gradient Boosting](#introduction-to-gradient-boosting)
- [Files](#files)
- [Example](#example)
- [Conclusion](#conclusion)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction to AdaBoost

AdaBoost, short for Adaptive Boosting, is a machine learning algorithm that combines multiple "weak" classifiers to create a powerful ensemble classifier. The algorithm iteratively trains weak classifiers on different subsets of the training data, adjusting the weights of misclassified samples to focus on difficult examples. In each iteration, AdaBoost assigns higher weights to misclassified samples, which forces subsequent classifiers to focus on these samples during training. The final classification decision is made by aggregating the predictions of all weak classifiers, weighted by their individual performance.

Adaboost has several advantages, such as its ability to handle both binary and multiclass classification problems, its resistance to overfitting, and its versatility in working with different base classifiers.

## Introduction to Gradient Boosting

Gradient Boosting is another boosting algorithm that can be used for both classification and regression tasks. Unlike AdaBoost, which focuses on classification, Gradient Boosting aims to minimize a loss function by iteratively adding "weak" models to the ensemble. In each iteration, a new model is trained to correct the errors made by the previous models. This approach allows Gradient Boosting to create a strong ensemble model by sequentially improving the predictions.

One of the key advantages of Gradient Boosting is its flexibility in handling various loss functions, making it suitable for a wide range of regression problems. It is also known for its robustness against outliers and its ability to capture complex nonlinear relationships between input features and the target variable.

## Files

### adaboost.py

The `adaboost.py` file implements the Adaboost algorithm for binary classification tasks. It includes functions for loading and preprocessing the data, training the Adaboost model, and making predictions using the trained model. Here are some of the key functions in the file: 
- `accuracy(y, pred)`: Calculates the accuracy of the predicted labels `pred` compared to the true labels `y`. 
- `parse_spambase_data(filename)`: Parses the spambase data from a file and returns the feature matrix `X` and the corresponding labels `y`. 
- `adaboost(X, y, num_iter, max_depth=1)`: Trains an Adaboost model using the input data `X` and labels `y`. It returns an array of decision trees and their corresponding weights. 
- `adaboost_predict(X, trees, trees_weights)`: Makes predictions on the input data `X` using the trained Adaboost model specified by the decision trees `trees` and their weights `trees_weights`.

### gradient_boosting_mse.py

The `gradient_boosting_mse.py` file implements the Gradient Boosting algorithm for regression tasks. It provides functions for loading the dataset, training the Gradient Boosting model, and making predictions. Here are some of the key functions in the file: 
- `load_dataset(path="data/rent-ideal.csv")`: Loads the dataset from the specified file path and returns the feature matrix `X` and the corresponding target values `y`. 
- `gradient_boosting_mse(X, y, num_iter, max_depth=1, nu=0.1)`: Trains a Gradient Boosting model using the input data `X` and target values `y`. It returns the mean of the target values and an array of decision trees. 
- `gradient_boosting_predict(X, trees, y_mean, nu=0.1)`: Makes predictions on the input data `X` using the trained Gradient Boosting model specified by the decision trees `trees`, the mean of the target values `y_mean`, and the shrinkage parameter `nu`.

Make sure to replace the necessary parameters and filenames with your own data.

## Example

Here's an example that demonstrates how to use the AdaBoost implementation:

```python
import adaboost

# Load the dataset
X, y = adaboost.parse_spambase_data("data/spambase.train")
X_test, y_test = adaboost.parse_spambase_data("data/spambase.test")

# Train the AdaBoost ensemble
num_iter = 10
max_depth = 1
trees, trees_weights = adaboost.adaboost(X, y, num_iter, max_depth)

# Make predictions using the trained ensemble
y_hat = adaboost.adaboost_predict(X, trees, trees_weights)
y_hat_test = adaboost.adaboost_predict(X_test, trees, trees_weights)

# Calculate accuracy for the training set
acc_train = adaboost.accuracy(y, y_hat)

# Calculate accuracy for the test set
acc_test = adaboost.accuracy(y_test, y_hat_test)

# Print the accuracies
print("Train Accuracy: %.4f" % acc_train)
print("Test Accuracy: %.4f" % acc_test)
```
Output:
```
Train Accuracy 0.9111
Test Accuracy 0.9190
```

And here's an example that demonstrates how to use the Gradient Boosting implementation:

```python
import gradient_boosting_mse

# Load the dataset
X, y = gradient_boosting_mse.load_dataset("data/tiny.rent.train")
X_test, y_test = gradient_boosting_mse.load_dataset("data/tiny.rent.test")

# Train the Gradient Boosting ensemble
num_iter = 10
max_depth = 1
nu = 0.1
y_mean, trees = gradient_boosting_mse.gradient_boosting_mse(X, y, num_iter, max_depth, nu)

# Make predictions using the trained ensemble
y_hat = gradient_boosting_mse.gradient_boosting_predict(X, trees, y_mean, nu)
y_hat_test = gradient_boosting_mse.gradient_boosting_predict(X_test, trees, y_mean, nu)

# Calculate R2 Score for the training set
r2_train = gradient_boosting_mse.r2_score(y, y_hat)

# Calculate R2 Score for the test set
r2_test = gradient_boosting_mse.r2_score(y_test, y_hat_test)

# Print the R2 Scores
print("Train R2 Score %.4f" % r2_train)
print("Test R2 Score %.4f" % r2_test)
```
Output:
```
Train R2 Score 0.6466
Test R2 Score 0.5297
```

Make sure to replace the filenames with the appropriate paths to your own data.

## Conclusion

Both algorithms provide a powerful and flexible approach to machine learning, allowing you to handle complex problems and improve predictive performance. They have been widely applied in various domains and have achieved state-of-the-art results in many applications.

Feel free to explore and utilize the code in this repository to understand the inner workings of Adaboost and Gradient Boosting. You can apply these algorithms to your own datasets and customize the parameters as needed.

If you have any questions or suggestions, please feel free to reach out to me. Happy boosting!

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

The initial codebase and project structure are adapted from the MSDS 630 course materials provided by the University of San Francisco (USFCA-MSDS). Special thanks to the course instructors for the inspiration.
