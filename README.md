# KNN-Model-With-Euclidean-Distance-

<p> The provided program implements a k-nearest neighbors (k-NN) algorithm for predicting the Corruption Perception Index (CPI) of countries based on macro-economic and social features. The program allows for both standard k-NN and weighted k-NN prediction models, where the user can specify the value of k. Additionally, the program demonstrates the normalization of data using range normalization, which is crucial when dealing with features measured in different units.

Here's a brief breakdown of the program:

1. Data Handling: The program reads the dataset from a file and extracts the relevant features and target variable (CPI) from it.
2. Distance Calculation: Euclidean distance is calculated between data points to measure similarity.
3. Prediction Functions: Two prediction functions are defined: one for standard k-NN and another for weighted k-NN, allowing for flexibility in model selection.
4. Normalization: The program includes a function for normalizing the data using range normalization to ensure fair comparison between features with different scales.
5. Prediction for Specific Country (Russia): The program provides predictions for the CPI of Russia using both k-NN and weighted k-NN models, both with and without normalization.
6. Results Output: Results are written to an output file, including the predictions for Russia and details of the nearest neighbors used in the prediction process.
7. Model Evaluation: The program evaluates the accuracy of the k-NN model on a binary classification task, comparing the performance of models using both non-normalized and normalized data.</p>

# Install the Package 

## pip install scikit-learn 

<p> Can retrieve package from the link https://scikit-learn.org/stable/install.html</p>

