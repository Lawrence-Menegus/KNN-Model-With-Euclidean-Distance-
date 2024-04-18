# Lawrence Menegus 
# Machine Learning CPSC 429 
# Program Description: Create a KNN model using Euclidean distance
# It prints out all the correct Distances and data on a Results.txt 
# file and acculates the accuracy with the Noramlized data and Unnormalized data



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Read data from file
data = []
with open('CPI.txt', 'r') as file:
    for line in file:
        data.append(line.strip().split(','))

# Extract relevant columns (features and target)
features = np.array([list(map(float, row[1:-1])) for row in data])
target = np.array([float(row[-1]) for row in data])

# Euclidean distance calculation
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function for k-NN prediction
def knn_predict(query, features, target, k):
    distances = [euclidean_distance(query, feature) for feature in features]
    nearest_neighbors = np.argsort(distances)[:k]
    prediction = np.mean(target[nearest_neighbors])
    return prediction

# Function for weighted k-NN prediction
def weighted_knn_predict(query, features, target, k):
    distances = [euclidean_distance(query, feature) for feature in features]
    nearest_neighbors = np.argsort(distances)[:k]
    weights = 1 / (np.array(distances)[nearest_neighbors] ** 2)
    weighted_prediction = np.sum(target[nearest_neighbors] * weights) / np.sum(weights)
    return weighted_prediction

# Normalize data using range normalization
def normalize_data(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

# Normalize the features
normalized_features = normalize_data(features)

# Add Russian Information to the data 
russia_features = np.array([67.62, 31.68, 10.00, 3.87, 12.90])
normalized_russia_features = (russia_features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))

# Perform predictions Nearest Neighbor prediction 
cpi_3nn = knn_predict(russia_features, features, target, 3)
cpi_weighted_16nn = weighted_knn_predict(russia_features, features, target, 16)
cpi_3nn_normalized = knn_predict(normalized_russia_features, normalized_features, target, 3)
cpi_weighted_16nn_normalized = weighted_knn_predict(normalized_russia_features, normalized_features, target, 16)

# Create a list to store the country names with their respective CPI for 3-NN and 16-NN
cpi_3nn_results = []
cpi_weighted_16nn_results = []

# Calculate Euclidean distance and store country names along with CPI values
for i in range(len(data)):
    dist = euclidean_distance(russia_features, features[i])
    cpi_3nn_results.append((data[i][0], dist, target[i]))
    weight = 1 / (dist ** 2)
    cpi_weighted_16nn_results.append((data[i][0], dist, target[i], weight, target[i] * weight))

# Sort results based on CPI for 3-NN and 16-NN
cpi_3nn_results.sort(key=lambda x: x[2])
cpi_weighted_16nn_results.sort(key=lambda x: x[4])

# Sort 3-NN and 16-NN results by ascending Euclidean distance for normalized and non-normalized data
cpi_3nn_results.sort(key=lambda x: x[1])
cpi_weighted_16nn_results.sort(key=lambda x: x[1])

# Write the result into the new file 
with open('Results.txt', 'w') as output_file:
    output_file.write("#############################################\n")
    output_file.write("Program output\n")
    output_file.write("#############################################\n")
    output_file.write("Original data:\n\n")
    
    
    # Align columns for original data
    max_lengths = [max(len(item) for item in col) for col in zip(*data)]
    for row in data:
        for i, item in enumerate(row):
            output_file.write(f"{item:<{max_lengths[i] + 2}}")
        output_file.write('\n')
        
    # Print out for 3-NN not Normalized CPI 
    output_file.write("\nCountry      Euclid     CPI\n")
    for result in cpi_3nn_results:
        output_file.write(f"{result[0]:<12} {result[1]:<10.4f} {result[2]:.4f}\n")
    output_file.write(f"CPI for 3-NN: {cpi_3nn:.4f}\n\n")
    
    # Print out for 16-NN  Not Normalized CPI 
    output_file.write("Country      Euclid     CPI    Weight  W*CPI\n")
    for result in cpi_weighted_16nn_results:
        output_file.write(f"{result[0]:<12} {result[1]:<10.4f} {result[2]:.4f} {result[3]:.4f} {result[4]:.4f}\n")
    output_file.write(f"CPI for weighted 16-NN: {cpi_weighted_16nn:.4f}\n\n")
    
    # Print Out Normalized Data CPI
    output_file.write("Normalized data:\n\n")
    # Appending the last column (CPI) from the original data to normalized features
    normalized_features_with_cpi = np.hstack((normalized_features, target.reshape(-1, 1)))
    for i, row in enumerate(normalized_features_with_cpi):
        output_file.write(f"{data[i][0]:<12}")
        for val in row:
            output_file.write(f" {val:<10.4f}")
        output_file.write(f"\n")
        
    # Print out for 3-NN Normalized Data CPI 
    output_file.write("\nCountry      Euclid     CPI\n")
    for i in range(len(data)):
        output_file.write(f"{data[i][0]:<12} {euclidean_distance(normalized_russia_features, normalized_features[i]):<10.4f} {target[i]:.4f}\n")
    output_file.write(f"After normalization: CPI for 3-NN: {cpi_3nn_normalized:.4f}\n\n")

    # Print out for 16-NN Normalized Data CPI 
    output_file.write("Country      Euclid     CPI     Weight  W*CPI\n")
    for i in range(len(data)):
        dist = euclidean_distance(normalized_russia_features, normalized_features[i])
        weight = 1 / (dist ** 2)
        output_file.write(f"{data[i][0]:<12} {dist:<10.4f} {target[i]:.4f} {weight:.4f} {target[i] * weight:.4f}\n")
    output_file.write(f"After normalization: CPI for weighted 16-NN: {cpi_weighted_16nn_normalized:.4f}\n\n")


# Metrics for the KNN model: 
binary_target = (target > target.mean()).astype(int)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, binary_target, test_size=0.2, random_state=42)

# Normalize training and testing features
X_train_normalized = normalize_data(X_train)
X_test_normalized = normalize_data(X_test)

# Predict using the non-normalized data
predictions_non_normalized = [knn_predict(x, X_train, y_train, 3) for x in X_test]
predictions_non_normalized = np.round(predictions_non_normalized).astype(int)

# Predict using the normalized data
predictions_normalized = [knn_predict(x, X_train_normalized, y_train, 3) for x in X_test_normalized]
predictions_normalized = np.round(predictions_normalized).astype(int)

# Calculate accuracy
accuracy_non_normalized = accuracy_score(y_test, predictions_non_normalized)
accuracy_normalized = accuracy_score(y_test, predictions_normalized)


# Print out Accuracy 
print(f"Accuracy with non-normalized data: {accuracy_non_normalized * 100:.2f}%")
print(f"Accuracy with normalized data: {accuracy_normalized * 100:.2f}%")