import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Function to generate fake LIWC embeddings
def generate_fake_data(class_name, num_datapoints, num_features=10):
    data = np.random.rand(num_datapoints, num_features)
    return pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(num_features)])

# Function to save fake data to CSV
def save_fake_data(class_names, num_datapoints_per_class, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for class_name in class_names:
        fake_data = generate_fake_data(class_name, num_datapoints_per_class)
        file_path = os.path.join(data_dir, f"{class_name}.csv")
        fake_data.to_csv(file_path, index=False)
    print("Fake data saved successfully!")

# Generate and save fake data
class_names = ["class1", "class2", "class3"]
num_datapoints_per_class = 100
data_dir = "fake_data"

save_fake_data(class_names, num_datapoints_per_class, data_dir)

class LIWCDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self, class_names):
        data = []
        labels = []
        for class_name in class_names:
            file_path = os.path.join(self.data_dir, f"{class_name}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                num_datapoints = min(len(df), self.n_datapoints_per_class)
                data.extend(df.iloc[:num_datapoints].values)
                labels.extend([class_name] * num_datapoints)
        return np.array(data), np.array(labels)

    def get_datasets(self, class_names, n_datapoints_per_class, test_size=0.2, val_size=0.1, random_state=None):
        self.n_datapoints_per_class = n_datapoints_per_class

        # Load data and labels
        data, labels = self.load_data(class_names)

        # Split data into train and test sets for each class
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify=labels, random_state=random_state)

        # Split train data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    data_dir = "fake_data"
    class_names = ["class1", "class2", "class3"]  # List of class names
    n_datapoints_per_class = 100  # Number of datapoints per class

    dataloader = LIWCDataset(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = dataloader.get_datasets(class_names, n_datapoints_per_class)

    print("Train data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)
    print("Test data shape:", X_test.shape)

if __name__ == "__main__":
    main()
