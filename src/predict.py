# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 04:35:37 2024

@author: samvi
"""

import argparse
import numpy as np
import pandas as pd
import pickle

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, dmax=None, minsamp=2, w=1.0):
        self.dmax = dmax
        self.minsamp = minsamp
        self.root = None
        self.w = w

    def fit(self, X, Y):
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, Y)

    def _grow_tree(self, X, Y, d=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(Y))
        
        # Stopping condition - tree depth, not enough data to split, or pure node        
        if (self.dmax is not None and d >= self.dmax) or n_samples < self.minsamp or n_classes == 1:
            leaf_value = self._most_common_label(Y)
            return Node(value=leaf_value)

        bfeature, bthreshold, bgain = self._best_split(X, Y)

        if bgain == 0:
            leaf_value = self._most_common_label(Y)
            return Node(value=leaf_value)

        left_n, right_n = self._split(X[:, bfeature], bthreshold)

        left = self._grow_tree(X[left_n], Y[left_n], d+1)
        right = self._grow_tree(X[right_n], Y[right_n], d+1)
        
        #Recursively build tree
        return Node(bfeature, bthreshold, left, right)

    def _best_split(self, X, Y):
        # Find the best split
        bgain = 0
        bfeature, bthreshold = None, None

        for feature in range(self.n_features):
            vals = X[:, feature]
            unq_vals = np.unique(vals.round(decimals=6))
            
            # 2 unique values => data column is a binary variable, can employ any threshold [0,1]
            if len(unq_vals) == 2:
                threshold = 0.5
                gain = self._information_gain(Y, vals, threshold)
                if gain > bgain:
                    bgain = gain
                    bfeature = feature
                    bthreshold = threshold
            else:
                # Finding optimal threshold for continuous variable by splitting at various points on the range of values based on gini
                thresholds = np.percentile(vals, [10,20,30,40,50,60,70,80,90])
                for threshold in thresholds:
                    gain = self._information_gain(Y, vals, threshold)
                    if gain > bgain:
                        bgain = gain
                        bfeature = feature
                        bthreshold = threshold

        return bfeature, bthreshold, bgain

    def _information_gain(self, Y, vals, t):
        parent_gini = self._gini(Y)
        left_n, right_n = self._split(vals, t)
        
        if len(left_n) == 0 or len(right_n) == 0:
            return 0
        
        n = len(Y)
        numleft, numright = len(left_n), len(right_n)
        gini_left, gini_right = self._gini(Y[left_n]), self._gini(Y[right_n])
        children_gini = (numleft/n)*gini_left + (numright/n)*gini_right
        
        return parent_gini - children_gini

    def _split(self, vals, t):
        left_n = np.argwhere(vals <= t).flatten()
        right_n = np.argwhere(vals > t).flatten()
        return left_n, right_n

    def _gini(self, Y):
        _, counts = np.unique(Y, return_counts=True)
        p = counts/len(Y)
        wp = p.copy()
        # Use a weight for gini calculation, since dataset is skewed towards IsFraud == 0
        if len(wp)>1:
            wp[1] *= self.w #  Increase weight of fraud class
        wp /= np.sum(wp) # Normalize weights
        return 1 - np.sum(wp**2)

    def _most_common_label(self, Y):
        # In order to determine prediction in leaf nodes
        return np.bincount(Y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def print_tree(self, node=None, depth=0):
       if node is None:
           node = self.root

       if node.value is not None:
           print("  " * depth + f"Predict {node.value}")
       else:
           print("  " * depth + f"X[{node.feature}] <= {node.threshold}")
           print("  " * (depth + 1) + "True:")
           self.print_tree(node.left, depth + 2)
           print("  " * (depth + 1) + "False:")
           self.print_tree(node.right, depth + 2)

    def get_node_info(self, node_path):
        node = self.root
        for direction in node_path:
            if direction == 'L':
                node = node.left
            elif direction == 'R':
                node = node.right
            if node is None:
                return None
        return {
            'feature': node.feature,
            'threshold': node.threshold,
            'value': node.value
        }

def load_data(path):
    """
    Loads data from a CSV file
    """
    return pd.read_csv(path)

def drop_missing_observations(df):
    """
    Drops missing observations(NaN values) from the dataset to ensure no missing value errors are thrown 
    """
    return df.dropna()

def preprocess_data(df):
    """
    Intital processing of the dataset before the models are trained - Handling missing values, creating features from categorical variables
    """
    # Create binary variable for Transfer or Cash Out
    df['is_transfer_or_cashout'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
    
    # Create binary variable if the full amount of the Origin old balance is the amount 
    df['is_full_amount'] = (df['amount'] == df['oldbalanceOrg']).astype(int)

    # Select features for the model
    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                'is_transfer_or_cashout', 'is_full_amount']
    
    return df[features]

def load_model(model_path):
    """
    Loads the trained model
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def calc_metrics(Y, y_pred):
    """
    Calculates evaluation metrics
    """
    tn = np.sum((Y==0) & (y_pred==0))
    fp = np.sum((Y==0) & (y_pred==1))
    fn = np.sum((Y==1) & (y_pred==0))
    tp = np.sum((Y==1) & (y_pred==1))
    
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0
    f1_score = 2 * (precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Confusion Matrix': [[tn, fp], [fn, tp]]
    }

def save_metrics(metrics, path):
    """
    Saves the metrics to a text file in the required format
    """
    with open(path, 'w') as f:
        f.write("Classification Metrics:\n")
        for key, value in metrics.items():
            if key == 'Confusion Matrix':
                f.write(f"{key}:\n")
                f.write(f"[[{value[0][0]}, {value[0][1]}],\n")
                f.write(f" [{value[1][0]}, {value[1][1]}]]\n")
            else:
                f.write(f"{key}: {value:.2f}\n")

def save_predictions(y_pred, path):
    """
    Saves the predictions to a CSV file
    """
    pd.DataFrame(y_pred, columns=['Prediction']).to_csv(path, index=False, header=False)

def main():
    parser = argparse.ArgumentParser(description="Predict using trained decision tree model")
    parser.add_argument("--model_path", required=True, help="Path to the saved model file")
    parser.add_argument("--data_path", required=True, help="Path to the data CSV file")
    parser.add_argument("--metrics_output_path", required=True, help="Path where the evaluation metrics will be saved")
    parser.add_argument("--predictions_output_path", required=True, help="Path where the predictions will be saved")
    args = parser.parse_args()
    # Load and preprocess data
    df = load_data(args.data_path)
    df = drop_missing_observations(df)
    X = preprocess_data(df)
    
    # Check if 'isFraud' column exists for metric calculation
    if 'isFraud' in df.columns:
        Y = df['isFraud'].values
    else:
        Y = None
    
    print("Data Loaded")

    # Load model and make predictions
    model = load_model(args.model_path)
    y_pred = model.predict(X.values)

    # Save predictions
    save_predictions(y_pred, args.predictions_output_path)
    
    print(f"Predictions saved to {args.predictions_output_path}")
    # Calculate and save metrics if true labels are available
    if Y is not None:
        metrics = calc_metrics(Y, y_pred)
        save_metrics(metrics, args.metrics_output_path)
    else:
        print("No 'isFraud' column found in the data. Skipping metric calculation.")
    print(f"Metrics saved to {args.metrics_output_path}")

if __name__ == "__main__":
    main()