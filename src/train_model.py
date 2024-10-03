# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:17:37 2024

@author: samvi
"""

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
    

def load_split_data(path):
    '''
    Loads the preprocessed dataset and splits into features and target

    Parameters
    ----------
    path : str
        filepath of preprocessed data.

    Returns
    -------
    X : pd.DataFrame
        Features.
    Y : pd.DataFrame
        Target.

    '''
    df = pd.read_csv(path)
    X = df.drop('isFraud', axis=1).values
    Y = df['isFraud'].values
    return X, Y

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def save_predictions(y_pred, path):
    pd.DataFrame(y_pred, columns=['Prediction']).to_csv(path, index=False)

def calc_metrics(Y, y_pred):
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
        'Confusion Matrix': [[tn, fp], [fn, tp]]}

def save_metrics(metrics, path):
    with open(path, 'w') as f:
        f.write("Classification Metrics:\n")
        for key, value in metrics.items():
            if key == 'Confusion Matrix':
                f.write(f"{key}:\n")
                f.write(f"[[{value[0][0]}, {value[0][1]}],\n")
                f.write(f" [{value[1][0]}, {value[1][1]}]]\n")
            else:
                f.write(f"{key}: {value:.2f}\n")

# Main execution
if __name__ == "__main__":
    path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\decision_tree_task\data\preprocessed_fraud_train.csv"
    X, Y = load_split_data(path)
    print("Data loaded. Shape of X:", X.shape, "Shape of Y:", Y.shape)

    # Train the model
    model = DecisionTreeClassifier(dmax=2, minsamp=2, w=10)
    print("Model initialized. Starting training...")
    model.fit(X, Y)
    print("Model training completed.")

    print("Making predictions...")
    y_pred = model.predict(X)
    print("Predictions made.")  

    print("Calculating metrics...")
    metrics = calc_metrics(Y, y_pred)
    print("Metrics:", metrics)
    
    print("Decision Tree Structure")
    model.print_tree()
    
    # Information about root and right and left child. 
    root_info = model.get_node_info([])
    print("\nRoot Node Info:", root_info)

    left_child_info = model.get_node_info(['L'])
    print("Left Child of Root Info:", left_child_info)

    right_child_info = model.get_node_info(['R'])
    print("Right Child of Root Info:", right_child_info)
    
    model_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\decision_tree_task\models\decision_tree_model_3.pkl"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    pred_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\decision_tree_task\results\decision_tree_model3_predictions.csv"
    save_predictions(y_pred, pred_path)
    print(f"Predictions saved to {pred_path}")

    metrics_path = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\decision_tree_task\results\model_3_results.txt"
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")

    print("Process completed.")

