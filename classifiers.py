import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define functions for each classifier
def train_logistic_regression(X_train, y_train):
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    pipeline.fit(X_train, y_train)
    return pipeline

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf

def train_support_vector_machine(X_train, y_train):
    pipeline = make_pipeline(StandardScaler(), SVC())
    pipeline.fit(X_train, y_train)
    return pipeline

def train_neural_network(X_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

# Define a function to evaluate a classifier
def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=iris.target_names)
    return accuracy, precision, recall, f1, cm, cr

# Train and evaluate each classifier
classifiers = {
    'Logistic Regression': train_logistic_regression(X_train, y_train),
    'Decision Tree': train_decision_tree(X_train, y_train),
    'Random Forest': train_random_forest(X_train, y_train),
    'Support Vector Machine': train_support_vector_machine(X_train, y_train),
    'Neural Network': train_neural_network(X_train, y_train)
}

plt.figure(figsize=(15, 18))

for i, (name, clf) in enumerate(classifiers.items(), 1):
    plt.subplot(5, 2, i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(name)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    accuracy, precision, recall, f1, _, cr = evaluate_classifier(clf, X_test, y_test)
    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-score: {f1:.2f}")
    print("Classification Report:")
    print(cr)
    print()

plt.tight_layout()
plt.show()

