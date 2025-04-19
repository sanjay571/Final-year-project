import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import joblib
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Specify the path to your CSV file
csv_path = r"C:\Users\medis\OneDrive\Documents\final project\c432\c432_test_patterns.csv"

# Read test patterns from CSV
df = pd.read_csv(csv_path)
test_patterns = df['Input'].tolist()  # 63 patterns, 36-bit inputs
fault_free_responses = df['Output'].tolist()  # 63 responses, 7-bit outputs (for reference)

# Ensure all patterns are 36 bits (they already are, but included for robustness)
def normalize_pattern(pattern, target_length=36):
    if len(pattern) > target_length:
        return pattern[:target_length]
    elif len(pattern) < target_length:
        return pattern + '0' * (target_length - len(pattern))
    return pattern

test_patterns_fixed = [normalize_pattern(seq) for seq in test_patterns]

# Convert to NumPy array
X = np.array([list(map(int, list(seq))) for seq in test_patterns_fixed]).astype(np.float32)

# Split Data
X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# Build Improved Autoencoder Model
input_dim = X_train.shape[1]  # 36 bits
encoding_dim = 16  # Adjusted for smaller input dimension

input_layer = Input(shape=(input_dim,))
encoded1 = Dense(64, activation='relu')(input_layer)
encoded1 = Dropout(0.2)(encoded1)
encoded2 = Dense(32, activation='relu')(encoded1)
encoded2 = Dropout(0.2)(encoded2)
encoded3 = Dense(encoding_dim, activation='relu')(encoded2)  # Code layer
decoded1 = Dense(32, activation='relu')(encoded3)
decoded1 = Dropout(0.2)(decoded1)
decoded2 = Dense(64, activation='relu')(decoded1)
decoded = Dense(input_dim, activation='sigmoid')(decoded2)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy')

# Train Autoencoder
history = autoencoder.fit(
    X_train, X_train, 
    epochs=100,
    batch_size=32,
    verbose=1,
    validation_split=0.2
)

# Plot Training and Validation Loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Extract Encoded Features
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Compute Reconstruction Error and Fault Mask
reconstructed_train = autoencoder.predict(X_train)
mse_train = np.mean(np.power(X_train - reconstructed_train, 2), axis=1)
threshold = np.mean(mse_train) + 1.5 * np.std(mse_train)
fault_mask_train = (mse_train > threshold).astype(int)
num_faults_train = np.sum(fault_mask_train)
print(f"Number of faults detected in training data (reconstruction error): {num_faults_train}")

reconstructed_test = autoencoder.predict(X_test)
mse_test = np.mean(np.power(X_test - reconstructed_test, 2), axis=1)
fault_mask_test = (mse_test > threshold).astype(int)
num_faults_test = np.sum(fault_mask_test)
print(f"Number of faults detected in test data (reconstruction error): {num_faults_test}")

# Train Random Forest with Adjusted Parameters
rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_classifier.fit(X_train_encoded, fault_mask_train)

# Evaluate Random Forest Accuracy and Confusion Matrix
y_pred_train = rf_classifier.predict(X_train_encoded)
train_accuracy = accuracy_score(fault_mask_train, y_pred_train)
print(f"\nRandom Forest Training Accuracy (on fault mask): {train_accuracy:.2f}")
cm_train = confusion_matrix(fault_mask_train, y_pred_train)
print("\nConfusion Matrix (Training Data):")
print(cm_train)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='coolwarm', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix (Training Data)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

y_pred_test = rf_classifier.predict(X_test_encoded)
test_accuracy = accuracy_score(fault_mask_test, y_pred_test)
print(f"Random Forest Testing Accuracy (on fault mask): {test_accuracy:.2f}")
print(classification_report(fault_mask_test, y_pred_test, zero_division=1))
cm_test = confusion_matrix(fault_mask_test, y_pred_test)
print("\nConfusion Matrix (Test Data):")
print(cm_test)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='coolwarm', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix (Test Data)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot Training and Testing Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(['Training'], [train_accuracy], color='blue')
plt.ylim(0, 1)
plt.title('Training Accuracy')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.bar(['Testing'], [test_accuracy], color='green')
plt.ylim(0, 1)
plt.title('Testing Accuracy')
plt.tight_layout()
plt.show()

# Reconstruct All 63 Patterns and Print Separately
reconstructed_all = autoencoder.predict(X)
reconstructed_all_binary = (reconstructed_all > 0.5).astype(int)

print("\nOriginal 63 Test Patterns:")
for i in range(len(X)):
    original = ''.join(map(str, X[i].astype(int)))
    print(f"Pattern {i+1}: {original}")

print("\nReconstructed 63 Test Patterns:")
for i in range(len(X)):
    reconstructed = ''.join(map(str, reconstructed_all_binary[i]))
    print(f"Pattern {i+1}: {reconstructed}")

# Save the Model
joblib.dump(rf_classifier, "RandomForest_fault_classifier.pkl")