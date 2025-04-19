import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import joblib
import pandas as pd
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directory for plots
output_dir = r"C:\Users\medis\OneDrive\Documents\final project\c6288\plots"
os.makedirs(output_dir, exist_ok=True)

# c6288 test patterns
test_patterns = [
    "11111111101000001011111111100000",
    "00000000000001100000000000000011",
    "11111111011111110111111111111111",
    "00000011111001111111111111110111",
    "00000000111111111111111110111111",
    "01111111111001111110111111111111",
    "00000000111100111111111111111011",
    "00000000000111010111111111111111",
    "00000001111111110000000011111111",
    "10111000000000001100100000000000",
    "11110100000000001011110000000000",
    "00000110011111111111111011111111",
    "00000000000000011000000000000001",
    "00111111100011111111011111111111",
    "11000000000000001110000000000000",
    "00000000000001111111111111111101",
    "00000011100111111111111111011000",
    "10001111100000001111000010000000",
    "11111110000000001111111100000000",
    "11111111110100001011111111110000",
    "00000000000001110110101010111110",
    "00000000001110011111111111111101",
    "11111111111111001111111111111110",
    "11111111000000001111111110000000",
    "00000000111111111111111100000000",
    "11111100000000001111111000000000",
    "11100000000000001111000000000000",
    "11111111111010001011111111111000",
    "11111111111111101111111111111111",
    "00011110000111111111101111111111",
    "00001100001111111111110111111111",
    "11110000000000001111100000000000",
    "11111111111110111101111111111111",
    "11111111000000000111111100000000",
    "00000111110011111111111111101111"
]

fault_free_responses = [
    "11000000011001111111010000000000",
    "00000000000000000000000000010001",
    "01000000010000001011111101111111",
    "00000010000110001100001010011011",
    "00000000100000001000000010111111",
    "01001000000110111010111111100111",
    "00000000100011001111100010101110",
    "00000000000010010111111111101110",
    "00000000000000010000000101111111",
    "11100100010000000000000000000000",
    "11001100110100000000000000000000",
    "00000101100010111111101110111111",
    "00000000000000010000000000000001",
    "00100010011101111101011110001111",
    "10101000000000000000000000000000",
    "00000000000001001111111111100110",
    "00000010011000000000101011011000",
    "11111000011100000100000000000000",
    "10000001011111100000000000000000",
    "11000000001100111111110100000000",
    "00000000000000101101010110110101",
    "00000000001001101111111100101101",
    "10000000000000101111111111111000",
    "10000000101111111000000000000000",
    "00000000100000000111111100000000",
    "10000010111110000000000000000000",
    "10010110000000000000000000000000",
    "11000000000110011111111101000000",
    "10000000000000010111111111111101",
    "00010001000111111110101000011111",
    "00001011111011111111011111011111",
    "10001011100000000000000000000000",
    "10100000000001010101111111111011",
    "01000000101111110000000000000000",
    "00000100001100000000001000110111"
]

# Data preparation
def normalize_pattern(pattern, target_length=32):
    if len(pattern) > target_length:
        return pattern[:target_length]
    elif len(pattern) < target_length:
        return pattern + '0' * (target_length - len(pattern))
    return pattern

test_patterns_fixed = [normalize_pattern(seq) for seq in test_patterns]
X = np.array([list(map(int, list(seq))) for seq in test_patterns_fixed]).astype(np.float32)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# Simplified Autoencoder without dropout, with regularization
input_dim = X_train.shape[1]  # 32
encoding_dim = 6  # Reduced further

input_layer = Input(shape=(input_dim,))
encoded = Dense(12, activation='relu', kernel_regularizer=l2(0.01))(input_layer)  # Reduced neurons
encoded = Dense(encoding_dim, activation='relu', kernel_regularizer=l2(0.01))(encoded)
decoded = Dense(12, activation='relu', kernel_regularizer=l2(0.01))(encoded)  # Reduced neurons
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.000005), loss='binary_crossentropy')

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# Train with early stopping
history = autoencoder.fit(
    X_train, X_train,
    epochs=200,  # Increased to allow early stopping to work effectively
    batch_size=32,  # Increased for smoother updates
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Plot 1: Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Autoencoder Training and Validation Loss for c6288', fontsize=14)
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_plot_c6288.png'))
plt.close()

# Encoder and feature extraction
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)  # Adjusted layer index
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Fault detection
reconstructed_train = autoencoder.predict(X_train)
mse_train = np.mean(np.power(X_train - reconstructed_train, 2), axis=1)
threshold = np.mean(mse_train) + 1.5 * np.std(mse_train)
fault_mask_train = (mse_train > threshold).astype(int)
num_faults_train = np.sum(fault_mask_train)
print(f"Number of faults detected in training data: {num_faults_train}")

reconstructed_test = autoencoder.predict(X_test)
mse_test = np.mean(np.power(X_test - reconstructed_test, 2), axis=1)
fault_mask_test = (mse_test > threshold).astype(int)
num_faults_test = np.sum(fault_mask_test)
print(f"Number of faults detected in test data: {num_faults_test}")

# Random Forest training
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_classifier.fit(X_train_encoded, fault_mask_train)

# Evaluate Random Forest
y_pred_train = rf_classifier.predict(X_train_encoded)
train_accuracy = accuracy_score(fault_mask_train, y_pred_train)
print(f"\nRandom Forest Training Accuracy: {train_accuracy:.2f}")
cm_train = confusion_matrix(fault_mask_train, y_pred_train)

# Plot 2: Training Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='coolwarm', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix (Training Data) for c6288', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cm_train_c6288.png'))
plt.close()

y_pred_test = rf_classifier.predict(X_test_encoded)
test_accuracy = accuracy_score(fault_mask_test, y_pred_test)
print(f"Random Forest Testing Accuracy: {test_accuracy:.2f}")
print(classification_report(fault_mask_test, y_pred_test, zero_division=0))
cm_test = confusion_matrix(fault_mask_test, y_pred_test)

# Plot 3: Testing Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='coolwarm', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'],
            annot_kws={"size": 30})
plt.title('Confusion Matrix (Test Data) for c6288', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cm_test_c6288.png'))
plt.close()

# Plot 4: Training and Testing Accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(['Training'], [train_accuracy], color='blue', width=0.5)
plt.ylim(0, 1)
plt.title('Training Accuracy for c6288', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.bar(['Testing'], [test_accuracy], color='green', width=0.5)
plt.ylim(0, 1)
plt.title('Testing Accuracy for c6288', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_plot_c6288.png'))
plt.close()

# Reconstruction output
reconstructed_all = autoencoder.predict(X)
reconstructed_all_binary = (reconstructed_all > 0.5).astype(int)

print("\nOriginal 35 Test Patterns for c6288:")
for i in range(len(X)):
    original = ''.join(map(str, X[i].astype(int)))
    print(f"Pattern {i+1}: {original}")

print("\nReconstructed 35 Test Patterns for c6288:")
for i in range(len(X)):
    reconstructed = ''.join(map(str, reconstructed_all_binary[i]))
    print(f"Pattern {i+1}: {reconstructed}")

# Save the model
model_save_path = r"C:\Users\medis\OneDrive\Documents\final project\c6288\RandomForest_fault_classifier_c6288.pkl"
joblib.dump(rf_classifier, model_save_path)

print(f"\nPlots have been saved to: {output_dir}")