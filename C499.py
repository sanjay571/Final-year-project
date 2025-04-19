import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ✅ Step 1: Define Test Patterns (57 patterns)
test_patterns = [
    "000010110010000000000100100000000000000000000101100100000000010010000000",
    "1110010000000100000000000000011000000010111100100000001000000000000000110",
    "0000000000000000000000000100000000010000000000000000000000000000000000000",
    "0000000000000000000000000000010000100000000000000000000000000000000000000",
    "0000000000000000000000110010000000000000000000000000000000001100110000",
    "0000000001000000000000000000000000000001000000000000000000000000000000000",
    "0000000000000000000010000000000001000000100000000000000000001000000000000",
    "0000000000000000000000000000001000000000000000000000000000000000000000000",
    "00000000000000000000000001000000000100001000000000000000000000000100000",
    "000000000000000000110000000000100000000000000000000000000011000000000011",
    "0000100000001100000000000000000000000000000001100000011000000000000000000",
    "000000010010000000001011001000000000000000000000100100000000101100100000",
    "000010110010000000000010100000000000000000000101100100000000001010000000",
    "010000000100000000100000011000000000000000100000001000000101000001100000",
    "1011100010010001000000000000011000000010110111000100100010000000000000110",
    "0000000000001000000000000000000000001000100000000000010000000000000000000",
    "000000000000000011000000000010000000000000000000000000000110000000001100",
    "0010000000000011000000000000000000000000000110000000000110000000000000000",
    "0000000000000000000000000010000000000000000000000000000000000000000000000",
    "0000000000000110101110001001000100100000100000000000001101011100010010001",
    "0010000001100000010000000100000000000000010100000011000000100000001000000",
    "000010110010000000000001001000000000000000000101100100000000000100100000",
    "0000000000000010000000000000000000000000000000000000000000000000000000000",
    "000000000000000000001100000010000000000000000000000000000000110000001100",
    "0011000000000010000000000000000000000000000110000000000110000000000000000",
    "0000000001000000000000000000000000000001100000000010000000000000000000000",
    "0010001010100010000000000000000000000000000100010001000100000000000000000",
    "000010110010000000000001010000000000000000000101100100000000000101000000",
    "0000001000110000000000000000000000000000000000011001100000000000000000000",
    "0000100000000000000000000000000000000100000000000000000000000000000000000",
    "0000100010000000000010000000000000000000000001000100000000001000000000000",
    "0000000000000110111001000000010000100000100000000000001101110010000000100",
    "000000000000000000000010001100000000000000000000000000000000001100110000",
    "0000000000000000001000000000000000000000000000000000000000000000000000000",
    "0000000000000000000010000000000001000000000000000000000000000000000000000",
    "000010110010000000000010010000000000000000000101100100000000001001000000",
    "1100000000000000000000000000100000000000011000000000000000000100000001000",
    "0010000010100000010000000100000000000000001100000101000000100000001000000",
    "0000000000000000000001000000000000000010000000000000000000000000000000000",
    "0100000001000000001000001010000000000000001000000010000000110000010100000",
    "0000000000000000000000000000100010000000000000000000000000000000000000000",
    "000001001000000000001011001000000000000000000010010000000000101100100000",
    "000000000000000000100010101000100000000000000000000000000010001000100010",
    "000000000000000000000010000000110000000000000000000000000000001100000011",
    "0000110000001000000000000000000000000000000001100000011000000000000000000",
    "0000001000000000000000000000000000000000000000000000000000000000000000000",
    "000000010100000000001011001000000000000000000000101000000000101100100000",
    "0000000000000000000001000000000000000010100000000000000000000100000000000",
    "0000000000001000000000000000000000001000000000000000000000000000000000000",
    "000000101000000000001011001000000000000000000001010000000000101100100000",
    "0000100000000000000000000000000000000100100001000000000000000000000000000",
    "0000001100000010000000000000000000000000000000011000000110000000000000000",
    "000000000000000000000000000010001000000010000000000000000000000000001000",
    "0000000000100000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000100000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000010000100000100000000000000000000000000000100",
    "0000000000010000000000000000000000000000000000000000000000000000000000000",
]

# Normalize all patterns to 73 bits
def normalize_pattern(pattern, target_length=73):
    if len(pattern) > target_length:
        return pattern[:target_length]
    elif len(pattern) < target_length:
        return pattern + '0' * (target_length - len(pattern))
    return pattern

test_patterns_fixed = [normalize_pattern(seq) for seq in test_patterns]

# Convert to NumPy array and reshape for 1D convolution (samples, timesteps, channels)
X = np.array([list(map(int, list(seq))) for seq in test_patterns_fixed]).astype(np.float32)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Shape: (57, 73, 1)

# Synthetic labels (fixed with seed; replace with real labels if available)
y = np.random.RandomState(42).randint(0, 2, size=(len(test_patterns),))

# ✅ Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# ✅ Step 3: Build Convolutional Autoencoder Model
input_shape = (73, 1)  # 73 timesteps, 1 channel

input_layer = Input(shape=input_shape)

# Encoder
encoded = Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_layer)
encoded = MaxPooling1D(pool_size=2)(encoded)  # Downsample to (36, 32)
encoded = Dropout(0.2)(encoded)
encoded = Conv1D(16, kernel_size=3, activation='relu', padding='same')(encoded)
encoded = MaxPooling1D(pool_size=2)(encoded)  # Downsample to (18, 16)
encoded = Dropout(0.2)(encoded)

# Flatten for latent representation
encoded_flat = Flatten()(encoded)  # Shape: (18 * 16) = 288
latent_dim = 32  # Reduce to a smaller latent space
encoded_dense = Dense(latent_dim, activation='relu')(encoded_flat)

# Decoder
decoded_dense = Dense(18 * 16, activation='relu')(encoded_dense)  # Expand back to 288
decoded_reshape = Reshape((18, 16))(decoded_dense)  # Reshape to (18, 16)
decoded = Conv1D(16, kernel_size=3, activation='relu', padding='same')(decoded_reshape)
decoded = UpSampling1D(size=2)(decoded)  # Upsample to (36, 16)
decoded = Dropout(0.2)(decoded)
decoded = Conv1D(32, kernel_size=3, activation='relu', padding='same')(decoded)
decoded = UpSampling1D(size=2)(decoded)  # Upsample to (72, 32)
decoded = Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')(decoded)  # Output (72, 1)

# Adjust output to match input shape (73, 1) by padding
from tensorflow.keras.layers import ZeroPadding1D
decoded = ZeroPadding1D(padding=(0, 1))(decoded)  # Pad 1 timestep to get (73, 1)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy')

# ✅ Step 4: Train Autoencoder
history = autoencoder.fit(
    X_train, X_train, 
    epochs=500, 
    batch_size=32, 
    verbose=1, 
    validation_split=0.2
)

# ✅ Step 4.1: Plot Training and Validation Loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Convolutional Autoencoder Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# ✅ Step 5: Extract Encoded Features
encoder = Model(inputs=autoencoder.input, outputs=encoded_dense)  # Latent representation (32-dim)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# ✅ Step 6: Compute Reconstruction Error and Fault Mask
reconstructed_train = autoencoder.predict(X_train)
mse_train = np.mean(np.power(X_train - reconstructed_train, 2), axis=(1, 2))  # Average over timesteps and channels
threshold = np.mean(mse_train) + 1.5 * np.std(mse_train)
fault_mask_train = (mse_train > threshold).astype(int)
num_faults_train = np.sum(fault_mask_train)
print(f"Number of faults detected in training data (reconstruction error): {num_faults_train}")

reconstructed_test = autoencoder.predict(X_test)
mse_test = np.mean(np.power(X_test - reconstructed_test, 2), axis=(1, 2))
fault_mask_test = (mse_test > threshold).astype(int)
num_faults_test = np.sum(fault_mask_test)
print(f"Number of faults detected in test data (reconstruction error): {num_faults_test}")

# ✅ Step 7: Train Random Forest
rf_classifier = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    min_samples_split=5, 
    min_samples_leaf=2, 
    random_state=42
)
rf_classifier.fit(X_train_encoded, fault_mask_train)

# ✅ Step 8: Evaluate Random Forest Accuracy and Confusion Matrix
# Training
y_pred_train = rf_classifier.predict(X_train_encoded)
train_accuracy = accuracy_score(fault_mask_train, y_pred_train)
print(f"\n✅ Random Forest Training Accuracy (on fault mask): {train_accuracy:.2f}")
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

# Testing
y_pred_test = rf_classifier.predict(X_test_encoded)
test_accuracy = accuracy_score(fault_mask_test, y_pred_test)
print(f"✅ Random Forest Testing Accuracy (on fault mask): {test_accuracy:.2f}")
print(classification_report(fault_mask_test, y_pred_test, zero_division=1))
cm_test = confusion_matrix(fault_mask_test, y_pred_test)
print("\nConfusion Matrix (Test Data):")
print(cm_test)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='coolwarm', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'],
            annot_kws={"size": 30} )
plt.title('Confusion Matrix (Test Data)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ✅ Step 8.1: Plot Training and Testing Accuracy
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

# ✅ Step 9: Reconstruct All 57 Patterns and Print Separately
reconstructed_all = autoencoder.predict(X)
reconstructed_all_binary = (reconstructed_all > 0.3).astype(int).reshape(reconstructed_all.shape[0], 73)

print("\nOriginal 57 Test Patterns:")
for i in range(len(X)):
    original = ''.join(map(str, X[i].astype(int).flatten()))
    print(f"Pattern {i+1}: {original}")

print("\nReconstructed 57 Test Patterns:")
for i in range(len(X)):
    reconstructed = ''.join(map(str, reconstructed_all_binary[i]))
    print(f"Pattern {i+1}: {reconstructed}")

# ✅ Step 10: Save the Model
joblib.dump(rf_classifier, "RandomForest_fault_classifier.pkl")