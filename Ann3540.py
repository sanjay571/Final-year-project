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

# Specify the path to your CSV file in the c3540 directory
# CSV file contains test patterns for c3540 circuit in c3540 project
csv_path = r"C:\Users\medis\OneDrive\Documents\final project\c3540\c3540_test_patterns.csv"

# Read test patterns from CSV in the c3540 directory
try:
    df = pd.read_csv(csv_path)
    # 265 patterns, 60-bit inputs for c3540 circuit in c3540 project
    test_patterns = df['Input'].tolist()
    # 265 responses, 26-bit outputs for c3540 circuit in c3540 project (for reference)
    fault_free_responses = df['Output'].tolist()

    # Verify the number of patterns (should be 265 for c3540 in c3540 project)
    if len(test_patterns) != 265:
        raise ValueError(f"Expected 265 patterns, but found {len(test_patterns)} patterns in the CSV for c3540 in c3540.")

    # Ensure all patterns are 60 bits for c3540 circuit in c3540 project
    # (they already are, but included for robustness)
    def normalize_pattern(pattern, target_length=60):
        if len(pattern) > target_length:
            return pattern[:target_length]
        elif len(pattern) < target_length:
            return pattern + '0' * (target_length - len(pattern))
        return pattern

    test_patterns_fixed = [normalize_pattern(seq) for seq in test_patterns]

    # Convert to NumPy array for c3540 circuit data in c3540 project
    X = np.array([list(map(int, list(seq))) for seq in test_patterns_fixed]).astype(np.float32)

    # Split Data for c3540 circuit in c3540 project
    X_train, X_test = train_test_split(
        X, test_size=0.2, random_state=42  # 212 train, 53 test (20% of 265 = 53)
    )

    print("Training samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])

    # Build Improved Autoencoder Model for c3540 circuit in c3540 project
    input_dim = X_train.shape[1]  # 60 bits
    encoding_dim = 16  # Kept the same, but can be adjusted if needed for c3540 in c3540

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

    # Train Autoencoder for c3540 circuit in c3540 project
    history = autoencoder.fit(
        X_train, X_train, 
        epochs=100,
        batch_size=32,
        verbose=1,
        validation_split=0.2
    )

    # Plot Training and Validation Loss for c3540 circuit in c3540 project
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training and Validation Loss for c3540 in c3540')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Extract Encoded Features for c3540 circuit in c3540 project
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    # Compute Reconstruction Error and Fault Mask for c3540 circuit in c3540 project
    reconstructed_train = autoencoder.predict(X_train)
    mse_train = np.mean(np.power(X_train - reconstructed_train, 2), axis=1)
    threshold = np.mean(mse_train) + 1.5 * np.std(mse_train)
    fault_mask_train = (mse_train > threshold).astype(int)
    num_faults_train = np.sum(fault_mask_train)
    print(f"Number of faults detected in training data (reconstruction error) for c3540 in c3540: {num_faults_train}")

    reconstructed_test = autoencoder.predict(X_test)
    mse_test = np.mean(np.power(X_test - reconstructed_test, 2), axis=1)
    fault_mask_test = (mse_test > threshold).astype(int)
    num_faults_test = np.sum(fault_mask_test)
    print(f"Number of faults detected in test data (reconstruction error) for c3540 in c3540: {num_faults_test}")

    # Train Random Forest with Adjusted Parameters for c3540 circuit in c3540 project
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf_classifier.fit(X_train_encoded, fault_mask_train)

    # Evaluate Random Forest Accuracy and Confusion Matrix for c3540 circuit in c3540 project
    y_pred_train = rf_classifier.predict(X_train_encoded)
    train_accuracy = accuracy_score(fault_mask_train, y_pred_train)
    print(f"\nRandom Forest Training Accuracy (on fault mask) for c3540 in c3540: {train_accuracy:.2f}")
    cm_train = confusion_matrix(fault_mask_train, y_pred_train)
    print("\nConfusion Matrix (Training Data) for c3540 in c3540:")
    print(cm_train)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='coolwarm', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix (Training Data) for c3540 in c3540')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    y_pred_test = rf_classifier.predict(X_test_encoded)
    test_accuracy = accuracy_score(fault_mask_test, y_pred_test)
    print(f"Random Forest Testing Accuracy (on fault mask) for c3540 in c3540: {test_accuracy:.2f}")
    print(classification_report(fault_mask_test, y_pred_test, zero_division=0))
    cm_test = confusion_matrix(fault_mask_test, y_pred_test)
    print("\nConfusion Matrix (Test Data) for c3540 in c3540:")
    print(cm_test)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='coolwarm', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix (Test Data) for c3540 in c3540')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Plot Training and Testing Accuracy for c3540 circuit in c3540 project
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(['Training'], [train_accuracy], color='blue')
    plt.ylim(0, 1)
    plt.title('Training Accuracy for c3540 in c3540')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.bar(['Testing'], [test_accuracy], color='green')
    plt.ylim(0, 1)
    plt.title('Testing Accuracy for c3540 in c3540')
    plt.tight_layout()
    plt.show()

    # Reconstruct All 265 Patterns and Print Separately for c3540 in c3540 project
    reconstructed_all = autoencoder.predict(X)
    reconstructed_all_binary = (reconstructed_all > 0.5).astype(int)  # Adjusted to 0.5 for better binary reconstruction

    print("\nOriginal 265 Test Patterns for c3540 in c3540:")
    for i in range(len(X)):
        original = ''.join(map(str, X[i].astype(int)))
        print(f"Pattern {i+1}: {original}")

    print("\nReconstructed 265 Test Patterns for c3540 in c3540:")
    for i in range(len(X)):
        reconstructed = ''.join(map(str, reconstructed_all_binary[i]))
        print(f"Pattern {i+1}: {reconstructed}")

    # Save the Model in the specified c3540 directory for c3540 circuit
    model_save_path = r"C:\Users\medis\OneDrive\Documents\final project\c3540\RandomForest_fault_classifier_c3540.pkl"
    joblib.dump(rf_classifier, model_save_path)

except FileNotFoundError:
    print(f"Error: The file '{csv_path}' was not found. Please ensure the CSV file exists at the specified path in c3540.")
except Exception as e:
    print(f"An error occurred: {str(e)}")