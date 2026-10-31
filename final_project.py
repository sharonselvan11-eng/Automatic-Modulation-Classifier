import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_signals(num_samples_per_class, snr_db):
    """
    Generates BPSK, QPSK, and 16-QAM signals with AWGN based on target SNR in dB.
    """
    X = []
    y = []
    
    # Convert SNR in dB to Noise Standard Deviation (sigma)
    # SNR_linear = 10^(SNR_db / 10) = Signal_Power / Noise_Variance
    # For normalized constellations, Signal Power is roughly 1.0 (average)
    # Therefore, Noise Variance (sigma^2) = 1 / SNR_linear -> sigma = 1 / sqrt(10^(SNR_db / 10))
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = np.sqrt(1.0 / (2.0 * snr_linear)) # Divided by 2 for I and Q channels

    for _ in range(num_samples_per_class):
        # 1. BPSK (2 points: -1, 1)
        bits_bpsk = np.random.choice([-1, 1])
        i_bpsk = bits_bpsk + np.random.normal(0, noise_std)
        q_bpsk = np.random.normal(0, noise_std)
        X.append([i_bpsk, q_bpsk])
        y.append(0) # Class 0: BPSK

        # 2. QPSK (4 points)
        bits_qpsk = np.random.choice([-1, 1], size=2)
        i_qpsk = (bits_qpsk[0] / np.sqrt(2)) + np.random.normal(0, noise_std)
        q_qpsk = (bits_qpsk[1] / np.sqrt(2)) + np.random.normal(0, noise_std)
        X.append([i_qpsk, q_qpsk])
        y.append(1) # Class 1: QPSK

        # 3. 16-QAM (16 points on a grid)
        bits_16qam = np.random.choice([-3, -1, 1, 3], size=2) / np.sqrt(10)
        i_16qam = bits_16qam[0] + np.random.normal(0, noise_std)
        q_16qam = bits_16qam[1] + np.random.normal(0, noise_std)
        X.append([i_16qam, q_16qam])
        y.append(2) # Class 2: 16-QAM

    return np.array(X), np.array(y)

print("Starting SNR Sweep Simulation (-10 dB to +20 dB)...")

# Define range of SNR values to test (from harsh noise to clean signal)
snr_range = range(-10, 21, 2)
accuracies = []

for snr in snr_range:
    # Generate training and testing data at this specific SNR level
    X, y = generate_signals(num_samples_per_class=1000, snr_db=snr)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MLP Classifier
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    
    # Predict and evaluate
    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc * 100)
    print(f"SNR: {snr:2d} dB -> Accuracy: {acc * 100:.2f}%")

# Plotting the Accuracy vs SNR Curve
plt.figure(figsize=(10, 6))
plt.plot(snr_range, accuracies, marker='o', linestyle='-', color='b', linewidth=2.5, markersize=6)
plt.title('ModulAI: Classification Accuracy vs. Signal-to-Noise Ratio (SNR)', fontsize=14, fontweight='bold')
plt.xlabel('SNR (dB) [Lower = More Noise, Higher = Cleaner Signal]', fontsize=12)
plt.ylabel('Classification Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(20, 105)

# Save the plot
output_filename = 'accuracy_vs_snr.png'
plt.savefig(output_filename, dpi=300)
print(f"\n Success! SNR sweep complete. Plot saved as '{output_filename}'.")
plt.show()