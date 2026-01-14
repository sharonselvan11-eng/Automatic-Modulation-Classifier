import sys
# Fix for Windows Emoji/Encoding errors
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# 1. GENERATE DATA (BPSK, QPSK, 16-QAM)
# ==========================================
print("📡 Generating 3-Class Dataset...")
np.random.seed(42)
n_samples = 1000
noise_std = 0.25  # Slightly lower noise to make BPSK visible

# Class 0: BPSK (2 points at -1, 1 on I-axis, 0 on Q-axis)
bpsk_i = np.random.choice([-1, 1], n_samples) + np.random.normal(0, noise_std, n_samples)
bpsk_q = np.random.normal(0, noise_std, n_samples) # Q is roughly 0
y_bpsk = np.zeros(n_samples)

# Class 1: QPSK (4 points)
qpsk_i = np.random.choice([-1, 1], n_samples) + np.random.normal(0, noise_std, n_samples)
qpsk_q = np.random.choice([-1, 1], n_samples) + np.random.normal(0, noise_std, n_samples)
y_qpsk = np.ones(n_samples)

# Class 2: 16-QAM (16 points)
qam_vals = [-3, -1, 1, 3]
qam_i = np.random.choice(qam_vals, n_samples) + np.random.normal(0, noise_std, n_samples)
qam_q = np.random.choice(qam_vals, n_samples) + np.random.normal(0, noise_std, n_samples)
y_qam = np.full(n_samples, 2)

# Combine
X = np.vstack((np.column_stack((bpsk_i, bpsk_q)), 
               np.column_stack((qpsk_i, qpsk_q)), 
               np.column_stack((qam_i, qam_q))))
y = np.concatenate((y_bpsk, y_qpsk, y_qam))

# ==========================================
# 2. TRAIN MODEL
# ==========================================
print("🧠 Training Neural Network...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# We use a slightly larger network now because the problem is harder
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=1)
clf.fit(X_scaled, y)

# Report Accuracy
acc = accuracy_score(y, clf.predict(X_scaled))
print(f"✅ Training Accuracy: {acc*100:.2f}%")

# ==========================================
# 3. VISUALIZE DECISION BOUNDARY
# ==========================================
print("🎨 Drawing the Map (this takes a moment)...")

# Create a meshgrid covering the chart area
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
h = 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict every point on the grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(scaler.transform(grid_points))
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(12, 10))
# The Contour Map
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis') # viridis handles 3 colors nicely

# The Scatter Points
plt.scatter(bpsk_i, bpsk_q, c='purple', label='BPSK (Class 0)', s=10, alpha=0.6)
plt.scatter(qpsk_i, qpsk_q, c='teal',   label='QPSK (Class 1)', s=10, alpha=0.6)
plt.scatter(qam_i, qam_q,   c='yellow', label='16-QAM (Class 2)', s=10, alpha=0.6)

plt.title(f'3-Way Modulation Classification\nAccuracy: {acc*100:.1f}%')
plt.xlabel('In-Phase (I)')
plt.ylabel('Quadrature (Q)')
plt.legend()
plt.grid(False)

print("Displaying plot...")
plt.show()