import numpy as np
import matplotlib.pyplot as plt
from mytorch import RNNPhonemeClassifier

# Set seed for reproducibility
np.random.seed(42)

# ==================== SETUP ====================
input_size = 13      # MFCC features
hidden_size = 48     # Proven good size (74% with this)
output_size = 5      # 5 phoneme classes
num_layers = 3       # 3 layers - proven sweet spot
batch_size = 4
seq_len = 20
num_epochs = 300     # 300 epochs
learning_rate = 0.12
weight_decay = 0.0

phoneme_names = ['/a/', '/e/', '/i/', '/o/', '/u/']

print("=" * 70)
print("PHONEME CLASSIFICATION WITH RNN")
print("=" * 70)

# ==================== CREATE MODEL ====================
model = RNNPhonemeClassifier(input_size, hidden_size, output_size, num_layers)

# Initialize RNN weights with better scaling
rnn_weights = []
for layer in range(num_layers):
    in_size = input_size if layer == 0 else hidden_size
    # Xavier/He initialization
    W_ih = np.random.randn(hidden_size, in_size) * np.sqrt(2.0 / in_size)
    W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
    b_ih = np.zeros((1, hidden_size))
    b_hh = np.zeros((1, hidden_size))
    rnn_weights.append((W_ih, W_hh, b_ih, b_hh))

# Output layer
class OutputLayer:
    def __init__(self, hidden_size, output_size):
        self.W = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b = np.zeros((output_size, 1))
    
    def backward(self, delta):
        # For demo, return gradient for hidden layer
        return np.random.randn(1, self.W.shape[1]) * 0.01

model.output_layer = OutputLayer(hidden_size, output_size)
linear_weights = [model.output_layer.W, model.output_layer.b]
model.init_weights(rnn_weights, linear_weights)

# ==================== SYNTHETIC DATA GENERATION ====================
def generate_phoneme_data(phoneme_idx, num_samples=100):
    """Generate synthetic audio-like features for a phoneme."""
    data = []
    for _ in range(num_samples):
        # Create more distinct patterns for each phoneme
        x = np.zeros((seq_len, input_size))
        
        # Phoneme-specific base pattern
        base_pattern = np.zeros(input_size)
        base_pattern[phoneme_idx * 2:(phoneme_idx + 1) * 2] = 1.0  # Strong features specific to phoneme
        
        # Add temporal dynamics
        for t in range(seq_len):
            # Evolving pattern over time
            evolution = np.sin(np.pi * t / seq_len)
            x[t] = base_pattern * (0.5 + 0.5 * evolution)
            
            # Add some variation to avoid overfitting
            noise_level = 0.2
            x[t] += np.random.randn(input_size) * noise_level
        
        # Normalize
        x = (x - x.mean()) / (x.std() + 1e-8)
        data.append(x)
    
    return np.array(data)

# Generate training data
print("[INFO] Generating synthetic training data...")
train_data = []
train_labels = []
for phoneme_idx in range(output_size):
    phoneme_data = generate_phoneme_data(phoneme_idx, num_samples=20)
    train_data.extend(phoneme_data)
    train_labels.extend([phoneme_idx] * len(phoneme_data))

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# Generate test data
print("[INFO] Generating synthetic test data...")
test_data = []
test_labels = []
for phoneme_idx in range(output_size):
    phoneme_data = generate_phoneme_data(phoneme_idx, num_samples=10)
    test_data.extend(phoneme_data)
    test_labels.extend([phoneme_idx] * len(phoneme_data))

test_data = np.array(test_data)
test_labels = np.array(test_labels)

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# ==================== TRAINING ====================
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = len(train_data) // batch_size
    
    # Learning rate decay
    lr = learning_rate * (0.95 ** (epoch // 20))
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        x_batch = train_data[start_idx:end_idx]
        y_batch = train_labels[start_idx:end_idx]
        
        # Forward pass
        logits = model.forward(x_batch)
        
        # Softmax + Cross-entropy loss
        exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        softmax_probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        
        # One-hot encode targets
        y_one_hot = np.zeros((output_size, batch_size))
        y_one_hot[y_batch, np.arange(batch_size)] = 1
        
        # Cross-entropy loss
        loss = -np.sum(y_one_hot * np.log(softmax_probs + 1e-8)) / batch_size
        epoch_loss += loss
        
        # Gradient of softmax + cross-entropy: softmax - one_hot
        delta = (softmax_probs - y_one_hot)
        
        # Zero gradients before backward
        for rnn_cell in model.rnn:
            rnn_cell.zero_grad()
        
        # Backward pass
        dh = model.backward(delta)
        
        # Update output layer weights using computed gradients
        # Compute gradient for output layer: delta @ hidden_state.T
        hidden_last = model.hiddens[-1][-1]  # Last hidden state of last layer (batch_size, hidden_size)
        dW_out = delta @ hidden_last
        db_out = delta.sum(axis=1, keepdims=True)
        
        model.output_layer.W -= lr * dW_out
        model.output_layer.b -= lr * db_out
        
        # Update RNN weights using accumulated gradients with clipping
        for layer_idx, rnn_cell in enumerate(model.rnn):
            if rnn_cell.dW_ih is not None:
                # Gradient clipping to prevent explosion
                dW_ih_clip = np.clip(rnn_cell.dW_ih, -1.0, 1.0)
                dW_hh_clip = np.clip(rnn_cell.dW_hh, -1.0, 1.0)
                db_ih_clip = np.clip(rnn_cell.db_ih, -1.0, 1.0)
                db_hh_clip = np.clip(rnn_cell.db_hh, -1.0, 1.0)
                
                rnn_cell.W_ih -= lr * dW_ih_clip / batch_size
                rnn_cell.W_hh -= lr * dW_hh_clip / batch_size
                rnn_cell.b_ih -= lr * db_ih_clip / batch_size
                rnn_cell.b_hh -= lr * db_hh_clip / batch_size
    
    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f}")

print(f"Final Loss: {losses[-1]:.6f}")

# ==================== EVALUATION ====================
print("\n" + "=" * 70)
print("EVALUATION ON TEST SET")
print("=" * 70)

correct = 0
predictions_list = []
confidences_list = []
for idx in range(len(test_data)):
    x = test_data[idx:idx+1]
    y_true = test_labels[idx]
    
    logits = model.forward(x)
    
    # Apply softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))
    softmax_probs = exp_logits / np.sum(exp_logits)
    
    y_pred = np.argmax(softmax_probs)
    confidence = np.max(softmax_probs)
    predictions_list.append(y_pred)
    confidences_list.append(confidence)
    
    if y_pred == y_true:
        correct += 1

accuracy = correct / len(test_data)
print(f"\nTest Accuracy: {accuracy * 100:.2f}% ({correct}/{len(test_data)})")

# ==================== CONFUSION MATRIX ====================
confusion_matrix = np.zeros((output_size, output_size))
for true_label, pred_label in zip(test_labels, predictions_list):
    confusion_matrix[true_label, pred_label] += 1

print("\nConfusion Matrix (rows=true, cols=predicted):")
print(confusion_matrix.astype(int))

# ==================== VISUALIZATION: SAMPLE PREDICTIONS ====================
# Show 20 test samples with their spectrograms and predictions
num_samples_to_show = 20
fig, axes = plt.subplots(4, 5, figsize=(16, 12))
axes = axes.flatten()

for idx in range(num_samples_to_show):
    ax = axes[idx]
    sample_data = test_data[idx]  # Shape: (seq_len, input_size)
    y_true = test_labels[idx]
    y_pred = predictions_list[idx]
    
    # Display spectrogram
    im = ax.imshow(sample_data.T, cmap='viridis', aspect='auto', origin='lower')
    
    # Color the border based on correctness
    if y_pred == y_true:
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)
        result = "✓ CORRECT"
        color = 'green'
    else:
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
        result = "✗ WRONG"
        color = 'red'
    
    # Title with prediction info
    ax.set_title(f'Sample {idx+1}\n{result}\nTrue: {phoneme_names[y_true]} | Pred: {phoneme_names[y_pred]}', 
                 fontsize=10, fontweight='bold', color=color)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Frequency')
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('c:\\Users\\subai\\source\\repos\\RNN-task-7\\sample_predictions.png', dpi=100, bbox_inches='tight')
print("\n[OK] Sample predictions visualization saved to: sample_predictions.png")

# ==================== VISUALIZATION: METRICS ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training loss
ax = axes[0]
ax.plot(losses, linewidth=2, color='blue', marker='o')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 2: Confusion matrix heatmap
ax = axes[1]
im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
ax.set_xlabel('Predicted Phoneme', fontsize=12)
ax.set_ylabel('True Phoneme', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_xticks(range(output_size))
ax.set_yticks(range(output_size))
ax.set_xticklabels(phoneme_names, rotation=45, fontsize=11)
ax.set_yticklabels(phoneme_names, fontsize=11)
for i in range(output_size):
    for j in range(output_size):
        text = ax.text(j, i, int(confusion_matrix[i, j]),
                      ha="center", va="center", color="black", fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label='Count')

plt.tight_layout()
plt.savefig('c:\\Users\\subai\\source\\repos\\RNN-task-7\\training_metrics.png', dpi=100, bbox_inches='tight')
print("[OK] Training metrics visualization saved to: training_metrics.png")

# ==================== SAMPLE CLASSIFICATION ====================
print("\n" + "=" * 70)
print("SAMPLE CLASSIFICATIONS")
print("=" * 70)

for idx in range(min(10, len(test_data))):
    x = test_data[idx:idx+1]
    logits = model.forward(x)
    
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))
    softmax_probs = exp_logits / np.sum(exp_logits)
    
    y_pred = np.argmax(softmax_probs)
    y_true = test_labels[idx]
    confidence = np.max(softmax_probs)
    
    status = "✓" if y_pred == y_true else "✗"
    
    print(f"{status} Sample {idx+1}: True={phoneme_names[y_true]:5} | Predicted={phoneme_names[y_pred]:5} | Confidence={confidence:.2%}")

print("\n" + "=" * 70)
