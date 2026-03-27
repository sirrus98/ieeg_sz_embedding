"""Script to build the remaining cells of the debug notebook"""
import json

# The notebook structure as Python code that will be inserted
cells_to_add = [
    # Cell 5: Section 2 header
    {
        "cell_type": "markdown",
        "source": "## 2. Signal Visualization by Label"
    },
    # Cell 6: Select examples
    {
        "cell_type": "code",
        "source": """# Select example windows for visualization
ictal_indices = np.where(all_labels == 1)[0]
interictal_indices = np.where(all_labels == 0)[0]

# Randomly select examples
np.random.seed(42)
n_examples = 5
ictal_examples = np.random.choice(ictal_indices, size=min(n_examples, len(ictal_indices)), replace=False)
interictal_examples = np.random.choice(interictal_indices, size=min(n_examples, len(interictal_indices)), replace=False)

# Time axis (10 seconds at 256 Hz)
fs = 256
time = np.arange(2560) / fs

print(f"Selected {len(ictal_examples)} ictal and {len(interictal_examples)} interictal examples")"""
    },
    # Cell 7: Plot ictal
    {
        "cell_type": "code",
        "source": """# Plot ictal examples
fig, axes = plt.subplots(n_examples, 1, figsize=(15, 3*n_examples))
if n_examples == 1:
    axes = [axes]

for i, idx in enumerate(ictal_examples):
    signals = all_signals[idx]
    n_ch = int(all_n_channels[idx])
    
    # Plot 3-5 random channels
    channels_to_plot = np.random.choice(n_ch, size=min(5, n_ch), replace=False)
    
    for ch in channels_to_plot:
        axes[i].plot(time, signals[ch], alpha=0.7, linewidth=0.8)
    
    axes[i].set_ylabel('Amplitude (normalized)')
    axes[i].set_title(f'Ictal Window {idx} - Patient {all_patient_ids[idx]} ({n_ch} channels, showing {len(channels_to_plot)})')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(0, 10)

axes[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('debug_ictal_signals.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Ictal signals plotted")"""
    },
    # Cell 8: Plot interictal
    {
        "cell_type": "code",
        "source": """# Plot interictal examples
fig, axes = plt.subplots(n_examples, 1, figsize=(15, 3*n_examples))
if n_examples == 1:
    axes = [axes]

for i, idx in enumerate(interictal_examples):
    signals = all_signals[idx]
    n_ch = int(all_n_channels[idx])
    
    # Plot 3-5 random channels
    channels_to_plot = np.random.choice(n_ch, size=min(5, n_ch), replace=False)
    
    for ch in channels_to_plot:
        axes[i].plot(time, signals[ch], alpha=0.7, linewidth=0.8)
    
    axes[i].set_ylabel('Amplitude (normalized)')
    axes[i].set_title(f'Interictal Window {idx} - Patient {all_patient_ids[idx]} ({n_ch} channels, showing {len(channels_to_plot)})')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(0, 10)

axes[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('debug_interictal_signals.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Interictal signals plotted")"""
    },
    # Cell 9: Overlay plots
    {
        "cell_type": "code",
        "source": """# Overlay comparison plot
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

# Ictal overlays
for idx in ictal_examples:
    signals = all_signals[idx]
    n_ch = int(all_n_channels[idx])
    ch = np.random.randint(0, n_ch)  # Random channel
    axes[0].plot(time, signals[ch], alpha=0.5, linewidth=1.0, color='red')

axes[0].set_ylabel('Amplitude (normalized)')
axes[0].set_title(f'Ictal Signals Overlay (n={len(ictal_examples)} windows, random channels)')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 10)

# Interictal overlays
for idx in interictal_examples:
    signals = all_signals[idx]
    n_ch = int(all_n_channels[idx])
    ch = np.random.randint(0, n_ch)  # Random channel
    axes[1].plot(time, signals[ch], alpha=0.5, linewidth=1.0, color='blue')

axes[1].set_ylabel('Amplitude (normalized)')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_title(f'Interictal Signals Overlay (n={len(interictal_examples)} windows, random channels)')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 10)

plt.tight_layout()
plt.savefig('debug_signals_overlay.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Overlay comparison plotted")"""
    }
]

print("Cells prepared for insertion")
print(f"Total cells to add: {len(cells_to_add)}")
for i, cell in enumerate(cells_to_add, start=5):
    print(f"  Cell {i}: {cell['cell_type']}")
