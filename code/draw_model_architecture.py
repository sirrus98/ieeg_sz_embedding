import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Create figure - horizontal layout
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.set_xlim(0, 22)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme
color_input = '#E8F4F8'
color_conv = '#B3D9E6'
color_pos_enc = '#C8E6C9'
color_spatial = '#FFE0B2'
color_temporal = '#E1BEE7'
color_output = '#FFCDD2'

# Helper function to draw boxes
def draw_box(ax, x, y, width, height, text, color, fontsize=10, fontweight='normal', 
             alpha=1.0, edgewidth=2):
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle="round,pad=0.15", 
                         edgecolor='black', facecolor=color, 
                         linewidth=edgewidth, alpha=alpha)
    ax.add_patch(box)
    # Split text by \n and draw each line
    lines = text.split('\n')
    if len(lines) == 1:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
                fontweight=fontweight)
    else:
        for i, line in enumerate(lines):
            offset = (len(lines)-1)/2 - i
            ax.text(x, y + offset*0.25, line, ha='center', va='center', 
                   fontsize=fontsize, fontweight=fontweight)
    return box

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, style='->', lw=2.5, color='black'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color=color, 
                           linewidth=lw, mutation_scale=25)
    ax.add_patch(arrow)
    return arrow

# Helper to draw a grid of small boxes (representing channels/frames)
def draw_grid(ax, x, y, rows, cols, cell_size, color, label=None):
    for i in range(rows):
        for j in range(cols):
            rect = Rectangle((x + j*cell_size, y - i*cell_size), 
                           cell_size*0.9, cell_size*0.9,
                           facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
    if label:
        ax.text(x + cols*cell_size/2, y + cell_size, label, 
               ha='center', va='bottom', fontsize=8, style='italic')

# Title
ax.text(11, 11, 'Model Architecture: Spatiotemporal iEEG Embedding', 
        ha='center', va='center', fontsize=14, fontweight='bold')

# Stage labels
stage_y = 10.2
ax.text(1.5, stage_y, 'Input', ha='center', fontsize=9, style='italic', color='gray')
ax.text(4.5, stage_y, 'Feature\nExtraction', ha='center', fontsize=9, style='italic', color='gray')
ax.text(8, stage_y, 'Positional\nEncoding', ha='center', fontsize=9, style='italic', color='gray')
ax.text(12, stage_y, 'Spatial Processing', ha='center', fontsize=9, style='italic', color='gray')
ax.text(16.5, stage_y, 'Temporal Processing', ha='center', fontsize=9, style='italic', color='gray')
ax.text(20.5, stage_y, 'Output', ha='center', fontsize=9, style='italic', color='gray')

# 1. Input - Multi-channel signal visualization
x_pos = 1.5
y_center = 6
draw_box(ax, x_pos, y_center, 1.8, 1.2, 'Multi-channel\niEEG Signal', 
         color_input, fontsize=10, fontweight='bold')

# Draw waveforms to represent channels
for i in range(4):
    y_wave = y_center + 0.4 - i*0.2
    x_wave = np.linspace(x_pos-0.6, x_pos+0.6, 20)
    y_vals = y_wave + 0.05*np.sin(x_wave*5 + i)
    ax.plot(x_wave, y_vals, 'b-', linewidth=1.5, alpha=0.7)

ax.text(x_pos, y_center-0.9, 'C channels\nT timepoints', 
        ha='center', fontsize=8, style='italic')

# Arrow to next stage
draw_arrow(ax, x_pos+0.9, y_center, x_pos+1.5, y_center)

# 2. Convolutional Feature Encoder
x_pos = 4.5
draw_box(ax, x_pos, y_center, 2.5, 2.5, '', color_conv, alpha=0.3, edgewidth=2.5)
ax.text(x_pos, y_center+1.5, 'Channel-wise Conv Stack', 
        ha='center', fontsize=10, fontweight='bold')

# Show per-channel processing
for i in range(3):
    y_conv = y_center + 0.6 - i*0.6
    # Input waveform
    draw_box(ax, x_pos-0.8, y_conv, 0.6, 0.3, f'Ch{i+1}', '#E3F2FD', fontsize=8)
    # Conv layers
    draw_box(ax, x_pos, y_conv, 0.8, 0.3, 'Conv1D\nStack', color_conv, fontsize=7)
    # Output features
    draw_box(ax, x_pos+0.8, y_conv, 0.6, 0.3, f'F×T', '#BBDEFB', fontsize=8)
    # Arrows
    draw_arrow(ax, x_pos-0.5, y_conv, x_pos-0.4, y_conv, lw=1)
    draw_arrow(ax, x_pos+0.4, y_conv, x_pos+0.5, y_conv, lw=1)

ax.text(x_pos, y_center-1.4, 'Independent processing\nper channel', 
        ha='center', fontsize=8, style='italic')

ax.text(x_pos, y_center-1.8, 'Output: C × T × F', 
        ha='center', fontsize=8, fontweight='bold', 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Arrow to next stage
draw_arrow(ax, x_pos+1.25, y_center, x_pos+2, y_center)

# 3. Spatiotemporal Positional Encoding
x_pos = 8
draw_box(ax, x_pos, y_center, 2.2, 2, 'Learnable\nPositional\nEncoding', 
         color_pos_enc, fontsize=10, fontweight='bold')

# Draw a grid representing spatial (channels) and temporal (frames)
grid_x = x_pos - 0.5
grid_y = y_center - 0.3
draw_grid(ax, grid_x, grid_y, 3, 4, 0.25, color_pos_enc)
ax.text(x_pos, grid_y - 0.85, 'Spatial (regions)', ha='center', fontsize=7)
ax.text(grid_x - 0.3, grid_y + 0.15, 'Temporal\n(frames)', ha='right', fontsize=7, rotation=0)

# Plus symbol for addition
ax.text(x_pos-0.9, y_center+0.7, '⊕', fontsize=28, ha='center', va='center', 
        fontweight='bold', color='#2E7D32')

# Arrow to next stage
draw_arrow(ax, x_pos+1.1, y_center, x_pos+1.8, y_center)

# 4. Spatial Transformer
x_pos = 12
draw_box(ax, x_pos, y_center, 3.2, 3.5, '', color_spatial, alpha=0.2, edgewidth=3)
ax.text(x_pos, y_center+2, 'Spatial Transformer', 
        ha='center', fontsize=11, fontweight='bold')

# Visualize spatial attention across channels
y_spatial = y_center + 0.8
ax.text(x_pos, y_spatial, 'Self-Attention', ha='center', fontsize=9, fontweight='bold')
ax.text(x_pos, y_spatial-0.3, 'across channels', ha='center', fontsize=8, style='italic')

# Draw channels attending to each other
y_ch = y_center - 0.2
channel_positions = [x_pos - 0.8, x_pos - 0.3, x_pos + 0.3, x_pos + 0.8]
for i, x_ch in enumerate(channel_positions):
    # Draw channel box
    rect = FancyBboxPatch((x_ch-0.15, y_ch-0.3), 0.3, 0.6,
                         boxstyle="round,pad=0.05", 
                         facecolor=color_spatial, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x_ch, y_ch, f'R{i+1}', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Draw attention connections
    for j, x_ch2 in enumerate(channel_positions):
        if i != j:
            ax.plot([x_ch, x_ch2], [y_ch-0.35, y_ch-0.35], 
                   'o-', color='orange', alpha=0.3, linewidth=1, markersize=2)

ax.text(x_pos, y_ch-0.7, 'Each region attends to all regions', 
        ha='center', fontsize=7, style='italic')

ax.text(x_pos, y_center-1.4, '× depth layers', ha='center', fontsize=8, style='italic')

ax.text(x_pos, y_center-1.8, 'Output: C × T × F', 
        ha='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Arrow to next stage
draw_arrow(ax, x_pos+1.6, y_center, x_pos+2.2, y_center)

# Reshape annotation
ax.text(x_pos+1.9, y_center+0.5, 'Reshape', ha='center', fontsize=7, 
        style='italic', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow'))

# 5. Temporal Transformer with CLS token
x_pos = 16.5
draw_box(ax, x_pos, y_center, 3.2, 3.5, '', color_temporal, alpha=0.2, edgewidth=3)
ax.text(x_pos, y_center+2, 'Temporal Transformer', 
        ha='center', fontsize=11, fontweight='bold')

# CLS token
y_temporal = y_center + 0.8
draw_box(ax, x_pos-1.2, y_temporal, 0.7, 0.5, '[CLS]', '#FFD700', 
         fontsize=9, fontweight='bold', edgewidth=2)
ax.text(x_pos-1.2, y_temporal-0.5, 'Prepended\n(learnable)', 
        ha='center', fontsize=6, style='italic')

# Arrow showing CLS concatenation
draw_arrow(ax, x_pos-0.85, y_temporal, x_pos-0.5, y_temporal, style='->', lw=1.5)

# Visualize temporal attention across frames
ax.text(x_pos+0.3, y_temporal, 'Self-Attention', ha='center', fontsize=9, fontweight='bold')
ax.text(x_pos+0.3, y_temporal-0.3, 'across time', ha='center', fontsize=8, style='italic')

# Draw frames attending to each other
y_frame = y_center - 0.2
frame_positions = [x_pos - 0.6, x_pos - 0.2, x_pos + 0.2, x_pos + 0.6, x_pos + 1.0]
for i, x_fr in enumerate(frame_positions):
    # Draw frame box
    label = 'CLS' if i == 0 else f'T{i}'
    fcolor = '#FFD700' if i == 0 else color_temporal
    rect = FancyBboxPatch((x_fr-0.12, y_frame-0.3), 0.24, 0.6,
                         boxstyle="round,pad=0.03", 
                         facecolor=fcolor, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x_fr, y_frame, label, ha='center', va='center', fontsize=6, fontweight='bold')
    
    # Draw attention connections
    for j, x_fr2 in enumerate(frame_positions):
        if i != j and abs(i-j) <= 2:  # Show some connections
            ax.plot([x_fr, x_fr2], [y_frame-0.35, y_frame-0.35], 
                   'o-', color='purple', alpha=0.3, linewidth=1, markersize=2)

ax.text(x_pos, y_frame-0.7, 'Each frame attends to all frames', 
        ha='center', fontsize=7, style='italic')

ax.text(x_pos, y_center-1.4, '× depth layers', ha='center', fontsize=8, style='italic')

ax.text(x_pos, y_center-1.8, 'Extract [CLS] token', 
        ha='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Arrow to next stage
draw_arrow(ax, x_pos+1.6, y_center, x_pos+2.2, y_center)

# 6. Output - SwAV Head
x_pos = 20.5
draw_box(ax, x_pos, y_center, 2, 2, '', color_output, alpha=0.3, edgewidth=2.5)
ax.text(x_pos, y_center+1.3, 'SwAV Head', 
        ha='center', fontsize=10, fontweight='bold')

# Show projection head structure
draw_box(ax, x_pos, y_center+0.5, 1.5, 0.4, 'Projection MLP', color_output, fontsize=8)
draw_arrow(ax, x_pos, y_center+0.3, x_pos, y_center-0.1, lw=1.5)
draw_box(ax, x_pos, y_center-0.3, 1.5, 0.4, 'L2 Normalize', color_output, fontsize=8)
draw_arrow(ax, x_pos, y_center-0.5, x_pos, y_center-1.0, lw=1.5)

# Final embedding
draw_box(ax, x_pos, y_center-1.3, 1.2, 0.4, 'Embedding', '#4CAF50', 
         fontsize=9, fontweight='bold', alpha=0.7)

# Add key insights box
insight_y = 2
insights = [
    "Key Design Choices:",
    "1. Per-channel convolution (independent feature extraction)",
    "2. Learnable positional encoding (spatial + temporal)",
    "3. Spatial attention: Regions communicate (which areas matter)",
    "4. Temporal attention: Frames communicate (when events occur)",
    "5. [CLS] token for global representation"
]

box_insight = FancyBboxPatch((0.3, insight_y-0.3), 8, 1.8,
                            boxstyle="round,pad=0.2", 
                            facecolor='#F5F5F5', edgecolor='black', 
                            linewidth=2, linestyle='--')
ax.add_patch(box_insight)

for i, text in enumerate(insights):
    weight = 'bold' if i == 0 else 'normal'
    size = 9 if i == 0 else 8
    ax.text(0.5, insight_y + 1.3 - i*0.3, text, ha='left', va='top', 
           fontsize=size, fontweight=weight)

plt.tight_layout()
plt.savefig('model_architecture_schematic.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
try:
    plt.savefig('model_architecture_schematic.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Saved model_architecture_schematic.png and .pdf")
except PermissionError:
    print("✓ Saved model_architecture_schematic.png (PDF file is open, skipped)")
plt.show()
