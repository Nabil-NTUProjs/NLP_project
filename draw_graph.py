import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Sample data for demonstration
experiments = [
    'layers=1_bs=64_lr=1e-05', 'layers=5_bs=32_lr=1e-05', 'layers=3_bs=32_lr=1e-05', 
    'layers=4_bs=64_lr=1e-05', 'layers=5_bs=64_lr=1e-05', 'layers=2_bs=64_lr=0.0001',
    'layers=4_bs=32_lr=1e-05', 'layers=3_bs=16_lr=1e-05', 'layers=1_bs=32_lr=0.0001',
    'layers=2_bs=64_lr=1e-05'
]
validation_accuracy = [0.758, 0.7486, 0.7402, 0.7383, 0.7383, 0.7373, 0.7355, 0.7345, 0.7336, 0.7326]
test_accuracy = [0.7486, 0.7317, 0.7439, 0.7514, 0.7411, 0.7289, 0.7345, 0.7411, 0.7326, 0.7289]
validation_loss = [0.0084, 0.0171, 0.0178, 0.0085, 0.0088, 0.0086, 0.0179, 0.0345, 0.0185, 0.0087]
test_loss = [0.0086, 0.0191, 0.0174, 0.0089, 0.009, 0.0089, 0.0193, 0.0354, 0.0188, 0.0089]
model_type = ['BiDeepRNN', 'BiDeepRNN', 'BiDeepRNN', 'BiDeepRNN', 'BiDeepRNN', 
              'DeepRNN', 'DeepRNN', 'DeepRNN', 'DeepRNN', 'BiDeepRNN']

# Plot setup
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar width and X-axis positions for each experiment
bar_width = 0.35
x = np.arange(len(experiments))

# Separate data by model type for coloring
bi_validation_acc = [val if model == 'BiDeepRNN' else 0 for val, model in zip(validation_accuracy, model_type)]
bi_test_acc = [test if model == 'BiDeepRNN' else 0 for test, model in zip(test_accuracy, model_type)]
deep_validation_acc = [val if model == 'DeepRNN' else 0 for val, model in zip(validation_accuracy, model_type)]
deep_test_acc = [test if model == 'DeepRNN' else 0 for test, model in zip(test_accuracy, model_type)]

# Bars for Validation Accuracy
ax1.bar(x - bar_width/2, bi_validation_acc, width=bar_width, color='#FF7F50', label='BiDeepRNN Validation Accuracy')
ax1.bar(x + bar_width/2, bi_test_acc, width=bar_width, color='#4682B4', label='BiDeepRNN Test Accuracy')
ax1.bar(x - bar_width/2, deep_validation_acc, width=bar_width, color='#FF4500', label='DeepRNN Validation Accuracy')
ax1.bar(x + bar_width/2, deep_test_acc, width=bar_width, color='#6A5ACD', label='DeepRNN Test Accuracy')

# Line plot for Validation and Test Loss
ax2 = ax1.twinx()
ax2.plot(x, validation_loss, color='orange', marker='o', linestyle='-', label='Validation Loss')
ax2.plot(x, test_loss, color='yellow', marker='^', linestyle='--', label='Test Loss')

# Axes labels and title
ax1.set_xlabel('Experiment')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
ax1.set_title('Top 10 Configs')

# Set x-ticks with experiment names
ax1.set_xticks(x)
ax1.set_xticklabels(experiments, rotation=45, ha="right")

# Custom legend
legend_elements = [
    Patch(facecolor='#FF7F50', label='BiDeepRNN Validation Accuracy'),
    Patch(facecolor='#4682B4', label='BiDeepRNN Test Accuracy'),
    Patch(facecolor='#FF4500', label='DeepRNN Validation Accuracy'),
    Patch(facecolor='#6A5ACD', label='DeepRNN Test Accuracy'),
    Line2D([0], [0], color='orange', marker='o', linestyle='-', label='Validation Loss'),
    Line2D([0], [0], color='yellow', marker='^', linestyle='--', label='Test Loss')
]
# Place legend outside the plot
ax1.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, -0.1))

# Display plot
plt.tight_layout()
plt.show()
