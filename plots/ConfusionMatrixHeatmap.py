# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

# Sample confusion matrix data
true_labels = np.random.randint(0, 5, 1000)
pred_labels = np.random.randint(0, 5, 1000) + np.random.choice([-1, 0, 1], 1000, p=[0.1, 0.8, 0.1])

# Ensure labels stay within bounds
pred_labels = np.clip(pred_labels, 0, 4)

# Create confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
classes = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', pad=20)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()