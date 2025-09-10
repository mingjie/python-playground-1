# Ablation study results
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

components = ['Baseline', '+Attention', '+BatchNorm', '+Dropout', '+All Components']
accuracy = [0.75, 0.78, 0.82, 0.85, 0.89]
std_dev = [0.02, 0.015, 0.018, 0.012, 0.01]

df_ablation = pd.DataFrame({
    'Components': components,
    'Accuracy': accuracy,
    'Std_Error': std_dev
})

plt.figure(figsize=(12, 8))
ax = sns.barplot(data=df_ablation, x='Components', y='Accuracy', 
                 palette='RdYlBu', yerr=df_ablation['Std_Error'])
plt.title('Ablation Study Results', pad=20)
plt.ylabel('Accuracy')
plt.xlabel('Model Components')
plt.ylim(0.7, 0.95)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Add value labels
for i, (container, acc, std) in enumerate(zip(ax.containers[0], accuracy, std_dev)):
    ax.text(i, acc + std + 0.005, f'{acc:.3f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()