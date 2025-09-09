# Learning curves for different models

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

epochs = range(1, 101)
train_loss_cnn = 1.5 * np.exp(-epochs/30) + np.random.normal(0, 0.05, 100)
val_loss_cnn = 1.5 * np.exp(-epochs/25) + 0.2 + np.random.normal(0, 0.05, 100)

train_loss_transformer = 1.8 * np.exp(-epochs/40) + np.random.normal(0, 0.05, 100)
val_loss_transformer = 1.8 * np.exp(-epochs/35) + 0.15 + np.random.normal(0, 0.05, 100)

# Prepare data
df_learning = pd.DataFrame({
    'Epoch': list(epochs) * 4,
    'Loss': np.concatenate([train_loss_cnn, val_loss_cnn, train_loss_transformer, val_loss_transformer]),
    'Model': ['CNN'] * 200 + ['Transformer'] * 200,
    'Type': ['Training'] * 100 + ['Validation'] * 100 + ['Training'] * 100 + ['Validation'] * 100
})

plt.figure(figsize=(12, 8))
ax = sns.lineplot(data=df_learning, x='Epoch', y='Loss', hue='Model', style='Type', 
                  markers=True, dashes=False, palette='tab10', linewidth=2.5)
plt.title('Learning Curves Comparison', pad=20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(title='Model & Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()