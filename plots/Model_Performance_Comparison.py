import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Sample data for model comparison
models = ['CNN', 'RNN', 'Transformer', 'BERT', 'GPT']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
data = np.random.rand(len(models), len(metrics)) * 0.3 + 0.7  # Random data between 0.7-1.0

df_model = pd.DataFrame(data, columns=metrics, index=models)
df_model = df_model.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
df_model.columns = ['Model', 'Metric', 'Score']

# Create the plot
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=df_model, x='Model', y='Score', hue='Metric', palette='viridis')
plt.title('Model Performance Comparison Across Different Metrics', pad=20)
plt.ylabel('Performance Score')
plt.xlabel('Models')
plt.ylim(0.6, 1.0)
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=10)

plt.tight_layout()
plt.show()