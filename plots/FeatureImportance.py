#
# 
#  
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Feature importance plot
features = [f'Feature_{i}' for i in range(1, 16)]
importance = np.abs(np.random.normal(0, 1, 15))
importance = importance / np.sum(importance)  # Normalize

df_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
df_importance = df_importance.sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(df_importance)))
bars = plt.barh(df_importance['Feature'], df_importance['Importance'], color=colors)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Analysis', pad=20)
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, df_importance['Importance'])):
    plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{value:.3f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.show()