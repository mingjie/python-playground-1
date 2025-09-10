# Distribution of predictions for different classes
np.random.seed(42)
class_a = np.random.normal(0.8, 0.15, 1000)
class_b = np.random.normal(0.6, 0.12, 1000)
class_c = np.random.normal(0.4, 0.18, 1000)

df_dist = pd.DataFrame({
    'Prediction_Score': np.concatenate([class_a, class_b, class_c]),
    'Class': ['Class A'] * 1000 + ['Class B'] * 1000 + ['Class C'] * 1000
})

plt.figure(figsize=(12, 8))
ax = sns.histplot(data=df_dist, x='Prediction_Score', hue='Class', 
                  bins=50, alpha=0.7, kde=True, palette='Set1')
plt.title('Distribution of Prediction Scores by Class', pad=20)
plt.xlabel('Prediction Score')
plt.ylabel('Frequency')
plt.legend(title='Classes')

# Add mean lines
means = [class_a.mean(), class_b.mean(), class_c.mean()]
colors = ['red', 'blue', 'green']
for mean, color in zip(means, colors):
    plt.axvline(mean, color=color, linestyle='--', alpha=0.8, linewidth=2)

plt.tight_layout()
plt.show()