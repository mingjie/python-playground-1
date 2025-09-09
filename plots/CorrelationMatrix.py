# Feature correlation matrix
np.random.seed(42)
n_features = 8
feature_names = [f'Feature_{i}' for i in range(1, n_features + 1)]

# Create correlated data
data_corr = np.random.randn(1000, n_features)
# Add some correlations
data_corr[:, 1] = data_corr[:, 0] * 0.7 + np.random.randn(1000) * 0.3
data_corr[:, 2] = data_corr[:, 0] * 0.5 + data_corr[:, 1] * 0.3 + np.random.randn(1000) * 0.4
data_corr[:, 4] = -data_corr[:, 3] * 0.6 + np.random.randn(1000) * 0.4

corr_matrix = np.corrcoef(data_corr.T)
df_corr = pd.DataFrame(corr_matrix, columns=feature_names, index=feature_names)

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, mask=mask, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Feature Correlation Matrix', pad=20)
plt.tight_layout()
plt.show()