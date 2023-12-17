# %%
# Sélectionner les variables d'intérêt
variables_interet = ['T', 'RH', 'CO(GT)', 'NOx(GT)']

# Sous-ensemble du DataFrame avec les variables d'intérêt
data_subset = qa[variables_interet]

# Graphiques de dispersion
sns.pairplot(data_subset)
plt.suptitle("Graphiques de dispersion entre T, RH, CO et NOx", y=1.02)
plt.show()

# Matrice de corrélation
correlation_matrix = data_subset.corr()

# Heatmap pour visualiser la corrélation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matrice de corrélation entre T, RH, CO et NOx")
plt.show()