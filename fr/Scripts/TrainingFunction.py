# Sélectionner les caractéristiques (variables indépendantes) et la variable cible
features = ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
cible = 'CO(GT)'

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(qa[features], qa[cible], test_size=0.2, random_state=42)

# Initialiser différents modèles de régression
lr = LinearRegression()
hr = HuberRegressor(epsilon=1.15, max_iter=1000)
rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
kn = KNeighborsRegressor()
ab = AdaBoostRegressor()
sv = SVR()
dt = DecisionTreeRegressor(max_features='auto', max_depth=3, random_state=42)
nn = MLPRegressor(hidden_layer_sizes=(500,), solver='adam', learning_rate_init=1e-2, max_iter=500)

# Liste des modèles
modeles = [(lr, 'Régression linéaire'),
           (hr, 'Régression de Huber'),
           (rf, 'Forêt aléatoire'),
           (gb, 'Gradient Boosting'),
           (kn, 'K-Neighbors'),
           (ab, 'Ada Boost'),
           (sv, 'SVR'),
           (dt, 'Arbre de décision'),
           (nn, 'MLP')]

# Boucle pour ajuster et évaluer chaque modèle
for modele, nom_modele in modeles:
    # Entraîner le modèle
    modele.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    predictions = modele.predict(X_test)

    # Évaluer le modèle
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Afficher les résultats
    print(f'Résultats pour {nom_modele}:')
    print(f'Mean Squared Error (Erreur quadratique moyenne): {mse}')
    print(f'R-squared (R²): {r2}\n')

    # Tracer le graphique des valeurs réelles par rapport aux valeurs prédites
    plt.scatter(y_test, predictions)
    plt.xlabel(f'{cible} réel')
    plt.ylabel(f'{cible} prédit')
    plt.title(f'{cible} réel vs prédit - {nom_modele}')
    plt.show()