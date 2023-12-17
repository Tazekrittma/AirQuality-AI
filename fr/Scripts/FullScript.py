# %%
# Importation des modules généraux
import numpy as np
import pandas as pd
from tqdm import tqdm

# Pour les graphiques
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import seaborn as sns

# Pour manipuler les types de chaînes temporelles
import datetime

# Pour l'analyse exploratoire des données (EDA)
#from pandas_profiling import ProfileReport

# Pour la construction du modèle d'apprentissage automatique
from sklearn.model_selection import train_test_split

# Différents régresseurs pour le modèle d'apprentissage automatique
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Pour l'évaluation du modèle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

# Pour l'optimisation de la recherche sur la grille des paramètres
from sklearn.model_selection import GridSearchCV

# Pour sauvegarder le modèle
import pickle

# Pour le tracé q-q
import scipy.stats as stats

# %% [markdown]
# Importent la data

# %%
qa = pd.read_csv("/content/AirQualityUCI.csv", sep = ";", decimal = ",")

# %%
qa.head()

# %% [markdown]
# Supprimons les colonnes 15 et 16 car ils contiennent des fausses données *unnamed*

# %%
qa.drop(['Unnamed: 15','Unnamed: 16'], axis=1, inplace=True, errors = 'ignore')

# %% [markdown]
# Remplaçons-200 qui représente un code pour des mesures fausses par nan

# %%
qa.replace(to_replace = -200, value = np.nan, inplace = True)

# %%
qa.head()

# %% [markdown]
# La colonne 4 contient trop de valeurs -200 que on a remplace par nan, on vas supprimer cette colonne

# %%
qa.drop('NMHC(GT)', axis=1, inplace=True, errors = 'ignore')
qa=qa.dropna()

# %%
qa.head()

# %% [markdown]
# Temp

# %%

#Création d'un objet format temp
qa['DateTime'] = (qa.Date) + ' ' + (qa.Time)
qa.DateTime = qa.DateTime.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H.%M.%S'))

# %%
#intégrons le nouveau fomrat de date en remplaçant l'ancien

qa['Weekday'] = qa['DateTime'].dt.day_name()
qa['Month'] = qa['DateTime'].dt.month_name()
qa['Hour'] = qa['DateTime'].dt.hour
qa['Date'] = pd.to_datetime(qa['Date'], format='%d/%m/%Y')
qa.drop('Time', axis=1, inplace=True, errors = 'ignore')

# %%
qa.head()

# %%
#Etude de la dataset
qa.describe()

# %% [markdown]
# orginazons selon jours heurs mois pour les futures etudes

# %%
month_df_list = []
day_df_list   = []
hour_df_list  = []

months = ['January','February','March', 'April', 'May','June',
          'July', 'August', 'September', 'October', 'November', 'December']

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for month in months:
    temp_df = qa.loc[(qa['Month'] == month)]
    month_df_list.append(temp_df)

for day in days:
    temp_df = qa.loc[(qa['Weekday'] == day)]
    day_df_list.append(temp_df)

for hour in range(24):
    temp_df = qa.loc[(qa['Hour'] == hour)]
    hour_df_list.append(temp_df)

# %% [markdown]
# une fonction pour la vizualisation

# %%
def df_time_plotter(df_list, time_unit, y_col):

    months = ['January','February','March', 'April', 'May','June',
              'July', 'August', 'September', 'October', 'November', 'December']

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    if time_unit == 'M':
        nRows = 3
        nCols = 4
        n_iter = len(months)
    elif time_unit == 'D':
        nRows = 2
        nCols = 4
        n_iter = len(days)
    elif time_unit == 'H':
        nRows = 4
        nCols = 6
        n_iter = 24
    else:
        print('time_unit doit etre un string M,D, or H')
        return 0

    fig, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize = (40,30))
    axs = axs.ravel()
    for i in range(n_iter):
        data = df_list[i]
        ax = axs[i]
        data.plot(kind ='scatter', x = 'DateTime', y= y_col , ax = ax, fontsize = 24)
        ax.set_ylabel('Pollutant Concentration',fontsize=30)
        ax.set_xlabel('')
        if time_unit == 'M':
            ax.set_title(y_col + ' ' + months[i],  size=40) # Title
        elif time_unit == 'D':
            ax.set_title(y_col + ' ' + days[i],  size=40) # Title
        else:
             ax.set_title(y_col + ' ' + str(i),  size=40) # Title
        ax.tick_params(labelrotation=60)


    # espacement entres les subplots
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.5)
    plt.show()

# %%
df_time_plotter(month_df_list,'M','PT08.S1(CO)')

# %% [markdown]
# un formatage basant sur la date

# %%
qa = qa[['Date','Month', 'Weekday','DateTime', 'Hour', 'CO(GT)','PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)',
         'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']]
qa.head()

# %% [markdown]
# quelques modifications

# %% [markdown]
# # **Entraînement des modèles**

# %% [markdown]
# **Etude du polluant CO**

# %%


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


# %% [markdown]
# **Analyse:** selon les résultas R-squared (R²) les meilleurs modèles pour l'analyse de CO sont:
# 1. Foret aleatoir  0.920
# 2. Ada Boost 0.917
# 3. Régression linéaire 0.916

# %% [markdown]
# **Etude du polluant NOX**

# %%


# Sélectionner les caractéristiques (variables indépendantes) et la variable cible
features = ['PT08.S1(CO)','CO(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
cible =  'NOx(GT)'

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


# %% [markdown]
# **Analyse:** *selon R-squared (R²) lesmeilleurs models pour l'analyse de NOX sont:*
# 
# 1. MLP  0.914
# 2. Gradien boosting 0.902
# 3. K-Neighbors 0.888

# %%


# %% [markdown]
# **Etude du polluant PT08.S5(O3)**

# %%


# Sélectionner les caractéristiques (variables indépendantes) et la variable cible
features = ['PT08.S1(CO)','CO(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)','NOx(GT)' , 'T', 'RH', 'AH']
cible =  'PT08.S5(O3)'

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


# %% [markdown]
# **Analyse:** *selon R-squared (R²) lesmeilleurs models pour l'analyse de NOX sont:*
# 
#  1. MLP  0.913
#  2. Gradien boosting 0.907
#  3. K-Neighbors 0.905

# %% [markdown]
# ***Etude de l'influence de la température (T), de l'humidité relative (RH), et de la concentration initiale de monoxyde de carbone (CO) et d'oxyde d'azote (NOx)***

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

# %% [markdown]
# **Analyse**
# 
# Les niveaux d'oxyde d'azote (NOx) peuvent augmenter avec la
# température (T) due à des réactions chimiques favorisées. L'humidité relative (RH) peut également influencer, parfois inhiber, la formation de NOx en modifiant les conditions de combustion.
# 
# 
# Une élévation de la température (T) peut augmenter les émissions de monoxyde de carbone (CO) dans certains processus de combustion. L'humidité relative (RH) peut influencer la combustion, modérant ou exacerbant la production de CO en fonction des conditions.

# %% [markdown]
# # **Conclusion:**
# 
# Cette analyse démontre que la température et l'humidité relative ont un impact significatif sur les concentrations de NOx et CO. Les modèles, notamment MLP et Gradient Boosting, présentent des performances élevées pour la prédiction des niveaux de NOx. Pour le CO, les modèles RandomForest et AdaBoost montrent une précision remarquable. Ces résultats soulignent l'importance de considérer les conditions environnementales dans la modélisation des polluants atmosphériques. Les R² élevés indiquent une bonne adéquation des modèles aux données, renforçant leur fiabilité pour la prédiction des niveaux de pollution. Ces conclusions fournissent des informations cruciales pour la compréhension et la gestion des émissions atmosphériques, contribuant ainsi à la qualité de l'air et à la protection de l'environnement.

# %% [markdown]
# BY M.TAZEKRITT
# 
# Apache 2.0 License


