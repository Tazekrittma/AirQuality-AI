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