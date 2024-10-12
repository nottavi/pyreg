import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

def get_api_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Une erreur s'est produite lors de la requête: {e}")
        return None
    except json.JSONDecodeError:
        print("Erreur lors du décodage JSON. La réponse n'est peut-être pas au format JSON.")
        return None

load_dotenv()
api_key = os.getenv('API_KEY')
symbol = input("Entrez le symbole de l'action: ")
nb_of_days = int(input("Entrez le nombre de jours à analyser: "))

api_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
resultat = get_api_data(api_url)

if not resultat:
    print("Échec de la récupération des données.")
    exit()

time_series = resultat['Time Series (Daily)']

dates = []
prices = []

for date, values in list(time_series.items())[:nb_of_days]:
    dates.append(datetime.strptime(date, '%Y-%m-%d'))
    prices.append(float(values['4. close']))

dates = dates[::-1]
prices = np.array(prices[::-1])

# Convertir les dates en nombres pour la régression
dates_ordinal = [d.toordinal() for d in dates]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(dates_ordinal, prices)

# Create the regression line
line = slope * np.array(dates_ordinal) + intercept

# Calcul de l'écart-type des résidus
residuals = prices - line
std_dev = np.std(residuals)

# Calcul des lignes d'écart-type
line_plus_1std = line + std_dev
line_minus_1std = line - std_dev
line_plus_2std = line + 2*std_dev
line_minus_2std = line - 2*std_dev

# Définir les couleurs
color_price = 'blue'
color_regression = 'red'
color_1std = 'green'
color_2std = 'orange'

# Créer la figure et les axes
fig, ax = plt.subplots(figsize=(14, 7))

# Tracer la courbe des prix et la ligne de régression
ax.plot(dates, prices, color='blue', label='Prix de clôture')
ax.plot(dates, line, color='r', label=f'Ligne de régression (R² = {r_value**2:.4f})')

# Tracer les lignes d'écart-type
ax.plot(dates, line_plus_1std, color='green', linestyle='--', label='+1 écart-type')
ax.plot(dates, line_minus_1std, color='green', linestyle='--', label='-1 écart-type')
ax.plot(dates, line_plus_2std, color='orange', linestyle=':', label='+2 écarts-types')
ax.plot(dates, line_minus_2std, color='orange', linestyle=':', label='-2 écarts-types')

# Configurer les étiquettes et le titre
ax.set_xlabel('Date')
ax.set_ylabel('Prix de clôture (EUR)')
ax.set_title(f'{symbol} Analyse de régression du prix des actions ({nb_of_days} derniers jours)')

# Ajouter une légende
ax.legend()

# Formater l'axe des x pour afficher les dates correctement
fig.autofmt_xdate()

# Obtenir le dernier prix et la dernière valeur de la ligne de régression
last_price = prices[-1]
last_regression_value = line[-1]

# Ajouter des annotations pour le dernier prix et la dernière valeur de régression
ax.annotate(f'{last_price:.2f}', 
            xy=(dates[-1], last_price), 
            xytext=(10, 0), 
            textcoords='offset points', 
            ha='left', 
            va='center',
            color=color_price)

ax.annotate(f'{last_regression_value:.2f}', 
            xy=(dates[-1], last_regression_value), 
            xytext=(10, 0), 
            textcoords='offset points', 
            ha='left', 
            va='center',
            color=color_regression)

# Étendre les limites de l'axe x pour laisser de la place aux annotations
ax.set_xlim(right=dates[-1] + timedelta(days=int(nb_of_days * 0.05)))

# Ajuster la mise en page
plt.tight_layout()

# Sauvegarder le graphique
plt.savefig(f'{symbol}_regression_analysis_{nb_of_days}days.png')
plt.close()

# Afficher les statistiques
print(f"Pente: {slope:.6f}")
print(f"Ordonnée à l'origine: {intercept:.6f}")
print(f"R-carré: {r_value**2:.6f}")
print(f"Valeur P: {p_value:.6f}")
print(f"Erreur standard: {std_err:.6f}")
print(f"Écart-type des résidus: {std_dev:.6f}")

last_date = dates[-1]
predicted_price = slope * dates_ordinal[-1] + intercept
actual_price = prices[-1]
print(f"Prix prédit pour {last_date.strftime('%Y-%m-%d')}: {predicted_price:.2f}")
print(f"Prix réel pour {last_date.strftime('%Y-%m-%d')}: {actual_price:.2f}")
print(f"Différence: {actual_price - predicted_price:.2f}")