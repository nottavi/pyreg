import matplotlib.pyplot as plt
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import os


# Load and parse the JSON data
def get_api_data(url):
    try:
        # Envoyer une requête GET à l'URL de l'API
        response = requests.get(url)
        
        # Vérifier si la requête a réussi (code 200)
        response.raise_for_status()
        
        # Convertir la réponse en JSON
        data = response.json()
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"Une erreur s'est produite lors de la requête: {e}")
        return None
    except json.JSONDecodeError:
        print("Erreur lors du décodage JSON. La réponse n'est peut-être pas au format JSON.")
        return None

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la valeur de API_KEY
api_key = os.getenv('API_KEY')
symbol = input("Entrez le symbole de l'action: ")
api_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="+symbol+"&outputsize=full&apikey="+api_key;
resultat = get_api_data(api_url)

if resultat:
    print("Données récupérées avec succès:")
    print(json.dumps(resultat, indent=2))  # Affiche les données JSON de manière formatée
else:
    print("Échec de la récupération des données.")


# Extract the time series data
time_series = resultat['Time Series (Daily)']

# Convert dates to numerical values and extract closing prices
dates = []
prices = []

for date, values in list(time_series.items())[:365]:
    dates.append(datetime.strptime(date, '%Y-%m-%d').toordinal())
    prices.append(float(values['4. close']))

# Convert to numpy arrays and reverse the order (oldest to newest)
dates = np.array(dates)[::-1]
prices = np.array(prices)[::-1]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(dates, prices)

# Create the regression line
line = slope * dates + intercept


# Calcul de l'écart-type des résidus
residuals = prices - line
std_dev = np.std(residuals)

# Calcul des lignes d'écart-type
line_plus_1std = line + std_dev
line_minus_1std = line - std_dev
line_plus_2std = line + 2*std_dev
line_minus_2std = line - 2*std_dev

# Plot the data and regression line
plt.figure(figsize=(12, 6))
# plt.scatter(dates, prices, alpha=0.5)
plt.plot(dates, prices, color='blue', label='Prix de clôture')
plt.plot(dates, line, color='r', label=f'Regression line (R² = {r_value**2:.4f})')
plt.xlabel('Date')
plt.ylabel('Closing Price (EUR)')
plt.title(symbol + ' Stock Price Regression Analysis')
plt.legend()

# Tracer les lignes d'écart-type
plt.plot(dates, line_plus_1std, color='green', linestyle='--', label='+1 écart-type')
plt.plot(dates, line_minus_1std, color='green', linestyle='--', label='-1 écart-type')
plt.plot(dates, line_plus_2std, color='orange', linestyle=':', label='+2 écarts-types')
plt.plot(dates, line_minus_2std, color='orange', linestyle=':', label='-2 écarts-types')

# Obtenir le dernier prix et la dernière valeur de la ligne de régression
last_price = prices[-1]
last_regression_value = line[-1]

# Ajouter des annotations pour le dernier prix et la dernière valeur de régression
# plt.annotate(f'Prix actuel: {last_price:.2f}', 
#             xy=(dates[-1], last_price), 
#             xytext=(10, 0), 
#             textcoords='offset points', 
#             ha='left', 
#             va='center')

# plt.annotate(f'Valeur de régression: {last_regression_value:.2f}', 
#             xy=(date[-1], last_regression_value), 
#             xytext=(10, 0), 
#             textcoords='offset points', 
#             ha='left', 
#             va='center')

# Étendre les limites de l'axe x pour laisser de la place aux annotations
# ax.set_xlim(right=date[-1] + (date[-1] - date[0]) * 0.05)

# Format x-axis labels as dates
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: datetime.fromordinal(int(x)).strftime('%Y-%m-%d')))
plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

plt.tight_layout()
plt.savefig(symbol + '_regression.png')
plt.close()

# Calculate and print some statistics
print(f"Slope: {slope:.6f}")
print(f"Intercept: {intercept:.6f}")
print(f"R-squared: {r_value**2:.6f}")
print(f"P-value: {p_value:.6f}")
print(f"Standard Error: {std_err:.6f}")

# Calculate the predicted price for the last date in the dataset
last_date = dates[-1]
predicted_price = slope * last_date + intercept
actual_price = prices[-1]
print(f"Predicted price for {datetime.fromordinal(last_date).strftime('%Y-%m-%d')}: {predicted_price:.2f}")
print(f"Actual price for {datetime.fromordinal(last_date).strftime('%Y-%m-%d')}: {actual_price:.2f}")
print(f"Difference: {actual_price - predicted_price:.2f}")
