from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import openpyxl
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import os

app = Flask(__name__)

# Usar backend no interactivo para matplotlib
plt.switch_backend('Agg')

# Definición de los activos
tickers = ['BTC-USD', 'ETH-USD', 'SAN.MC', 'BBVA.MC', 'BNP.PA', 'GS', 'JPM',
           'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSM', 'TCEHY',
           'GC=F', 'CL=F', 'SI=F', 'NG=F', 'AGG', 'BND', 'EMB', 'IGOV']

def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_std_dev

def generate_random_portfolios(returns, num_portfolios):
    num_assets = len(returns.columns)
    results = np.zeros((num_portfolios, 4))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return, portfolio_std_dev = portfolio_performance(weights, returns)
        portfolio_sharpe_ratio = portfolio_return / portfolio_std_dev
        results[i,:3] = [portfolio_return, portfolio_std_dev, portfolio_sharpe_ratio]
        weights_record.append(weights)

    return results, weights_record

def optimize_portfolio_for_volatility(returns, target_volatility):
    num_assets = len(returns.columns)

    def portfolio_volatility(weights):
        return portfolio_performance(weights, returns)[1]

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: target_volatility - portfolio_volatility(x)}
    )

    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        return None

def find_efficient_portfolio_by_volatility(results, weights_record, target_volatility, risk_free_rate):
    idx = np.where(results[:, 1] <= target_volatility)[0]
    if len(idx) > 0:
        closest_idx = idx[np.argmin(target_volatility - results[idx, 1])]
        efficient_portfolio = weights_record[closest_idx]
        max_return_idx = idx[np.argmax(results[idx, 0])]
        efficient_portfolio = weights_record[max_return_idx]
        portfolio_return = results[max_return_idx, 0]
        portfolio_volatility = results[max_return_idx, 1]
        sharpe_ratio = results[max_return_idx, 2]
        return efficient_portfolio, sharpe_ratio, portfolio_return, portfolio_volatility, True
    else:
        closest_idx = np.argmin(results[:, 1])
        min_volatility = results[closest_idx, 1]
        return None, None, None, None, False

def plot_efficient_frontier(results, min_volatility_idx, max_sharpe_idx, efficient_portfolio, efficient_volatility, returns, filename=None):
    plt.figure(figsize=(12, 8))
    plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Ratio de Sharpe')
    plt.title('Simulación de carteras aleatorias y frontera eficiente')
    plt.xlabel('Volatilidad')
    plt.ylabel('Retorno')
    plt.scatter(results[min_volatility_idx, 1], results[min_volatility_idx, 0], color='r', marker='*', s=100, label='Mínima volatilidad')
    plt.scatter(results[max_sharpe_idx, 1], results[max_sharpe_idx, 0], color='g', marker='*', s=100, label='Máximo Sharpe')
    if efficient_portfolio is not None:
        efficient_return, efficient_volatility = portfolio_performance(efficient_portfolio, returns)
        plt.scatter(efficient_volatility, efficient_return, color='b', marker='*', s=200, label=f'Cartera eficiente ({efficient_volatility:.2%} volatilidad)')
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename)

    plt.close()

def print_weights_return_sharpe(weights, tickers, sharpe_ratio, portfolio_return, portfolio_volatility, risk_free_rate):
    result = "\n".join([f"{ticker}: {weight:.2%}" for ticker, weight in zip(tickers, weights)])
    if portfolio_return is not None:
        result += f"\n\nRendimiento esperado de la cartera: {portfolio_return:.2%}"
        result += f"\nVolatilidad de la cartera: {portfolio_volatility:.2%}"
    else:
        result += "\nEl rendimiento esperado de la cartera no está disponible."
    result += f"\nÍndice de Sharpe: {sharpe_ratio:.4f}"
    if risk_free_rate is not None:
        result += f"\nTasa libre de riesgo: {risk_free_rate:.2%}"
    else:
        result += "\nTasa libre de riesgo no disponible"
    return result

def export_portfolio_to_excel(weights, tickers, sharpe_ratio, portfolio_return, portfolio_volatility, risk_free_rate, filename='efficient_portfolio.xlsx'):
    data = {
        'Ticker': tickers,
        'Ponderación': weights
    }
    df = pd.DataFrame(data)
    summary_data = {
        'Métrica': ['Rendimiento esperado', 'Volatilidad', 'Índice de Sharpe', 'Tasa libre de riesgo'],
        'Valor': [portfolio_return, portfolio_volatility, sharpe_ratio, risk_free_rate]
    }
    summary_df = pd.DataFrame(summary_data)
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Ponderaciones', index=False)
        summary_df.to_excel(writer, sheet_name='Resumen', index=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            target_volatility = float(request.form['volatility']) / 100.0  # Convertir a decimal
            risk_free_rate = get_risk_free_rate()
            
            end_date = datetime.now()
            start_date = end_date - pd.DateOffset(years=5)
            data = download_data(tickers, start_date, end_date)
            returns = calculate_returns(data)
            
            num_portfolios = 10000
            results, weights_record = generate_random_portfolios(returns, num_portfolios)
            
            efficient_portfolio, sharpe_ratio, portfolio_return, portfolio_volatility, found = find_efficient_portfolio_by_volatility(results, weights_record, target_volatility, risk_free_rate)
            
            if not found:
                return render_template('index.html', error_message=f'No se encontró ninguna cartera con una volatilidad igual o inferior al {target_volatility:.2%}.')
            
            max_sharpe_idx = np.argmax(results[:, 2])
            min_volatility_idx = np.argmin(results[:, 1])
            
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            plot_filename = f'static/efficient_frontier_{timestamp}.png'
            plot_efficient_frontier(results, min_volatility_idx, max_sharpe_idx, efficient_portfolio, target_volatility, returns, filename=plot_filename)
            
            excel_filename = f'static/efficient_portfolio_{timestamp}.xlsx'
            export_portfolio_to_excel(efficient_portfolio, tickers, sharpe_ratio, portfolio_return, portfolio_volatility, risk_free_rate, filename=excel_filename)
            
            weights_summary = print_weights_return_sharpe(efficient_portfolio, tickers, sharpe_ratio, portfolio_return, portfolio_volatility, risk_free_rate)
            return render_template('index.html', results=weights_summary, plot_image=plot_filename, excel_file=excel_filename)
        
        except Exception as e:
            return render_template('index.html', error_message=str(e))
    
    return render_template('index.html')

@app.route('/<path:filename>', methods=['GET', 'POST'])
def download_file(filename):
    directory = os.path.join(app.root_path, 'static')
    return send_from_directory(directory, filename, as_attachment=True)

def get_risk_free_rate():
    url = 'https://fred.stlouisfed.org/series/DFF'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    rate_tag = soup.find('span', {'class': 'series-meta-observation-value'})

    if rate_tag:
        try:
            risk_free_rate = float(rate_tag.text.strip().strip('%')) / 100.0
            return risk_free_rate
        except ValueError:
            print("Error al convertir la tasa libre de riesgo a número.")
    
    print("Error al obtener la tasa libre de riesgo desde FRED.")
    return 0.05

if __name__ == '__main__':
    app.run(debug=True)
