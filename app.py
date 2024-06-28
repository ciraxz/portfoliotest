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

app = Flask(__name__)

# Definición de los activos
tickers = ['BTC-USD', 'ETH-USD', 'SAN.MC', 'BBVA.MC', 'BNP.PA', 'GS', 'JPM',
           'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSM', 'TCEHY',
           'GC=F', 'CL=F', 'SI=F', 'NG=F', 'AGG', 'BND', 'EMB', 'IGOV']

def download_data(tickers, start_date, end_date):
    """
    Descarga datos históricos de los activos utilizando yfinance.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_returns(data):
    """
    Calcula los rendimientos diarios de los datos.
    """
    returns = data.pct_change(fill_method=None).dropna()
    return returns

def portfolio_performance(weights, returns):
    """
    Calcula el rendimiento y la volatilidad de la cartera dados los pesos y los rendimientos.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_std_dev

def generate_random_portfolios(returns, num_portfolios):
    """
    Genera 'num_portfolios' carteras aleatorias y calcula su rendimiento, volatilidad y ratio de Sharpe.
    Asegura que las carteras generadas estén en la frontera eficiente.
    """
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    num_generated = 0
    max_attempts = 100 * num_portfolios  # Intentos máximos para generar carteras eficientes
    
    while num_generated < num_portfolios and len(weights_record) < max_attempts:
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std_dev = portfolio_performance(weights, returns)
        portfolio_sharpe_ratio = portfolio_return / portfolio_std_dev
        
        if np.isclose(portfolio_std_dev, results[1]).any():
            continue  # Evitar duplicados exactos
            
        results[0, num_generated] = portfolio_return
        results[1, num_generated] = portfolio_std_dev
        results[2, num_generated] = portfolio_sharpe_ratio
        weights_record.append(weights)
        num_generated += 1
    
    return results[:, :num_generated], weights_record

def optimize_portfolio_for_volatility(returns, target_volatility):
    """
    Optimiza la cartera para una volatilidad objetivo y devuelve la cartera en la frontera eficiente más cercana por debajo de esa volatilidad si no se encuentra exactamente.
    """
    num_assets = len(returns.columns)
    args = (returns,)
    
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
    """
    Encuentra la cartera eficiente para una volatilidad objetivo.
    """
    idx = np.where(results[1] <= target_volatility)[0]
    if len(idx) > 0:
        closest_idx = idx[np.argmin(target_volatility - results[1][idx])]
        efficient_portfolio = weights_record[closest_idx]
        max_return_idx = idx[np.argmax(results[0][idx])]
        efficient_portfolio = weights_record[max_return_idx]
        portfolio_return = results[0][max_return_idx]
        portfolio_volatility = results[1][max_return_idx]
        sharpe_ratio = results[2][max_return_idx]
        return efficient_portfolio, sharpe_ratio, portfolio_return, portfolio_volatility, True
    else:
        closest_idx = np.argmin(results[1])
        min_volatility = results[1][closest_idx]
        print(f"No se encontró ninguna cartera con volatilidad objetivo {target_volatility:.2%}.")
        print(f"La mínima volatilidad encontrada es {min_volatility:.2%}.")
        return None, None, None, None, False

def plot_efficient_frontier(results, min_volatility_idx, max_sharpe_idx, efficient_portfolio, efficient_volatility, returns, filename=None):
    """
    Grafica la frontera eficiente y marca la cartera eficiente encontrada.
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Ratio de Sharpe')
    plt.title('Simulación de carteras aleatorias y frontera eficiente')
    plt.xlabel('Volatilidad')
    plt.ylabel('Retorno')
    plt.scatter(results[1,min_volatility_idx], results[0,min_volatility_idx], color='r', marker='*', s=100, label='Mínima volatilidad')
    plt.scatter(results[1,max_sharpe_idx], results[0,max_sharpe_idx], color='g', marker='*', s=100, label='Máximo Sharpe')
    if efficient_portfolio is not None:
        efficient_return, efficient_volatility = portfolio_performance(efficient_portfolio, returns)
        plt.scatter(efficient_volatility, efficient_return, color='b', marker='*', s=200, label=f'Cartera eficiente ({efficient_volatility:.2%} volatilidad)')
    plt.legend()
    plt.grid(True)
    
    if filename:
        plt.savefig(filename)  # Guardar gráfico como imagen
        
    plt.close()

def print_weights_return_sharpe(weights, tickers, sharpe_ratio, portfolio_return, portfolio_volatility, risk_free_rate):
    """
    Imprime las ponderaciones de los activos, el índice de Sharpe y el rendimiento de la cartera en formato vertical.
    """
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
    """
    Exporta las ponderaciones de los activos, el índice de Sharpe y el rendimiento de la cartera a un archivo Excel.
    """
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
            
            # Descarga de datos
            end_date = datetime.now()
            start_date = end_date - pd.DateOffset(years=5)
            data = download_data(tickers, start_date, end_date)
            
            # Cálculo de rendimientos
            returns = calculate_returns(data)
            
            # Generar carteras aleatorias
            num_portfolios = 10000  # Reducido para mejorar tiempo de ejecución
            results, weights_record = generate_random_portfolios(returns, num_portfolios)
            
            # Encontrar la cartera eficiente para la volatilidad objetivo
            efficient_portfolio, sharpe_ratio, portfolio_return, portfolio_volatility, found = find_efficient_portfolio_by_volatility(results, weights_record, target_volatility, risk_free_rate)
            
            if not found:
                error_message = f"No se encontró ninguna cartera con una volatilidad objetivo de {target_volatility:.2%}. Por favor, intente con un valor diferente."
                return render_template('index.html', error_message=error_message)
            
            # Graficar la frontera eficiente
            min_volatility_idx = np.argmin(results[1])
            max_sharpe_idx = np.argmax(results[2])
            plot_filename = 'static/efficient_frontier.png'
            plot_efficient_frontier(results, min_volatility_idx, max_sharpe_idx, efficient_portfolio, portfolio_volatility, returns, filename=plot_filename)
            
            # Mostrar los resultados en la página web
            results_text = print_weights_return_sharpe(efficient_portfolio, tickers, sharpe_ratio, portfolio_return, portfolio_volatility, risk_free_rate)
            
            # Exportar la cartera eficiente a Excel
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            excel_filename = f'static/efficient_portfolio_{timestamp}.xlsx'
            export_portfolio_to_excel(efficient_portfolio, tickers, sharpe_ratio, portfolio_return, portfolio_volatility, risk_free_rate, filename=excel_filename)
            
            return render_template('index.html', results=results_text, plot_image=plot_filename, excel_file=excel_filename)
        
        except Exception as e:
            error_message = f"Se produjo un error: {e}"
            return render_template('index.html', error_message=error_message)
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static', filename)

def get_risk_free_rate():
    """
    Obtiene la tasa libre de riesgo desde la página web de FRED utilizando web scraping.
    """
    url = 'https://fred.stlouisfed.org/series/DFF'
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        value = soup.find('span', class_='series-meta-observation-value').get_text()
        risk_free_rate = float(value[:-1]) / 100.0  # Convertir a decimal
        return risk_free_rate
    except Exception as e:
        print(f"Error al obtener la tasa libre de riesgo desde {url}: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True)
