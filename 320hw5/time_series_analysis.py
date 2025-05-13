import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    df = pd.read_csv('dataset/attendance_weather_merged.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    df = df[df['type'] != 'shitpic'].reset_index(drop=True)
    
    combined_data = df.groupby('date')['count'].sum().reset_index()
    
    weather_cols = ['t_max', 't_min', 'precipitation', 'wind_speed']
    for col in weather_cols:
        combined_data[col] = df.groupby('date')[col].first().values
    
    combined_data['day_of_week'] = combined_data['date'].dt.dayofweek
    combined_data['month'] = combined_data['date'].dt.month
    combined_data['is_weekend'] = combined_data['day_of_week'].isin([5, 6]).astype(int)
    
    combined_data = combined_data.interpolate(method='linear')
    
    return combined_data

def find_optimal_arima_params(data, max_p=3, max_d=2, max_q=3):
    p = range(0, max_p + 1)
    d = range(0, max_d + 1)
    q = range(0, max_q + 1)
    
    best_aic = float('inf')
    best_params = None
    
    for params in itertools.product(p, d, q):
        try:
            model = ARIMA(data, order=params)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = params
        except:
            continue
    
    return best_params

def analyze_time_series(data):
    findings = []
    findings.append("\nTIME SERIES ANALYSIS")
    findings.append("=" * 50)
    
    ma3 = data['count'].rolling(window=3, center=True).mean()
    ma7 = data['count'].rolling(window=7, center=True).mean()
    
    findings.append("\n1. TREND ANALYSIS")
    findings.append(f"3-day MA trend at end: {ma3.iloc[-3:].mean():.1f}")
    findings.append(f"7-day MA trend at end: {ma7.iloc[-7:].mean():.1f}")
    
    stl = STL(data['count'], period=7).fit()
    
    findings.append("\n2. DECOMPOSITION ANALYSIS")
    findings.append(f"Trend strength: {np.abs(stl.trend).mean():.1f}")
    findings.append(f"Seasonal strength: {np.abs(stl.seasonal).mean():.1f}")
    findings.append(f"Residual strength: {np.abs(stl.resid).mean():.1f}")
    
    df_result = adfuller(data['count'])
    findings.append("\n3. DICKEY-FULLER TEST")
    findings.append(f"DF Statistic: {df_result[0]:.3f}")
    findings.append(f"p-value: {df_result[1]:.3f}")
    findings.append("Series is " + ("stationary" if df_result[1] < 0.05 else "non-stationary"))
    
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    try:
        best_params = find_optimal_arima_params(train['count'])
        findings.append("\n4. ARIMA MODEL")
        findings.append(f"Optimal parameters (p,d,q): {best_params}")
        
        model = ARIMA(train['count'], order=best_params)
        arima_model = model.fit()
        
        arima_pred = arima_model.forecast(len(test))
        
        rmse = np.sqrt(mean_squared_error(test['count'], arima_pred))
        mae = mean_absolute_error(test['count'], arima_pred)
        r2 = r2_score(test['count'], arima_pred)
        
        findings.append(f"RMSE: {rmse:.2f}")
        findings.append(f"MAE: {mae:.2f}")
        findings.append(f"R2 Score: {r2:.3f}")
        
    except Exception as e:
        findings.append("\n4. ARIMA MODEL")
        findings.append(f"Error fitting ARIMA model: {str(e)}")
    
    weather_vars = ['t_max', 't_min', 'precipitation', 'wind_speed']
    findings.append("\n5. WEATHER IMPACT")
    
    for var in weather_vars:
        corr, p_value = stats.pearsonr(data['count'], data[var])
        findings.append(f"{var} correlation: {corr:.3f} (p={p_value:.3f})")
    
    create_visualizations(data, train, test, arima_pred, stl, ma3, ma7, weather_vars)
    
    return "\n".join(findings)

def create_visualizations(data, train, test, arima_pred, stl, ma3, ma7, weather_vars):
    plt.figure(figsize=(15, 8))
    plt.plot(data['date'], data['count'], label='Original', alpha=0.5)
    plt.plot(data['date'], ma3, label='3-day MA', alpha=0.8)
    plt.plot(data['date'], ma7, label='7-day MA', alpha=0.8)
    plt.title('Attendance Trends with Moving Averages')
    plt.legend()
    plt.savefig('Results/trends_overview.png')
    plt.close()
    
    plt.figure(figsize=(15, 12))
    plt.subplot(411)
    plt.plot(data['date'], data['count'])
    plt.title('Original Time Series')
    plt.subplot(412)
    plt.plot(data['date'], stl.trend)
    plt.title('Trend')
    plt.subplot(413)
    plt.plot(data['date'], stl.seasonal)
    plt.title('Seasonal')
    plt.subplot(414)
    plt.plot(data['date'], stl.resid)
    plt.title('Residual')
    plt.tight_layout()
    plt.savefig('Results/stl_decomposition.png')
    plt.close()
    
    plt.figure(figsize=(15, 8))
    plt.plot(test.index, test['count'], label='Actual')
    plt.plot(test.index, arima_pred, label='Forecast')
    plt.fill_between(test.index, 
                     arima_pred - 1.96 * np.std(test['count'] - arima_pred),
                     arima_pred + 1.96 * np.std(test['count'] - arima_pred),
                     alpha=0.2)
    plt.title('ARIMA Forecast vs Actual')
    plt.legend()
    plt.savefig('Results/arima_forecast.png')
    plt.close()
    
    max_lags = min(int(len(data) * 0.25), 40)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    acf_values = acf(data['count'], nlags=max_lags)
    pacf_values = pacf(data['count'], nlags=max_lags)
    ax1.stem(range(len(acf_values)), acf_values)
    ax1.set_title('Autocorrelation Function')
    ax2.stem(range(len(pacf_values)), pacf_values)
    ax2.set_title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.savefig('Results/acf_pacf.png')
    plt.close()
    
    n_vars = len(weather_vars)
    fig = plt.figure(figsize=(15, 3 * ((n_vars + 1) // 2)))
    
    for i, var in enumerate(weather_vars, 1):
        ax = plt.subplot(((n_vars + 1) // 2), 2, i)
        
        count_bins = np.linspace(data['count'].min(), data['count'].max(), 20)
        var_bins = np.linspace(data[var].min(), data[var].max(), 20)
        
        hist, xedges, yedges = np.histogram2d(data[var], data['count'], 
                                             bins=[var_bins, count_bins])
        
        im = ax.imshow(hist.T, origin='lower', aspect='auto', 
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      cmap='YlOrRd')
        
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(var)
        ax.set_ylabel('Count')
        ax.set_title(f'Count vs {var}\nr = {data[var].corr(data["count"]):.2f}')
    
    plt.tight_layout()
    plt.savefig('Results/weather_impact.png')
    plt.close()

data = load_and_prepare_data()

with open('Results/time_series_observations.txt', 'w') as f:
    f.write(analyze_time_series(data)) 