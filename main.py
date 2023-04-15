import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.layers import LSTM, Dense
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, \
    QTableWidget, QTableWidgetItem, QComboBox, QFormLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import joblib
import plotly.express as px
import plotly.io as pio
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

pio.renderers.default = 'browser'

# Fetching the Stock Market Data
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
start_date = '2020-01-01'
end_date = '2023-12-31'

# Download the data
data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']

# Calculating the Correlation Matrix
correlation_matrix = data.corr()


# Feature Engineering
def create_features(data, indicators=None):
    if indicators is None:
        indicators = ['return', 'ma_7', 'ma_21', 'volatility', 'momentum', 'log_return', 'rolling_max', 'rolling_min',
                      'rolling_std', 'rsi', 'macd', 'macd_signal', 'bollinger_hband', 'bollinger_lband']

    features = pd.DataFrame()
    for symbol in data.columns:
        if 'return' in indicators:
            features[f'{symbol}_return'] = data[symbol].pct_change()
        if 'ma_7' in indicators:
            features[f'{symbol}_ma_7'] = data[symbol].rolling(window=7).mean()
        if 'ma_21' in indicators:
            features[f'{symbol}_ma_21'] = data[symbol].rolling(window=21).mean()
        if 'volatility' in indicators:
            features[f'{symbol}_volatility'] = data[symbol].pct_change().rolling(window=21).std()
        if 'momentum' in indicators:
            features[f'{symbol}_momentum'] = data[symbol].pct_change(4)
        if 'log_return' in indicators:
            features[f'{symbol}_log_return'] = np.log(data[symbol] / data[symbol].shift(1))
        if 'rolling_max' in indicators:
            features[f'{symbol}_rolling_max'] = data[symbol].rolling(window=21).max()
        if 'rolling_min' in indicators:
            features[f'{symbol}_rolling_min'] = data[symbol].rolling(window=21).min()
        if 'rolling_std' in indicators:
            features[f'{symbol}_rolling_std'] = data[symbol].rolling(window=21).std()
        if 'rsi' in indicators:
            features[f'{symbol}_rsi'] = RSIIndicator(data[symbol]).rsi()
        if 'macd' in indicators:
            macd = MACD(data[symbol])
            features[f'{symbol}_macd'] = macd.macd()
            features[f'{symbol}_macd_signal'] = macd.macd_signal()
        if 'bollinger_hband' in indicators:
            bollinger = BollingerBands(data[symbol])
            features[f'{symbol}_bollinger_hband'] = bollinger.bollinger_hband()
        if 'bollinger_lband' in indicators:
            bollinger = BollingerBands(data[symbol])
            features[f'{symbol}_bollinger_lband'] = bollinger.bollinger_lband()
    features['Target'] = data['AAPL'].shift(-1)  # Predict next day price
    return features.dropna()


features = create_features(data)

# Automated Feature Selection
X = features.drop(columns=['Target'])
y = features['Target']
selector = SelectKBest(score_func=f_regression, k='all')
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Convert selected features back to DataFrame
X = pd.DataFrame(X_selected, columns=selected_features)

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Machine Learning Pipeline and Models
def build_model(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'Multi-Layer Perceptron': MLPRegressor(random_state=42, max_iter=500)
}

results = {}

# Hyperparameter Tuning using GridSearchCV
param_grids = {
    'Random Forest': {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20],
    },
    'Gradient Boosting': {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.1, 0.05, 0.01],
    },
    'XGBoost': {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.1, 0.05, 0.01],
        'regressor__max_depth': [3, 5, 7],
    },
    'Support Vector Regressor': {
        'regressor__C': [0.1, 1, 10],
        'regressor__epsilon': [0.1, 0.2, 0.3],
    },
    'Multi-Layer Perceptron': {
        'regressor__hidden_layer_sizes': [(100,), (100, 100)],
        'regressor__learning_rate_init': [0.001, 0.01],
    }
}

for model_name, model in models.items():
    if model_name in param_grids:
        pipeline = build_model(model)
        grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = build_model(model)
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'MSE': mse, 'R²': r2, 'predictions': y_pred, 'model': best_model}
    print(f'{model_name} MSE: {mse}, R²: {r2}')

# LSTM Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
    LSTM(50),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=1)
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
r2_lstm = r2_score(y_test, y_pred_lstm)

results['LSTM'] = {'MSE': mse_lstm, 'R²': r2_lstm, 'predictions': y_pred_lstm, 'model': lstm_model}
print(f'LSTM MSE: {mse_lstm}, R²: {r2_lstm}')

# Plotting
figs = []
plotly_figs = []
for model_name, result in results.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, result['predictions'], alpha=0.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    ax.set_title(f'{model_name}: Actual vs Predicted')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    figs.append((fig, f'{model_name} - MSE: {result["MSE"]:.4f}, R²: {result["R²"]:.4f}'))

    fig = px.scatter(x=y_test, y=result['predictions'], labels={'x': 'Actual', 'y': 'Predicted'},
                     title=f'{model_name}: Actual vs Predicted')
    fig.add_shape(type='line', x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
                  line=dict(color='Red', dash='dash'))
    plotly_figs.append((fig, f'{model_name} - MSE: {result["MSE"]:.4f}, R²: {result["R²"]:.4f}'))


# Save and Load Models
def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    return joblib.load(filename)


class MplCanvas(FigureCanvas):
    def __init__(self, fig):
        self.fig = fig
        super().__init__(self.fig)


class StockMarketAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Stock Market Analysis")
        self.setGeometry(100, 100, 1200, 1000)

        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Tab 1: Stock Data
        tab1 = QWidget()
        self.create_stock_data_tab(tab1)
        self.tabs.addTab(tab1, "Stock Data")

        # Tab 2: Correlation Matrix
        tab2 = QWidget()
        self.create_correlation_matrix_tab(tab2)
        self.tabs.addTab(tab2, "Correlation Matrix")

        # Tabs for each model's performance
        for i, (fig, label) in enumerate(figs):
            tab = QWidget()
            self.create_model_tab(tab, fig, label)
            self.tabs.addTab(tab, label.split(' - ')[0])

        # Tab for model comparison
        tab_comparison = QWidget()
        self.create_model_comparison_tab(tab_comparison)
        self.tabs.addTab(tab_comparison, "Model Comparison")

    def create_stock_data_tab(self, tab):
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        self.symbols_input = QLineEdit(','.join(symbols))
        self.start_date_input = QLineEdit(start_date)
        self.end_date_input = QLineEdit(end_date)
        self.indicator_select = QComboBox()
        self.indicator_select.addItems(
            ['return', 'ma_7', 'ma_21', 'volatility', 'momentum', 'log_return', 'rolling_max', 'rolling_min',
             'rolling_std', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband'])
        self.indicator_select.setEditable(True)
        self.indicator_select.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        download_button = QPushButton("Download Data")
        download_button.clicked.connect(self.download_data)

        form_layout.addRow(QLabel("Symbols (comma separated):"), self.symbols_input)
        form_layout.addRow(QLabel("Start Date (YYYY-MM-DD):"), self.start_date_input)
        form_layout.addRow(QLabel("End Date (YYYY-MM-DD):"), self.end_date_input)
        form_layout.addRow(QLabel("Indicators:"), self.indicator_select)

        layout.addLayout(form_layout)
        layout.addWidget(download_button)

        self.data_text = QtWidgets.QTextEdit()
        self.data_text.setPlainText(data.head().to_string())
        layout.addWidget(self.data_text)
        tab.setLayout(layout)

    def download_data(self):
        global data, correlation_matrix, features, X, y, X_train, X_test, y_train, y_test
        symbols = self.symbols_input.text().split(',')
        start_date = self.start_date_input.text()
        end_date = self.end_date_input.text()
        indicators = self.indicator_select.currentText().split(',')

        data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
        self.data_text.setPlainText(data.head().to_string())

        correlation_matrix = data.corr()
        features = create_features(data, indicators)
        X = features.drop(columns=['Target'])
        y = features['Target']
        selector = SelectKBest(score_func=f_regression, k='all')
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        X = pd.DataFrame(X_selected, columns=selected_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.update_correlation_matrix_tab()
        self.update_model_tabs()

    def create_correlation_matrix_tab(self, tab):
        layout = QVBoxLayout()
        self.correlation_layout = layout
        tab.setLayout(layout)
        self.update_correlation_matrix_tab()

    def update_correlation_matrix_tab(self):
        if hasattr(self, 'correlation_canvas') and self.correlation_canvas:
            self.correlation_canvas.setParent(None)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Stock Price Correlation Matrix')
        self.correlation_canvas = MplCanvas(fig)
        self.correlation_layout.addWidget(self.correlation_canvas)

    def create_model_tab(self, tab, fig, label):
        layout = QVBoxLayout()
        canvas = MplCanvas(fig)
        layout.addWidget(canvas)
        tab.setLayout(layout)

    def update_model_tabs(self):
        global results, figs, plotly_figs
        results = {}
        for model_name, model in models.items():
            if model_name in param_grids:
                pipeline = build_model(model)
                grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=3, scoring='neg_mean_squared_error',
                                           n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = build_model(model)
                best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[model_name] = {'MSE': mse, 'R²': r2, 'predictions': y_pred, 'model': best_model}
            print(f'{model_name} MSE: {mse}, R²: {r2}')

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
            LSTM(50),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=1)
        y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)
        r2_lstm = r2_score(y_test, y_pred_lstm)
        results['LSTM'] = {'MSE': mse_lstm, 'R²': r2_lstm, 'predictions': y_pred_lstm, 'model': lstm_model}
        print(f'LSTM MSE: {mse_lstm}, R²: {r2_lstm}')

        figs = []
        plotly_figs = []
        for model_name, result in results.items():
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(y_test, result['predictions'], alpha=0.5)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
            ax.set_title(f'{model_name}: Actual vs Predicted')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            figs.append((fig, f'{model_name} - MSE: {result["MSE"]:.4f}, R²: {result["R²"]:.4f}'))

            fig = px.scatter(x=y_test, y=result['predictions'], labels={'x': 'Actual', 'y': 'Predicted'},
                             title=f'{model_name}: Actual vs Predicted')
            fig.add_shape(type='line', x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
                          line=dict(color='Red', dash='dash'))
            plotly_figs.append((fig, f'{model_name} - MSE: {result["MSE"]:.4f}, R²: {result["R²"]:.4f}'))

        for i in range(len(self.tabs)):
            if ' - ' in self.tabs.tabText(i):
                self.tabs.removeTab(i)
                i -= 1

        for i, (fig, label) in enumerate(figs):
            tab = QWidget()
            self.create_model_tab(tab, fig, label)
            self.tabs.addTab(tab, label.split(' - ')[0])

    def create_model_comparison_tab(self, tab):
        layout = QVBoxLayout()
        self.comparison_table = QTableWidget()
        self.comparison_table.setRowCount(len(results))
        self.comparison_table.setColumnCount(3)
        self.comparison_table.setHorizontalHeaderLabels(['Model', 'MSE', 'R²'])

        for i, (model_name, result) in enumerate(results.items()):
            self.comparison_table.setItem(i, 0, QTableWidgetItem(model_name))
            self.comparison_table.setItem(i, 1, QTableWidgetItem(f'{result["MSE"]:.4f}'))
            self.comparison_table.setItem(i, 2, QTableWidgetItem(f'{result["R²"]:.4f}'))

        layout.addWidget(self.comparison_table)
        tab.setLayout(layout)


def main():
    app = QApplication(sys.argv)
    window = StockMarketAnalysisApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
