import pandas as pd
import numpy as np
import time
import requests
import tensorflow as tf
import optuna
import yfinance as yf
import gym
import random
from gym import spaces
from trading_ig import IGService
from trading_ig.config import config
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from stable_baselines3 import PPO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from stable_baselines3.common.env_util import make_vec_env

from trading_ig.rest import IGService, ApiExceededException
from tenacity import Retrying, wait_exponential, retry_if_exception_type


retryer = Retrying(wait=wait_exponential(),
    retry=retry_if_exception_type(ApiExceededException))

ig_service = IGService(config.username, config.password, config.api_key, config.acc_type, retryer=retryer)


# Define global constants
SLIPPAGE = 0.001  # 0.1% slippage
TRANSACTION_COST = 0.001  # 0.1% transaction cost
POP_SIZE = 10
NGEN = 5

# Define IG API credentials
api_key = 'a0ef648259005286bea79501c599dc0110b728a3'
username = 'us_login'
password = 'pssword'
api_url = 'https://demo-api.ig.com/gateway/deal'

# Define trading symbols
symbols = ['BHP.AX', 'CBA.AX', 'CSL.AX', 'NAB.AX', 'ANZ.AX', 'WBC.AX', 'MQG.AX', 'WDS.AX', 'FMG.AX', 'TSLA', 'GOOGL', 'WES.AX']
timeframe = '5D'
features = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume']

# Function to fetch real-time data using Yahoo Finance
def fetch_realtime_data_yfinance(symbol):
    try:
        # Retrieve real-time data using yfinance
        stock_data = yf.Ticker(symbol)
        data = stock_data.history(period='1d')  # Adjust the period as needed
        # Extract the latest data point
        latest_data = data.tail(1)
        
        # Assuming you want to return a dictionary with relevant information
        return {
            'symbol': symbol,
            'closePrice': float(latest_data['Close']),
            'openPrice': float(latest_data['Open']),
            'highPrice': float(latest_data['High']),
            'lowPrice': float(latest_data['Low']),
            'volume': float(latest_data['Volume']),
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to fetch news using Yahoo Finance API
def fetch_news_yfinance(symbol):
    try:
        # Using yfinance to get news headlines for the specified symbol
        ticker = yf.Ticker(symbol)
        news_data = ticker.news
        headlines = news_data['title'].tolist()
        return headlines
    except Exception as e:
        print(f"Error: {e}")
        return []

# Function to prepare live data for machine learning
def prepare_live_data(live_data, window=10):
    # Check if live_data is a dictionary and convert it to DataFrame
    if isinstance(live_data, dict):
        live_data = pd.DataFrame([live_data])
    
    # Original features of the function
    # Assuming 'live_data' is a DataFrame with relevant columns like 'Open', 'High', 'Low', 'Close', 'Volume'
    # Add here any additional processing you originally had, such as feature engineering, scaling, etc.

    # Example: Scaling the data - adjust as needed based on your original code
    # 1. Normalization and Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(live_data)

    # 2. Moving Averages
    #live_data['SMA'] = live_data['Close'].rolling(window=window).mean()
    #live_data['EMA'] = live_data['Close'].ewm(span=window, adjust=False).mean()
    
    live_data['SMA_5'] = live_data['Close'].rolling(window=5).mean()
    live_data['SMA_10'] = live_data['Close'].rolling(window=10).mean()
    live_data['SMA_20'] = live_data['Close'].rolling(window=20).mean()
    live_data['EMA_5'] = live_data['Close'].ewm(span=5, adjust=False).mean()
    live_data['EMA_10'] = live_data['Close'].ewm(span=10, adjust=False).mean()
    live_data['EMA_20'] = live_data['Close'].ewm(span=20, adjust=False).mean()

    # Calculate RSI
    rsi = pandas_ta.rsi(live_data['Close'], length=14)
    live_data['RSI_14'] = rsi

    # 3. Feature Extraction (Example: RSI)
    delta = live_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    live_data['RSI'] = 100 - (100 / (1 + RS))

    # Combine scaled data with additional features
    live_data_scaled = pd.DataFrame(scaled_data, columns=live_data.columns)
    live_data_final = pd.concat([live_data_scaled, live_data[['SMA', 'EMA', 'RSI']]], axis=1)

    # Handling any NaN values that might have been introduced
    live_data_final = live_data_final.fillna(method='bfill')

    return scaled_data


# Function to execute trading decisions based on real-time data
def execute_trading_decisions(ig_service, symbol, live_data, rl_model, risk_per_trade):
    # Prepare the live data for the model prediction
    live_X = prepare_live_data(live_data, window=10)

    # Predict the action using the RL model
    live_action, _states = rl_model.predict(live_X.iloc[-1].values)

    # Determine the trade size based on the risk per trade
    position_size = calculate_position_size(ml_model, total_trading_capital, live_data, close_price, slippage, transaction_cost)

    # Execute the trading action
    try:
        if live_action == 1:
            place_order(ig_service, symbol, 'BUY', position_size)
        elif live_action == -1:
            place_order(ig_service, symbol, 'SELL', position_size)
        # Optionally handle 'hold' or other actions if needed
    except Exception as e:
        # Handle any exceptions, such as issues with the API call
        print(f"Error executing trade for {symbol}: {e}")


def place_order(ig_service, symbol, direction, position_size):
    """
    Place an order using the IG trading platform.
    Parameters:
        ig_service: The IGService instance to use for placing the order.
        symbol (str): The trading symbol for the order.
        direction (str): 'BUY' or 'SELL'.
        position_size (float): The size of the position.
    """
    ig_service = IGService(config.username, config.password, config.api_key, config.acc_type, retryer=retryer)
    ig_service.create_session()
   
    
    for symbol in symbols_list:
        # Determine direction and position_size for each symbol using your functions
        directioner = direction.upper()  # Replace with your actual function
        position_size = round(position_size, 2)
        position_sizer = position_size  
    
    try:
        # Map specific symbols to custom EPICs
        
        epico = symbol
        if epico == "AAPL":
            epica = "UA.D.AAPL.CASH.IP"
        elif epico == "BHP.AX":
            epica = "AA.D.BHP.CASH.IP"
        elif epico == "CBA.AX":
            epica = "AA.D.CBA.CASH.IP"
        elif epico == "CSL.AX":
            epica = "AA.D.CSL.CASH.IP"
        elif epico == "NAB.AX":
            epica = "AA.D.NAB.CASH.IP"
        elif epico == "ANZ.AX":
            epica = "AA.D.ANZ.CASH.IP"
        elif epico == "WBC.AX":
            epica = "AA.D.WBC.CASH.IP"             
        elif epico == "MQG.AX":
            epica = "AA.D.MQG.CASH.IP"
        elif epico == "WDS.AX":
            epica = "AA.D.WPL.CASH.IP"             

        elif epico == "FMG.AX":
            epica = "AA.D.FMG.CASH.IP"
        elif epico == "TSLA":
            epica = "UD.D.TSLA.CASH.IP"
        elif epico == "GOOGL":
            epica = "UB.D.GOOGL.CASH.IP"            
        elif epico == "WES.AX":
            epica = "AA.D.WESAU.CASH.IP"            
        else:
            raise ValueError(f"No custom EPIC mapping for symbol: {symbol}") 

        order_info = {
            "currency_code": "AUD",  # Modify as needed
            "direction": directioner,  # "BUY" or "SELL"
            "size": position_sizer,  # Size of the order
            "epic": epica,  # Use the custom EPIC
            "order_type": "MARKET",
            "expiry": "DFB",
            "force_open": "false",
            "guaranteed_stop": "false",
            "level": None,
            "limit_distance": None,
            "limit_level": None,
            "quote_id": None,
            "stop_level": None,
            "stop_distance": None,
            "trailing_stop": None,
            "trailing_stop_increment": None
            # Additional parameters as needed
        }
        
        # Print the order information
        print("Order Info:", order_info)
        
        # Place the order using the IGService instance
        response = ig_service.create_open_position(**order_info)
        print('Order placed successfully:', response)
    except Exception as e:
        print('Error placing order:', e)

symbols_list = ["AAPL", "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "ANZ.AX", "WBC.AX", "MQG.AX", "WDS.AX", "FMG.AX", "TSLA", "GOOGL", "WES.AX"]  # List of symbols to place orders
# Function to create a more complex neural network architecture
def create_complex_nn(X_train):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_ml_model():
    """
    Build a machine learning model to predict risk_per_trade.
    """
    model = Sequential([
        Dense(64, input_dim=feature_count, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Output is a risk percentage
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_ml_model(model, X_train, y_train):
    """
    Train the machine learning model.
    """
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    return model

def predict_risk_per_trade(model, market_data):
    """
    Predict risk_per_trade using the trained ML model.
    """
    scaled_data = StandardScaler().fit_transform(np.array(market_data).reshape(-1, feature_count))
    predicted_risk = model.predict(scaled_data)
    return predicted_risk[0][0]

def get_account_balance(ig_service):
    ig_service = IGService(config.username, config.password, config.api_key, config.acc_type)
    ig_service.create_session()
    response = ig_service.fetch_accounts()
    #response = ig_service.fetch_account_by_type(config.acc_type)
    if 'balance' in response:
        return response['balance']
    else:
        raise Exception("Account balance information not available")

# Function to find EPICs for multiple symbols


epics = {
    "AAPL": "UA.D.AAPL.CASH.IP",  # Replace with the actual EPIC for AAPL
    "BHP.AX": "AA.D.BHP.CASH.IP",
    "CBA.AX": "AA.D.CBA.CASH.IP",
    "CSL.AX": "AA.D.CSL.CASH.IP",
    "NAB.AX": "AA.D.NAB.CASH.IP",
    "ANZ.AX": "AA.D.ANZ.CASH.IP",
    "WBC.AX": "AA.D.WBC.CASH.IP",
    "MQG.AX": "AA.D.MQG.CASH.IP",
    "WDS.AX": "AA.D.WPL.CASH.IP",
    "FMG.AX": "AA.D.FMG.CASH.IP", 
    "TSLA":  "UD.D.TSLA.CASH.IP",
    "GOOGL": "UB.D.GOOGL.CASH.IP",
    "WES.AX": "AA.D.WESAU.CASH.IP",
}


# Function to get current prices for multiple symbols
def get_close_price(ig_service, epics):
    close_price = {}
    for symbol, epic in epics.items():
        response = ig_service.fetch_market_by_epic(epic)
        if response and 'snapshot' in response:
            high = response['snapshot']['high']
            low = response['snapshot']['low']
            close_price[symbol] = (high + low) / 2
        else:
            close_price[symbol] = None
            
    return close_price
ig_service = IGService(config.username, config.password, config.api_key, config.acc_type)
total_trading_capital = get_account_balance(ig_service)

def calculate_position_size(ml_model, total_trading_capital, market_data, close_price, slippage, transaction_cost):
    """
    Calculate the position size for a trade, with risk_per_trade determined by an ML model.
    """
    if isinstance(total_trading_capital, pd.Series) and len(total_trading_capital ) == 1:
        total_trading_capital = float(total_trading_capital.iloc[0])
    else:
        total_trading_capital = float(total_trading_capital)  # Assuming it's already a single numeric value

    current_price = float(current_price)
    stop_loss_level = float(stop_loss_level)
    risk_per_trade_percentage = float(risk_per_trade_percentage)
    slippage = float(slippage)
    risk_per_trade = predict_risk_per_trade(ml_model, market_data)

    capital_at_risk = total_trading_capital * risk_per_trade
    total_cost_per_unit = close_price + slippage + transaction_cost
    position_size = capital_at_risk / total_cost_per_unit

    return position_size

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, window_size, df):
        self.window_size = window_size
        super(TradingEnv, self).__init__()
        # Ensure df is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The df argument must be a pandas DataFrame")

        self.df = df

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(30)  # where N is the number of actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(30,), dtype=np.float32)  # where M is the size of observation

        # Add other initializations if needed
    def reset(self):
        # Reset the environment to an initial state
        self.current_step = 0
        #self.current_step = np.zeros(self.observation_space.shape)
      
        return self.get_observation(self.current_step)

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = self.calculate_reward(action)
        next_state = self.get_observation(self.current_step)
        return next_state, reward, done, {}
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
   
    def render(self, mode='console'):
        # Render the environment to the screen
        if mode == 'console':
            print(f"Step: {self.current_step}")
            print(f"Action: {action}")
        elif mode == 'human':
            # For a more advanced visualization, e.g., using matplotlib
            plt.figure(figsize=(15, 6))
            plt.plot(self.df['Close'][:self.current_step])
            plt.title("Stock Price")
            plt.show()

    def get_observation(self, step):
        # Get data observation at a given step
        # Implementation depends on your data structure
        if not isinstance(step, int):
            raise TypeError("Step must be an integer")
        start_index = max(0, step - self.window_size)
        return self.df.iloc[start_index:step]
        #return self.df.iloc[step - self.window_size:step]

    def calculate_reward(self, action):
        # Calculate the reward for an action
        # Implementation depends on your trading strategy
        return 0  # Placeholder, implement your reward calculation

def rl_model(df):
    # Convert the dictionary with scalar values into a DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df[0])
    window_size = 10
    # Assuming 'df' is your DataFrame and you need to split it into features (X) and target (y)
    # Modify the following lines to correctly extract X and y from df
    X = df.drop('closePrice', axis=1)  # Replace 'target_column' with the actual name of your target column
    y = df['closePrice']  # Replace 'target_column' with the actual name of your target column

    # Create and train the RL model
    env = TradingEnv(window_size, df)
    model = PPO("MlpPolicy", env, df)
    #model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50)
    return model
    
# Function to perform live trading
def live_trading(ig_service, symbols, df, live_data, short_window=10, long_window=50, risk_per_trade=1.0):
    # Initialize and train the RL model
    trained_rl_model = rl_model(df)
    if isinstance(live_data, dict):
        live_data = pd.DataFrame([live_data])
    for symbol in symbols:
        live_data = fetch_live_data(ig_service, symbol)  # Fetch live data
        if 'Close' in live_data.columns:
            # Prepare your data for prediction
            live_X = prepare_data_for_prediction(live_data)
            action, _states = trained_rl_model.predict(live_X, deterministic=True)
            
    # Original features of the function
    # Assuming 'live_data' is a DataFrame with relevant columns like 'Open', 'High', 'Low', 'Close', 'Volume'
    # Add here any additional processing you originally had, such as feature engineering, scaling, etc.

    # Example: Scaling the data - adjust as needed based on your original code
    # 1. Normalization and Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(live_data)

    # 2. Moving Averages
    live_data['SMA'] = live_data['Close'].rolling(window=window).mean()
    live_data['EMA'] = live_data['Close'].ewm(span=window, adjust=False).mean()

    # 3. Feature Extraction (Example: RSI)
    delta = live_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    live_data['RSI'] = 100 - (100 / (1 + RS))

    # Combine scaled data with additional features
    live_data_scaled = pd.DataFrame(scaled_data, columns=live_data.columns)
    live_data_final = pd.concat([live_data_scaled, live_data[['SMA', 'EMA', 'RSI']]], axis=1)

    # Handling any NaN values that might have been introduced
    live_data_final = live_data_final.fillna(method='bfill')

    return scaled_data


def set_up_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
def fetch_historical_data(symbol, timeframe, limit=500):
    try:
        df = yf.download(symbol, period='max', interval=timeframe)
        # Rename columns to match expected names in your script
        df.rename(columns={
            'Close': 'closePrice',
            'Open': 'openPrice',
            'High': 'highPrice',
            'Low': 'lowPrice',
            'Volume': 'volume'
        }, inplace=True)
        return df
    except Exception as e:
        print(f"Error: {e}")
		
def fetch_ohlcv(symbol, timeframe, limit=500):
    try:
        # Replace this line with the actual implementation of fetch_historical_data
        df = fetch_historical_data(symbol, timeframe, limit=limit)
        return df
    except Exception as e:
        print(f"Error: {e}")

def compute_rsi(data, window=14):
    """ Compute the Relative Strength Index (RSI) for given data. """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def prepare_data(df, target_col='Signal', features=['closePrice', 'openPrice', 'highPrice', 'lowPrice', 'volume'], window=10):
    """
    Prepare financial data for machine learning with enhanced feature engineering.

    :param df: DataFrame containing financial data.
    :param target_col: The name of the target column.
    :param features: List of column names to be used as features.
    :param window: Window size for rolling calculations.
    :return: Tuple of (features DataFrame, target Series).
    """
    # Ensure that all expected columns are present
    if not all(col in df.columns for col in features):
        raise ValueError("One or more required columns are missing in the DataFrame")

    # Basic Feature Engineering
    df['Signal'] = np.where(df['closePrice'].shift(-1) > df['closePrice'], 1, 0)
    df['Returns'] = df['closePrice'].pct_change()

    # Additional Financial Indicators
    df['SMA'] = df['closePrice'].rolling(window=window).mean()  # Simple Moving Average
    df['STD'] = df['closePrice'].rolling(window=window).std()   # Standard Deviation
    df['RSI'] = compute_rsi(df['closePrice'], window)           # Relative Strength Index

    # Data Normalization/Standardization
    scaler = StandardScaler()
    df[features + ['Returns', 'SMA', 'STD', 'RSI']] = scaler.fit_transform(df[features + ['Returns', 'SMA', 'STD', 'RSI']])

    # Drop any rows with missing values to avoid errors in downstream processing
    df.dropna(inplace=True)

    # Extracting features and target
    X = df[features + ['Returns', 'SMA', 'STD', 'RSI']]
    y = df[target_col]

    return X, y


def moving_average_crossover_strategy(df, short_window, long_window, risk_per_trade):
    df['Short_MA'] = df['closePrice'].rolling(window=short_window).mean()
    df['Long_MA'] = df['closePrice'].rolling(window=long_window).mean()

    df['Signal'] = np.where(df['Short_MA'][short_window:] > df['Long_MA'][short_window:], 1, 0)
    df['Position'] = df['Signal'].diff()

    df['Returns'] = df['closePrice'].pct_change()

    df['Position_Size'] = risk_per_trade / 100 * df['closePrice'].shift(-1) / df['closePrice'] * df['Signal']

    slippage_cost = np.abs(df['Position'].shift(-1)) * SLIPPAGE
    transaction_cost = np.abs(df['Position'].shift(-1)) * TRANSACTION_COST

    df['Returns'] -= slippage_cost
    df['Returns'] -= transaction_cost

    df['Position_Size'] *= (1 - SLIPPAGE - TRANSACTION_COST)

    return df


def train_rl_model(X_train, y_train, total_timesteps=10, tensorboard_log=None, learning_rate=0.001, n_steps=2048):
    if learning_rate <= 0:
        # Option 1: Set a default value
        learning_rate = 0.001
        # Option 2: Raise an error
        # raise ValueError(f"Invalid learning rate: {learning_rate}")
    # Ensure n_steps is an integer and greater than 1
    n_steps = max(1, int(n_steps))

    env = CustomTradingEnvironment(X_train, y_train)
    n_steps = 2 #change later to 5 
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, learning_rate=learning_rate, n_steps=n_steps)
    model.learn(total_timesteps=total_timesteps)
    return model

	
def train_deep_learning_model(X_train, y_train, epochs=50, batch_size=32):
    # Convert X_train to 3D shape for LSTM
    X_train = np.expand_dims(X_train, axis=2)
    
    model = create_complex_nn(X_train)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)
    return model

def evaluate_deep_learning_model(model, X_test, y_test):
    # Convert X_test to 3D shape for LSTM
    X_test = np.expand_dims(X_test, axis=2)

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}')

    return precision, recall, f1

def evaluate_rl_model(rl_model, X_test, y_test):
    total_reward = 0
    env = CustomTradingEnvironment(X_test, y_test)
    obs = env.reset()
    while not env.done:
        action, _ = rl_model.predict(obs)
        obs, reward, _, _ = env.step(action)
        total_reward += reward
    return total_reward

def hyperparameter_tuning_optuna(X_train, y_train, X_test, y_test, n_trials=10):
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        n_steps = trial.suggest_int('n_steps', 1000, 50000)
        model = train_rl_model(X_train, y_train, total_timesteps=10, tensorboard_log="./ppo_trading_tensorboard/",
                                learning_rate=learning_rate, n_steps=n_steps)
        return evaluate_rl_model(model, X_test, y_test)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def hyperparameter_tuning_genetic_algorithm(X_train, y_train, X_test, y_test):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 1e-5, 1e-1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    # Register the mate function
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POP_SIZE)
    algorithms.eaMuPlusLambda(population, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, cxpb=0.7, mutpb=0.2, ngen=NGEN, stats=None, halloffame=None)

    best_individual = tools.selBest(population, k=1)[0]
    best_learning_rate, best_n_steps = best_individual

    model = train_rl_model(X_train, y_train, total_timesteps=10, tensorboard_log="./ppo_trading_tensorboard/",
                           learning_rate=best_learning_rate, n_steps=best_n_steps)

    return evaluate_rl_model(model, X_test, y_test)

def fitness(individual, X_train, y_train, X_test, y_test):
    learning_rate, n_steps = individual
    model = train_rl_model(X_train, y_train, total_timesteps=10, tensorboard_log="./ppo_trading_tensorboard/",
                           learning_rate=learning_rate, n_steps=n_steps)
    return evaluate_rl_model(model, X_test, y_test),
	

def place_order(ig_service, symbol, quantity, action):
    # Implement the logic to place an order using the IG API
    # api.place_order(symbol, quantity, action)
    print(f"Placing {action} order for {quantity} units of {symbol}")

def fetch_realtime_data(ig_service, symbol):
    # Implement the logic to fetch real-time data using the IG API
    # data = api.fetch_realtime_data(symbol)
    data = {"closePrice": 100.0}  # Placeholder, replace with actual data
    return data

def moving_average_crossover_strategy_live(ig_service, df, short_window, long_window, risk_per_trade):
    for symbol in symbols:
        while True:
            # Fetch real-time data
            live_data = fetch_realtime_data(ig_service, symbol)

            # Prepare data for machine learning
            live_X = prepare_live_data(live_data, window=10)  # Implement prepare_live_data

            # Predict using the trained model
            live_action, _states = rl_model.predict(live_X.iloc[-1].values)

            # Execute trading decisions
            if live_action == 1:
                # Place a buy order
                place_order(ig_service, symbol, calculate_position_size(risk_per_trade, live_data['closePrice'], SLIPPAGE, TRANSACTION_COST), action='buy')
            else:
                # Place a sell order or other logic based on your strategy
                place_order(ig_service, symbol, -calculate_position_size(risk_per_trade, live_data['closePrice'], SLIPPAGE, TRANSACTION_COST), action='sell')

            # Add delays between iterations (adjust as needed)
            time.sleep(60)  # Delay for 1 minute (adjust as needed)

def fetch_new_data():
    # Implement code to fetch new data here
    # For example, you can use APIs, web scraping, or any other data source

    # Sample data (replace with actual data fetching code)
    X_new = [1, 2, 3, 4, 5]
    y_new = [10, 20, 30, 40, 50]

    return X_new, y_new
	
class IGSession:
    def __init__(self, api_key, username, password, acc_type):
        self.api_key = api_key
        self.username = username
        self.password = password
        self.acc_type = acc_type  # 'LIVE' or 'DEMO'
        self.acc_number = "Z5GSS8"
        self.ig_service = None
        self.connect()

    def connect(self):
        try:
            self.ig_service = IGService(self.username, self.password, self.api_key, acc_type=self.acc_type, acc_number=self.acc_number)
            self.ig_service.create_session()
            print("Connected to IG API successfully.")
        except Exception as e:
            print(f"Failed to connect to IG API: {e}")

class CustomTradingEnvironment(gym.Env):
    def __init__(self, X, y):
        super(CustomTradingEnvironment, self).__init__()
        self.X = X
        self.y = y
        self.current_step = 0

        # Define action and observation space
        # These should be modified according to your specific environment
        self.action_space = spaces.Discrete(2)  # Example for two actions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.done = False
        return self.X.iloc[self.current_step].values

    def step(self, action):
        reward = 0
        #if self.y.iloc[self.current_step] == action:
        if (self.y.iloc[self.current_step] == action).all():  
            reward = 1
        self.current_step += 1
        if self.current_step >= len(self.X) - 1:
            self.done = True
        return self.X.iloc[self.current_step].values, reward, self.done, {}

# Additional parameters for learning from mistakes
# Add these lines where variables are defined
X_new, y_new = fetch_new_data()  # Implement fetch_new_data function to get new data
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.001

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Example usage
#memory_buffer = ReplayBuffer(capacity=1000)
memory_buffer = []

if len(memory_buffer) >= BATCH_SIZE:
    minibatch = random.sample(memory_buffer, BATCH_SIZE)
    
    # Extract components of the samples and convert them to numpy arrays
    obses = np.array([sample[0] for sample in minibatch])
    actions = np.array([sample[1] for sample in minibatch])
    rewards = np.array([sample[2] for sample in minibatch])
    next_obses = np.array([sample[3] for sample in minibatch])
    dones = np.array([sample[4] for sample in minibatch])

    # Pass the aggregated data to the learn method
    model.learn(obses, actions, rewards, next_obses, dones)

total_episodes = 1000   
"""for episode in range(total_episodes):

    obs = env.reset()
    done = False
    while not done:
        action = model.predict(obs)
        new_obs, reward, done, _ = env.step(action)

        # Add experience to the memory buffer
        memory_buffer.push(obs, action, reward, new_obs, done)

        # Update obs with new_obs for the next iteration
        obs = new_obs

        # Training logic goes here, including sampling from the buffer
        if len(memory_buffer) > batch_size:
            minibatch = memory_buffer.sample(batch_size)
            # Train your model using the minibatch """

def train_rl_model_with_memory(X_train, y_train, total_timesteps=50, tensorboard_log=None):
    # Ensure tensorboard_log is a string or None
    if not isinstance(tensorboard_log, str) and tensorboard_log is not None:
        raise ValueError("tensorboard_log must be a string or None")

    env = CustomTradingEnvironment(X_train, y_train)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    total_episodes = 1000
    batch_size=32   
    # Make sure memory_buffer has enough experiences
    if len(memory_buffer) >= batch_size:
        minibatch = random.sample(memory_buffer, batch_size)

    """for episode in range(total_episodes):

        obs = env.reset()
        done = False
    while not done:
        action = model.predict(obs)
        new_obs, reward, done, _ = env.step(action)

        # Add experience to the memory buffer
        memory_buffer.push(obs, action, reward, new_obs, done)

        # Update obs with new_obs for the next iteration
        obs = new_obs

        # Training logic goes here, including sampling from the buffer
        if len(memory_buffer) > batch_size:
            minibatch = memory_buffer.sample(batch_size)
            # Train your model using the minibatch """

    for t in range(total_timesteps):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            new_obs, reward, done, _ = env.step(action)

            # Store the experience in memory buffer
            memory_buffer.append((obs, action, reward, new_obs, done))
            if len(memory_buffer) > MEMORY_CAPACITY:
                memory_buffer.pop(0)

            obs = new_obs

            # Sample a random minibatch and perform a learning step
            if len(memory_buffer) >= BATCH_SIZE:
                minibatch = random.sample(memory_buffer, BATCH_SIZE)
                for sample in minibatch:
                    model.learn([sample[0]], [sample[1]], [sample[2]], [sample[3]], [sample[4]])

    return model


# Feedback Loop: Evaluate past decisions and adjust model parameters
def feedback_loop(model, X_test, y_test):
    total_reward = 0
    env = CustomTradingEnvironment(X_test, y_test)
    obs = env.reset()
    while not env.done:
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        total_reward += reward

    # Provide feedback and adjust model parameters based on total_reward
    if total_reward < 0:
        model.policy.lr *= 0.9  # Decrease learning rate for underperforming periods
    else:
        model.policy.lr *= 1.1  # Increase learning rate for successful periods

    return total_reward

# Continuous Learning: Periodically retrain the model with new data
def continuous_learning(model, X_new, y_new, total_timesteps=50, tensorboard_log=None):
    # Retrain the model with the new data
    env = CustomTradingEnvironment(X_new, y_new)
    model.learn(total_timesteps=total_timesteps)

    return model

# Adaptive Strategies: Implement adaptive strategies based on market conditions
def adaptive_strategies(model, market_condition_indicator):
    # Adjust model parameters or strategy based on market conditions
    if market_condition_indicator == 'bullish':
        model.policy.net_arch = [64, 64]  # Adjust neural network architecture for bullish markets
    elif market_condition_indicator == 'bearish':
        model.policy.net_arch = [32, 32]  # Adjust neural network architecture for bearish markets

    return model

# Risk Management Improvements: Dynamically adjust risk based on historical performance
def dynamic_risk_management(model, X_train, y_train, X_test, y_test, initial_risk=1.0):
    # Evaluate past performance
    total_reward_train = evaluate_rl_model(model, X_train, y_train)
    total_reward_test = evaluate_rl_model(model, X_test, y_test)

    # Adjust risk based on historical performance
    if total_reward_test < total_reward_train:
        model.policy.risk_factor *= 0.9  # Decrease risk factor for underperforming periods
    else:
        model.policy.risk_factor *= 1.1  # Increase risk factor for successful periods

    return model

def main():
    set_up_gpu()
    for symbol in symbols:
        df = fetch_historical_data(symbol, timeframe, limit=500)
        data = fetch_ohlcv(symbol, timeframe)
        X, y = prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
        
        # Deep learning model training
        deep_learning_model = train_deep_learning_model(X_train, y_train, epochs=10, batch_size=32)
        evaluate_deep_learning_model(deep_learning_model, X_test, y_test)

        # Reinforcement learning model training
        rl_model = train_rl_model(X_train, y_train, total_timesteps=50, tensorboard_log="./ppo_trading_tensorboard/")

        # Reinforcement learning model evaluation
        total_reward = evaluate_rl_model(rl_model, X_test, y_test)
        print(f'Total Reward for {symbol} (PPO): {total_reward}')

        # Fetch news using yfinance
        news_headlines = fetch_news_yfinance(symbol)
        print(f"News Headlines for {symbol} (yfinance): {news_headlines}")

        # Optuna hyperparameter tuning
        best_params_optuna = hyperparameter_tuning_optuna(X_train, y_train, X_test, y_test, n_trials=10)
        rl_model_optuna = train_rl_model(X_train, y_train, total_timesteps=10, tensorboard_log="./ppo_trading_tensorboard/",
                                         learning_rate=best_params_optuna["learning_rate"], n_steps=best_params_optuna["n_steps"])
        total_reward_optuna = evaluate_rl_model(rl_model_optuna, X_test, y_test)
        print(f'Total Reward for {symbol} (Optuna): {total_reward_optuna}')

        # Genetic algorithm hyperparameter tuning
        total_reward_genetic = hyperparameter_tuning_genetic_algorithm(X_train, y_train, X_test, y_test)
        print(f'Total Reward for {symbol} (Genetic Algorithm): {total_reward_genetic}') 

        # Initialize the API service
        ig_service = IGService(config.username, config.password, config.api_key, config.acc_type)
        
        # Fetch live or real-time data
        live_data = fetch_realtime_data(ig_service, symbol)  

        # Live trading
        live_trading(ig_service, symbols, df, live_data, short_window=10, long_window=50, risk_per_trade=1.0)
        
        # Fetch new data
        X_new, y_new = fetch_new_data()    
		
		# Reinforcement learning model training with memory
    rl_model_with_memory = train_rl_model_with_memory(X_train, y_train, total_timesteps=50, tensorboard_log="./ppo_trading_tensorboard/")

    # Feedback Loop: Evaluate past decisions and adjust model parameters
    total_reward_feedback = feedback_loop(rl_model_with_memory, X_test, y_test)
    print(f'Total Reward with Feedback Loop: {total_reward_feedback}')

    # Continuous Learning: Retrain the model with new data
    rl_model_continuous = continuous_learning(rl_model_with_memory, X_new, y_new, total_timesteps=50, tensorboard_log="./ppo_trading_tensorboard/")

    # Adaptive Strategies: Adjust model parameters based on market conditions
    rl_model_adaptive = adaptive_strategies(rl_model_continuous, market_condition_indicator='bullish')

    # Risk Management Improvements: Dynamically adjust risk based on historical performance
    rl_model_dynamic_risk = dynamic_risk_management(rl_model_adaptive, X_train, y_train, X_test, y_test, initial_risk=1.0)

    # Live trading
    live_trading(ig_service, symbols, df, live_data, short_window=10, long_window=50, risk_per_trade=1.0)
if __name__ == '__main__':
    api_key = 'a0ef648259005286bea79501c599dc0110b728a3'
    username = 'us_login'
    password = 'pssword'
    acc_type = 'LIVE'  # or 'DEMO' for a demo account

    ig_session = IGSession(api_key, username, password, acc_type)
    # Now you can use ig_session to interact with the IG API
    main()


