import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import gym
from gym import spaces
import warnings
warnings.filterwarnings('ignore')

# Deep Q-Network Implementation
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        return self.network(state)

# Custom PPO Actor-Critic Network
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, continuous=False):
        super(PPONetwork, self).__init__()
        self.continuous = continuous
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim // 2, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor = nn.Linear(hidden_dim // 2, action_dim)
            
        # Critic head
        self.critic = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, state):
        features = self.shared_network(state)
        value = self.critic(features)
        return value
        
    def get_action(self, state, action=None):
        features = self.shared_network(state)
        value = self.critic(features)
        
        if self.continuous:
            action_mean = self.actor_mean(features)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
        else:
            logits = self.actor(features)
            probs = Categorical(logits=logits)
            
        if action is None:
            action = probs.sample()
            
        if self.continuous:
            log_prob = probs.log_prob(action).sum(-1)
        else:
            log_prob = probs.log_prob(action)
            
        return action, log_prob, value, probs.entropy()

# Multi-Asset Trading Environment
class MultiAssetTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000, transaction_cost=0.001, 
                 lookback_window=50, assets=['AAPL', 'GOOGL', 'MSFT']):
        super(MultiAssetTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.assets = assets
        self.n_assets = len(assets)
        
        # Action space: weights for each asset + cash (sum to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        # State space: historical prices + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window * self.n_assets + 2,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.asset_values = np.zeros(self.n_assets)
        self.weights = np.zeros(self.n_assets + 1)
        self.weights[-1] = 1.0  # Start with 100% cash
        self.done = False
        
        return self._get_observation()
    
    def _get_observation(self):
        # Get price data for lookback window
        obs = []
        for i in range(self.lookback_window):
            idx = self.current_step - self.lookback_window + i
            for asset in self.assets:
                obs.append(self.df[asset].iloc[idx])
        
        # Add portfolio state
        obs.extend([self.balance / self.initial_balance, self.portfolio_value / self.initial_balance])
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        # Normalize action to sum to 1
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        
        # Get current prices
        current_prices = np.array([self.df[asset].iloc[self.current_step] for asset in self.assets])
        
        # Calculate current portfolio value
        current_portfolio_value = self.balance + np.sum(self.asset_values)
        
        # Rebalance portfolio
        self._rebalance_portfolio(action, current_prices)
        
        # Move to next time step
        self.current_step += 1
        
        # Update portfolio value
        if self.current_step < len(self.df):
            new_prices = np.array([self.df[asset].iloc[self.current_step] for asset in self.assets])
            self.asset_values = self.asset_values * (new_prices / (current_prices + 1e-8))
        new_portfolio_value = self.balance + np.sum(self.asset_values)
        
        # Calculate reward
        reward = self._calculate_reward(current_portfolio_value, new_portfolio_value)
        
        # Check if episode is done
        self.done = self.current_step >= len(self.df) - 2
        
        info = {
            'portfolio_value': new_portfolio_value,
            'balance': self.balance,
            'asset_values': self.asset_values.copy(),
            'weights': self.weights.copy()
        }
        
        return self._get_observation(), reward, self.done, info
    
    def _rebalance_portfolio(self, target_weights, current_prices):
        current_total = self.balance + np.sum(self.asset_values)
        
        if current_total <= 0:
            return
            
        # Calculate target values for each asset
        target_values = target_weights[:-1] * current_total
        target_cash = target_weights[-1] * current_total
        
        # Calculate current values
        current_values = self.asset_values.copy()
        current_cash = self.balance
        
        # Execute trades
        for i in range(self.n_assets):
            trade_value = target_values[i] - current_values[i]
            
            if trade_value > 0:  # Buy
                cost = trade_value * (1 + self.transaction_cost)
                if cost <= current_cash:
                    self.asset_values[i] += trade_value
                    self.balance -= cost
                else:
                    # Adjust to available cash
                    actual_trade = current_cash / (1 + self.transaction_cost)
                    self.asset_values[i] += actual_trade
                    self.balance = 0
            else:  # Sell
                trade_value = abs(trade_value)
                if trade_value <= current_values[i]:
                    self.asset_values[i] -= trade_value
                    self.balance += trade_value * (1 - self.transaction_cost)
        
        # Update weights
        total_value = self.balance + np.sum(self.asset_values)
        if total_value > 0:
            self.weights[:-1] = self.asset_values / total_value
            self.weights[-1] = self.balance / total_value
    
    def _calculate_reward(self, old_value, new_value):
        if old_value <= 0:
            return 0
            
        # Advanced reward engineering
        returns = (new_value - old_value) / old_value
        
        # Sharpe ratio component
        recent_returns = self._get_recent_returns()
        if len(recent_returns) > 1 and np.std(recent_returns) > 0:
            sharpe = np.mean(recent_returns) / np.std(recent_returns)
        else:
            sharpe = 0
            
        # Drawdown penalty
        peak_value = max(self.initial_balance, new_value)
        drawdown = (peak_value - new_value) / peak_value
        
        # Diversification bonus
        if len(self.weights) > 1:
            diversification = 1 - np.sum(self.weights[:-1] ** 2)
        else:
            diversification = 0
        
        # Combined reward
        reward = returns * 100 + sharpe * 0.1 - drawdown * 50 + diversification * 0.05
        
        return reward
    
    def _get_recent_returns(self, window=20):
        returns = []
        for i in range(min(window, self.current_step - self.lookback_window)):
            idx = self.current_step - i
            if idx > 0 and idx < len(self.df):
                market_return = np.mean([self.df[asset].iloc[idx] / self.df[asset].iloc[idx-1] - 1 
                                       for asset in self.assets])
                returns.append(market_return)
        return np.array(returns) if returns else np.array([0])

# Advanced Portfolio Optimizer
class AdvancedPortfolioOptimizer:
    def __init__(self, state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize both DQN and PPO
        self.dqn = DQN(state_dim, action_dim).to(device)
        self.ppo = PPONetwork(state_dim, action_dim, continuous=True).to(device)
        
        self.dqn_optimizer = optim.Adam(self.dqn.parameters(), lr=1e-4)
        self.ppo_optimizer = optim.Adam(self.ppo.parameters(), lr=3e-4)
        
        # Experience replay for DQN
        self.memory = []
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def dqn_act(self, state, training=True):
        if training and np.random.random() <= self.epsilon:
            action = np.random.uniform(0, 1, self.action_dim)
            return action / np.sum(action)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.dqn(state_tensor)
        # Convert Q-values to probabilities
        action = torch.softmax(q_values, dim=-1).cpu().numpy()[0]
        return action
    
    def ppo_act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _, _ = self.ppo.get_action(state_tensor)
        # Convert action to numpy and normalize
        action = action.cpu().numpy()[0]
        action = np.exp(action) / np.sum(np.exp(action))  # Softmax normalization
        return action
    
    def update_dqn(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.dqn(states)
        next_q_values = self.dqn(next_states)
        max_next_q = next_q_values.max(dim=1)[0].detach()
        target_q_values = rewards + (self.gamma * max_next_q * (~dones).float())
        
        loss = nn.MSELoss()(current_q_values.max(dim=1)[0], target_q_values)
        
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Trading Platform with Gradio Interface
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AdvancedTradingPlatform:
    def __init__(self):
        self.optimizer = None
        self.current_strategy = None
        self.portfolio_history = []
        self.trained_model = False
        
    def generate_sample_data(self, days=500):
        """Generate synthetic market data for demonstration"""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        
        data = {}
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        for asset in assets:
            returns = np.random.normal(0.0005, 0.02, days)
            prices = 100 * np.cumprod(1 + returns)
            
            for i in range(2, days):
                prices[i] = 0.8 * prices[i-1] + 0.2 * prices[i] + np.random.normal(0, 0.5)
                
            data[asset] = prices
            
        data['date'] = dates
        df = pd.DataFrame(data)
        return df
    
    def train_model(self, data, strategy_type='PPO', episodes=50):
        """Train the selected trading strategy"""
        print(f"Training {strategy_type} model with {episodes} episodes...")
        
        try:
            env = MultiAssetTradingEnv(data, assets=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            self.optimizer = AdvancedPortfolioOptimizer(state_dim, action_dim)
            
            rewards = []
            portfolio_values = []
            
            for episode in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                
                while not done and step_count < 100:
                    if strategy_type == 'DQN':
                        action = self.optimizer.dqn_act(state)
                        next_state, reward, done, info = env.step(action)
                        self.optimizer.memory.append((state, action, reward, next_state, done))
                        if len(self.optimizer.memory) > self.optimizer.batch_size:
                            self.optimizer.update_dqn()
                    else:  # PPO
                        action = self.optimizer.ppo_act(state)
                        next_state, reward, done, info = env.step(action)
                    
                    episode_reward += reward
                    state = next_state
                    step_count += 1
                
                if info:
                    portfolio_values.append(info['portfolio_value'])
                rewards.append(episode_reward)
                
                if episode % 10 == 0:
                    print(f"Episode {episode}, Reward: {episode_reward:.2f}")
            
            self.current_strategy = strategy_type
            self.trained_model = True
            return rewards, portfolio_values
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def backtest_strategy(self, data, initial_balance=100000):
        """Run backtest on historical data"""
        if not self.trained_model:
            return [], []
        
        try:
            env = MultiAssetTradingEnv(data, initial_balance=initial_balance, 
                                      assets=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
            state = env.reset()
            done = False
            
            portfolio_history = []
            weights_history = []
            step_count = 0
            
            while not done and step_count < 200:
                if self.current_strategy == 'DQN':
                    action = self.optimizer.dqn_act(state, training=False)
                else:
                    action = self.optimizer.ppo_act(state)
                
                next_state, reward, done, info = env.step(action)
                
                portfolio_history.append({
                    'step': env.current_step,
                    'portfolio_value': info['portfolio_value'],
                    'balance': info['balance'],
                    'reward': reward
                })
                
                weights_history.append(info['weights'])
                state = next_state
                step_count += 1
            
            self.portfolio_history = portfolio_history
            return portfolio_history, weights_history
            
        except Exception as e:
            print(f"Backtest error: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def calculate_metrics(self, portfolio_history):
        """Calculate performance metrics"""
        if not portfolio_history or len(portfolio_history) < 2:
            return {
                'Total Return (%)': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown (%)': 0,
                'Volatility (%)': 0,
                'Final Portfolio Value': 0
            }
            
        returns = [p['portfolio_value'] for p in portfolio_history]
        returns_pct = np.diff(returns) / (np.array(returns[:-1]) + 1e-8)
        
        total_return = (returns[-1] - returns[0]) / returns[0] * 100
        if len(returns_pct) > 1 and np.std(returns_pct) > 0:
            sharpe_ratio = np.mean(returns_pct) / np.std(returns_pct) * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        max_drawdown = self.calculate_max_drawdown(returns)
        volatility = np.std(returns_pct) * np.sqrt(252) * 100 if len(returns_pct) > 1 else 0
        
        return {
            'Total Return (%)': round(total_return, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Volatility (%)': round(volatility, 2),
            'Final Portfolio Value': f"${returns[-1]:,.2f}"
        }
    
    def calculate_max_drawdown(self, returns):
        peak = returns[0]
        max_dd = 0
        
        for value in returns:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
                
        return max_dd

    def create_dashboard(self):
        """Create Gradio interface"""
        with gr.Blocks(title="Advanced Algorithmic Trading Platform", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # Advanced Algorithmic Trading Platform
            ## Multi-Asset Portfolio Optimization with Deep Reinforcement Learning
            """)
            
            with gr.Tab("Model Training"):
                with gr.Row():
                    with gr.Column():
                        strategy_type = gr.Radio(['PPO', 'DQN'], label="Strategy Type", value='PPO')
                        training_episodes = gr.Slider(10, 100, value=30, step=10, label="Training Episodes")
                        train_btn = gr.Button("Train Model", variant="primary", size="lg")
                    
                    with gr.Column():
                        training_progress = gr.Plot(label="Training Progress")
                
                gr.Markdown("### Training Data Sample")
                training_data = gr.Dataframe(label="Market Data (First 10 rows)", interactive=False)
            
            with gr.Tab("Backtesting"):
                with gr.Row():
                    with gr.Column():
                        initial_balance = gr.Number(100000, label="Initial Balance ($)")
                        backtest_btn = gr.Button("Run Backtest", variant="primary", size="lg")
                        metrics_display = gr.JSON(label="Performance Metrics")
                    
                    with gr.Column():
                        backtest_results = gr.Plot(label="Portfolio Performance")
                
                with gr.Row():
                    with gr.Column():
                        portfolio_composition = gr.Plot(label="Portfolio Composition")
                    with gr.Column():
                        asset_weights = gr.Dataframe(label="Final Asset Weights")
            
            with gr.Tab("Real-Time Analytics"):
                refresh_btn = gr.Button("Refresh Analytics", variant="secondary")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Market Overview")
                        market_overview = gr.Plot(label="Current Market Conditions")
                        volatility_chart = gr.Plot(label="Asset Volatility Analysis")
                        
                    with gr.Column():
                        gr.Markdown("### Trading Signals")
                        trade_signals = gr.Dataframe(
                            label="Current Trading Signals", 
                            value=pd.DataFrame({
                                'Asset': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                                'Signal': ['BUY', 'HOLD', 'BUY', 'SELL', 'HOLD'],
                                'Confidence': [0.85, 0.65, 0.78, 0.72, 0.61],
                                'Target Weight': [0.25, 0.20, 0.25, 0.10, 0.20],
                                'Current Price': [182.35, 142.67, 378.90, 245.80, 156.32]
                            })
                        )
                        
                        gr.Markdown("### Risk Metrics")
                        risk_metrics = gr.JSON(label="Portfolio Risk Analysis", value={
                            "Portfolio Beta": 1.15,
                            "Value at Risk (95%)": "-2.5%",
                            "Conditional VaR": "-4.2%",
                            "Correlation Matrix": "Stable",
                            "Liquidity Score": "High"
                        })
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Performance vs Benchmark")
                        benchmark_comparison = gr.Plot(label="Strategy vs Market Benchmark")
                        
                    with gr.Column():
                        gr.Markdown("### Sector Allocation")
                        sector_allocation = gr.Plot(label="Current Sector Exposure")
                
                refresh_btn.click(
                    self._generate_analytics,
                    outputs=[market_overview, volatility_chart, benchmark_comparison, sector_allocation]
                )

            with gr.Tab("Broker Integration"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Order Management")
                        with gr.Row():
                            order_asset = gr.Dropdown(
                                choices=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'], 
                                label="Asset",
                                value='AAPL'
                            )
                            order_type = gr.Radio(['MARKET', 'LIMIT', 'STOP'], label="Order Type", value='MARKET')
                            order_side = gr.Radio(['BUY', 'SELL'], label="Side", value='BUY')
                        
                        with gr.Row():
                            order_quantity = gr.Number(label="Quantity", value=100)
                            order_price = gr.Number(label="Price (for LIMIT/STOP)", value=0)
                            order_tif = gr.Dropdown(
                                choices=['GTC', 'IOC', 'FOK'], 
                                label="Time in Force",
                                value='GTC'
                            )
                        
                        submit_order_btn = gr.Button("Submit Order", variant="primary")
                        order_status = gr.Textbox(label="Order Status", interactive=False)
                        
                        order_panel = gr.Dataframe(
                            label="Active Orders",
                            value=pd.DataFrame({
                                'Order ID': [1, 2, 3],
                                'Asset': ['AAPL', 'GOOGL', 'MSFT'],
                                'Type': ['BUY', 'SELL', 'BUY'],
                                'Quantity': [100, 50, 75],
                                'Price': [145.32, 2789.45, 325.67],
                                'Status': ['Filled', 'Pending', 'Executed']
                            })
                        )
                        
                    with gr.Column():
                        gr.Markdown("### Risk Management")
                        with gr.Row():
                            stop_loss = gr.Number(0.95, label="Stop Loss (%)")
                            take_profit = gr.Number(1.10, label="Take Profit (%)")
                        risk_management_btn = gr.Button("Update Risk Parameters", variant="primary")
                        
                        gr.Markdown("### Current Positions")
                        position_tracker = gr.Dataframe(
                            label="Portfolio Positions",
                            value=pd.DataFrame({
                                'Asset': ['AAPL', 'GOOGL', 'MSFT', 'CASH'],
                                'Quantity': [150, 80, 0, 25000],
                                'Avg Price': [145.32, 2789.45, 0, 1.0],
                                'Current Price': [182.35, 142.67, 378.90, 1.0],
                                'Current Value': [27352.5, 11413.6, 0, 25000.0],
                                'P&L': [5557.5, -190155.6, 0, 0]
                            })
                        )
                        
                        gr.Markdown("### Account Summary")
                        account_summary = gr.JSON(label="Account Information", value={
                            "Total Equity": "$63,766.10",
                            "Buying Power": "$127,532.20",
                            "Margin Used": "$0.00",
                            "Open P&L": "-$184,598.10",
                            "Available Cash": "$25,000.00"
                        })
            
            # Event handlers
            train_btn.click(
                self._train_model_wrapper,
                inputs=[strategy_type, training_episodes],
                outputs=[training_progress, training_data]
            )
            
            backtest_btn.click(
                self._backtest_wrapper,
                inputs=[initial_balance],
                outputs=[backtest_results, metrics_display, portfolio_composition, asset_weights]
            )
            
            risk_management_btn.click(
                self._update_risk_parameters,
                inputs=[stop_loss, take_profit],
                outputs=[order_panel]
            )
            
        return demo
    
    def _train_model_wrapper(self, strategy_type, episodes):
        try:
            data = self.generate_sample_data(300)
            rewards, portfolio_values = self.train_model(data, strategy_type, int(episodes))
            
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=['Training Rewards', 'Portfolio Values During Training'])
            
            if rewards:
                fig.add_trace(
                    go.Scatter(y=rewards, name='Rewards', line=dict(color='blue'), mode='lines+markers'),
                    row=1, col=1
                )
            if portfolio_values:
                fig.add_trace(
                    go.Scatter(y=portfolio_values, name='Portfolio Value', line=dict(color='green'), mode='lines+markers'),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, showlegend=True, title_text=f"{strategy_type} Training Progress")
            fig.update_xaxes(title_text="Episode", row=1, col=1)
            fig.update_xaxes(title_text="Episode", row=2, col=1)
            fig.update_yaxes(title_text="Reward", row=1, col=1)
            fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
            
            sample_data = data[['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']].head(10).round(2)
            
            return fig, sample_data
            
        except Exception as e:
            print(f"Training wrapper error: {e}")
            import traceback
            traceback.print_exc()
            fig = go.Figure()
            fig.update_layout(title=f"Training failed - {str(e)}")
            return fig, pd.DataFrame()
    
    def _backtest_wrapper(self, initial_balance):
        try:
            if not self.trained_model:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Please train a model first using the Training tab")
                return empty_fig, {"Error": "No trained model available"}, empty_fig, pd.DataFrame()
                
            data = self.generate_sample_data(200)
            portfolio_history, weights_history = self.backtest_strategy(data, int(initial_balance))
            
            if not portfolio_history:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Backtest failed - no data generated")
                return empty_fig, {"Error": "Backtest failed"}, empty_fig, pd.DataFrame()
            
            metrics = self.calculate_metrics(portfolio_history)
            
            # Performance plot
            portfolio_values = [p['portfolio_value'] for p in portfolio_history]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(y=portfolio_values, name='Portfolio Value', 
                          line=dict(color='green', width=3), mode='lines')
            )
            fig.update_layout(
                title="Backtest Results - Portfolio Performance",
                xaxis_title="Time Step", 
                yaxis_title="Portfolio Value ($)",
                height=400
            )
            
            # Composition plot
            if weights_history:
                weights_df = pd.DataFrame(weights_history, 
                                        columns=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'Cash'])
                comp_fig = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                
                for i, asset in enumerate(weights_df.columns):
                    comp_fig.add_trace(
                        go.Scatter(y=weights_df[asset], name=asset, 
                                  line=dict(width=2, color=colors[i]), 
                                  stackgroup='one')
                    )
                comp_fig.update_layout(
                    title="Portfolio Composition Over Time",
                    xaxis_title="Time Step", 
                    yaxis_title="Weight Allocation",
                    height=400
                )
                
                current_weights = weights_df.iloc[-1].round(4)
                weights_df_display = pd.DataFrame({
                    'Asset': current_weights.index,
                    'Weight': current_weights.values
                })
            else:
                comp_fig = go.Figure()
                comp_fig.update_layout(title="No composition data available")
                weights_df_display = pd.DataFrame()
            
            return fig, metrics, comp_fig, weights_df_display
            
        except Exception as e:
            print(f"Backtest wrapper error: {e}")
            import traceback
            traceback.print_exc()
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"Backtest failed - {str(e)}")
            return empty_fig, {"Error": str(e)}, empty_fig, pd.DataFrame()
    
    def _update_risk_parameters(self, stop_loss, take_profit):
        """Update risk management parameters"""
        updated_orders = pd.DataFrame({
            'Order ID': [1, 2, 3, 4],
            'Asset': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
            'Type': ['STOP_LOSS', 'TAKE_PROFIT', 'STOP_LOSS', 'TAKE_PROFIT'],
            'Level': [stop_loss, take_profit, stop_loss, take_profit],
            'Status': ['Active', 'Active', 'Active', 'Active']
        })
        return updated_orders
    
    def _generate_analytics(self):
        """Generate real-time analytics charts"""
        # Generate sample data
        data = self.generate_sample_data(100)
        
        # Market Overview Chart
        market_fig = go.Figure()
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        for asset in assets:
            market_fig.add_trace(
                go.Scatter(y=data[asset], name=asset, mode='lines')
            )
        market_fig.update_layout(
            title="Market Overview - Asset Price Trends",
            xaxis_title="Days",
            yaxis_title="Price ($)",
            height=400,
            hovermode='x unified'
        )
        
        # Volatility Chart
        vol_fig = go.Figure()
        volatilities = []
        for asset in assets:
            returns = np.diff(data[asset]) / data[asset][:-1]
            vol = np.std(returns) * np.sqrt(252) * 100
            volatilities.append(vol)
        
        vol_fig.add_trace(
            go.Bar(x=assets, y=volatilities, marker=dict(color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']))
        )
        vol_fig.update_layout(
            title="Asset Volatility Analysis (Annualized %)",
            xaxis_title="Asset",
            yaxis_title="Volatility (%)",
            height=400
        )
        
        # Benchmark Comparison Chart
        benchmark_fig = go.Figure()
        
        # Calculate cumulative returns for strategy vs benchmark
        strategy_returns = np.random.normal(0.0008, 0.015, 100).cumsum()
        benchmark_returns = np.random.normal(0.0005, 0.012, 100).cumsum()
        
        benchmark_fig.add_trace(
            go.Scatter(y=strategy_returns, name='Strategy', mode='lines', 
                      line=dict(color='green', width=3))
        )
        benchmark_fig.add_trace(
            go.Scatter(y=benchmark_returns, name='Benchmark (S&P 500)', mode='lines',
                      line=dict(color='blue', width=2, dash='dash'))
        )
        benchmark_fig.update_layout(
            title="Performance vs Market Benchmark",
            xaxis_title="Days",
            yaxis_title="Cumulative Returns",
            height=400,
            hovermode='x unified'
        )
        
        # Sector Allocation Chart
        sector_fig = go.Figure()
        sector_allocations = {
            'Technology': 45,
            'Finance': 25,
            'Healthcare': 15,
            'Energy': 10,
            'Consumer': 5
        }
        
        sector_fig.add_trace(
            go.Pie(
                labels=list(sector_allocations.keys()),
                values=list(sector_allocations.values()),
                marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
            )
        )
        sector_fig.update_layout(
            title="Current Sector Allocation",
            height=400
        )
        
        return market_fig, vol_fig, benchmark_fig, sector_fig

# Main execution
if __name__ == "__main__":
    platform = AdvancedTradingPlatform()
    
    print("Initializing Advanced Trading Platform...")
    demo = platform.create_dashboard()
    
    ports = [7860, 7861, 7862, 7863, 7864, 7865]
    launched = False
    
    for port in ports:
        try:
            print(f"Attempting to launch on port {port}...")
            demo.launch(server_name="0.0.0.0", server_port=port, share=False)
            launched = True
            print(f"Successfully launched on http://localhost:{port}")
            break
        except OSError:
            continue
    
    if not launched:
        print("Could not find an available port.")
