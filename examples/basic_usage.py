"""
Basic usage example for the Algorithmic Trading Platform
"""

from trading_platform import AdvancedTradingPlatform

def main():
    """Demo of the trading platform functionality"""
    print("🚀 Algorithmic Trading Platform Demo")
    
    # Initialize platform
    platform = AdvancedTradingPlatform()
    
    # Generate sample data
    data = platform.generate_sample_data(100)
    print(f"📊 Generated market data for {len(data)} days")
    
    # Train a model
    rewards, portfolio_values = platform.train_model(data, strategy_type='PPO', episodes=10)
    print(f"🤖 Model trained with {len(rewards)} episodes")
    
    # Run backtest
    portfolio_history, weights_history = platform.backtest_strategy(data)
    print(f"📈 Backtest completed with {len(portfolio_history)} steps")
    
    print("✅ Demo completed! Launch the web interface to explore all features.")

if __name__ == "__main__":
    main()
