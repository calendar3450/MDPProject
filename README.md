# Pair Trading Reinforcement Learning MDP Environment
Use an MDP with integrated news sentiment to build a trading strategy so that a third party can reproduce all results.

## Software Requirements

-Software-
 - Python 3.8 +  
  - gymnasium ≥ 0.28  
  - numpy, pandas  
  - yfinance
  - nltk (VADER sentiment analysis)  
  - beautifulsoup4 (web scraping)  
  - stable-baselines3 (DQN implementation)  
  - matplotlib, seaborn (evaluation visualization)
  - DQN

## Data Sources
- Price data: Yahoo Financ
- News headlines yfinance.Ticker("Company").news
- Sentiment model: NLTK VADER (https://www.nltk.org/_modules/nltk/sentiment/vader.html)

## Step
 1. Identify candidate pairs. For each pair under consideration, compute their cointegration coefficient (using the Yahoo API stock price notebook).
 2. Check if the pair is suitable for trading by ensuring the cointegration coefficient ≤ 0.15.
 3. In the MDPModel.ipynb notebook, aggregate the news scores and price data and feed them into the model.
 4. Save the model so it can be applied to other stocks.

## Goal
Why pair trading?
 A market‑neutral strategy using mean reversion of the spread between two correlated assets. (https://arxiv.org/pdf/2407.16103)
 
Why RL + news sentiment?
 Traditional threshold rules don’t reflect real‑time news changes. By incorporating news sentiment into the discount factor or reward, the strategy can adapt to changing market conditions.

Goals:
 1.  Implement the MDP environment
 2.  Integrate dynamic discount factors based on news sentiment
 3.  Train on data to maximize risk‑adjusted returns
  
## Key Achievements
1. Environment implementation: `PairTradingEnv`  
   - **State**: 8‑dimensional vector (spread, MA, STD, Z-score, price, diff_score, MCD_closed_price, YUM_closed_price)  
   - **Action**: {0: Hold, 1: Long, 2: Short}  
   - **Reward**: P&L after transaction costs, adjusted by sentiment‑weighted  R_t = (PnL_t - c) * γ_t; γ_t = γ_0 + a *sent_t
  
2. Sentiment integration::
   y_t = y_0 + a(Compound score average)
   
4. Back Test: 2020/01/02
   - Total reward: $199.9

## Performance Metric:
  - Total Dollar.

## Problem Definition
Automate an MDP that takes long, short, or hold actions on the spread of two stocks at each time step to control risk and maximize returns, adjusting dynamically to live news sentiment.

## Related Work and References
 1. Optimizing the Pairs-Trading Strategy Using Deep Reinforcement Learning with Trading and Stop-Loss Boundaries (https://onlinelibrary.wiley.com/doi/10.1155/2019/3582516)
 2. Pairs Trading with Robust Kalman Filter and Hidden Markov Model (https://medium.com/@kaichong.wang/statistical-arbitrage-1-pairs-trading-with-robust-kalman-filter-and-hidden-markov-model-62d0a1a0e4ae)
 3. Deep Reinforcement Learning Pairs Trading  (https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2447&context=gradreports)
 4. Pair trading Kyobo life
 5. What hedge funds use? A review of pair‑trading strategies (https://insight.stockplus.com/articles/5336)
 6. Learning U.S. Financial Engineering at 44 (https://wikidocs.net/book/4978)

## 3.State, Action, Transition, and Observation
- State: s_t ∈ ℝ^8
- Action: {0,1,2}
- Transition:  Next state determined by new prices and market stochasticity
- Observation: State vector provided at each step (O(s_t)=s_t)

## 4.Solution Method

 Using Data:
 - Policy: a_t = argmax_a Q(s_t,a)
 - Reward: R_t = (PnL_t - c) * γ_t; γ_t = γ_0 + a *sent_t
 - Loss: MSE between predicted Q‑value and target Q‑value


## 5. Implementation Details
 1. Environment (`PairTradingEnv`)
    - action_space = Discrete(3)
    - `reset()`, `step(a)`, `render()` Implement
      
 2. Data Preprocessing:
    - python
  spread_MA = spread.rolling(win).mean()
  spread_STD = spread.rolling(win).std(ddof=0)
  Z_score = (spread - spread_MA) / spread_STD
  diff_score =  np.zeros_like(spread)
  MCD =raw["MCD"]
  YUM = raw["YUM"]

## 6. Resources:
1. https://www.insightbig.com/post/developing-a-profitable-pairs-trading-strategy-with-python
2. https://databento.com/blog/build-a-pairs-trading-strategy-in-python
3. https://medium.databento.com/build-a-pairs-trading-strategy-in-python-a-step-by-step-guide-dcee006e1a50?gi=5738dae53da6
4. https://medium.com/@ngao7/markov-decision-process-value-iteration-2d161d50a6ff
5. https://wire.insiderfinance.io/markov-decision-processes-mdp-ai-meets-finance-algorithms-series-7f34de5680d5
6. https://python.plainenglish.io/understanding-markov-decision-processes-17e852cd9981
7. https://www.datacamp.com/tutorial/markov-chains-python-tutorial
8. https://blog.naver.com/chunjein/100203065865
9. https://www.youtube.com/watch?v=YDMSqal-RZ4
10. https://domino.ai/blog/deep-reinforcement-learning
11. https://www.nltk.org/howto/sentiment.html
12. https://alexanderozkan.com/Sentiment-Analysis-as-a-Trading-Indicator/
13. https://newsdata.io/blog/access-yahoo-finance-news-api/
14. https://developer.yahoo.com/api/
15. https://ranaroussi.github.io/yfinance/

