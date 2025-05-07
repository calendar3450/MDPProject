#!/usr/bin/env python
# coding: utf-8

# ## Stochastic sequential decision problem 
# Goal: The goal is to find the trading sequence for the YUM–MCD pair that maximizes cumulative dollar profit and loss.
# Stochastic: The prices of the two assets and the news sentiment scores evolve probabilistically over time. 
# Sequential: The agent must observe the state each trading day and make sequential BUY/SELL/HOLD decisions.
# 
# ## Feasible solution method
# The objective is to maximize dollar P/L through mean reversion on the YUM↔MCD spread. Since the price and news time series contain stochastic elements, it forms an MDP.
# 
# # Why compare YUM–MCD?
# Because these two stocks form a suitable pair. In the Yahoo API stock price file, you can see the method used to find such pairs. The code shows how they were identified and what is needed.

# In[399]:


import gym
from gym import spaces
import numpy as np
import pandas as pd
import nltk
import yfinance as yf, statsmodels.api as sm, pandas as pd
import yfinance as yf

from nltk.sentiment import SentimentIntensityAnalyzer


# In[370]:


sia = SentimentIntensityAnalyzer()

# Preparation to obtain YUM’s news via the Yahoo API and compute corresponding sentiment scores.
ticker = yf.Ticker("YUM")
summaryList=[] # Store news headlines.
for article in ticker.news:
    summaryList.append(article["content"]["summary"])
    
totalScore = [] # Store the scores of the collected headlines.

for i in summaryList:
    sentiment_scores = sia.polarity_scores(i)
    totalScore.append(sentiment_scores)
    
compoundScore = [s['compound'] for s in totalScore]
avgCoundScoreYUM = sum(compoundScore)/len(compoundScore)
print(compoundScore)
print(round(avgCoundScoreYUM,4)) # Compute the average score to indicate whether the current news is positive or negative. Range: (–1 to +1).


# In[372]:


# Other company news
ticker = yf.Ticker("MCD")
summaryList=[]
for article in ticker.news:
    summaryList.append(article["content"]["summary"])
    
totalScore = []

for i in summaryList:
    sentiment_scores = sia.polarity_scores(i)
    totalScore.append(sentiment_scores)
    
compoundScore = [s['compound'] for s in totalScore]
avgCoundScoreMCD = sum(compoundScore)/len(compoundScore)

print(compoundScore)
print(round(avgCoundScoreMCD,4))# Compute the average score to indicate whether the current news is positive or negative. Range: (–1 to +1).


# In[374]:


#  5-years
tickers   = ["YUM", "MCD"]
start_day = "2020-01-01"

raw = yf.download(tickers, start=start_day, progress=False)["Close"]

# log spread = ln(P_YUM) − ln(P_MCD)
log_price = np.log(raw)
spread    = log_price["YUM"] - log_price["MCD"]

# 3. Moving average, STD
win = 30
spread_MA = spread.rolling(win).mean()
spread_STD = spread.rolling(win).std(ddof=0)
Z_score = (spread - spread_MA) / spread_STD
diff_score =  np.zeros_like(spread)
MCD =raw["MCD"]
YUM = raw["YUM"]

# 4. DataFrame
df = pd.DataFrame({ # State 
    "spread"    : spread, # You can directly see by what percentage YUM is more expensive than MCD.
    "spread_MA" : spread_MA, # Recent Average Why?: Indicates the ‘normal (mean) position’ in a mean-reversion strategy. Simply using the overall historical mean reacts too slowly when the time series shifts, so we use a rolling mean instead.
    "spread_STD": spread_STD, # Volatility (σ) over the same window—how much does it fluctuate. Why?: Provides a scale reference to judge whether Fred’s ±5¢ move is ‘large’ or ‘small
    "Z_score"   : Z_score, # With thresholds like ±2σ, you can easily define Long/Short entry and exit rules.  A deep RL model can also instantly perceive the ‘normalized distance’ using only the Z_score.
    "price"     : spread, # Log spread +: P_YUM is relatively more expensive than P_MCD. - P_YUM is relatively cheaper than P_MCD. price = 0.1 => e^0.1 = 1.105 110% more expensive 
    # When Long (1), if the spread narrows → profit
    # When Short (2), if the spread widens → profit
    "diff_score": diff_score, # News score. Why are all of data same? we don't need past score.
    "MCD_closed_price": MCD,
    "YUM_closed_price": YUM
}).dropna()          # Remove NAN.

latest_idx = df.index[-1]
df.at[latest_idx, "diff_score"] = ( round(avgCoundScoreYUM, 4) - round(avgCoundScoreMCD, 4))


# In[376]:


print(df)


# ## Find Beta(The variables needed to convert the log-transformed values back into dollar terms. ):
# Using this beta value, we adjust the portfolio by selling or buying MCD shares.

# In[390]:


# Remove any missing values.
clean = (
    raw.replace([np.inf, -np.inf], np.nan)
        .dropna()                          
)

X = sm.add_constant(clean["MCD"]) #Transform into 2D by adding a constant term.
model = sm.OLS(clean["YUM"], X).fit() # Fit an OLS (Ordinary Least Squares) regression.
beta  = round(model.params["MCD"], 4)
print("β =", beta)

pair_price = clean["YUM"] - beta * clean["MCD"] # Formula to convert back into real dollar terms.
print(pair_price.head())


# In[437]:


class PairTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    # Policy 
    def sample_policy(self):
        # Must use News score.
        row = self.data.iloc[self.current_step]
        z = float(self.data.iloc[self.current_step]['Z_score'])
        diff_score = float(self.data.iloc[self.current_step]['diff_score'])
        if diff_score == 0: # News scores start from the current time point
            if z >  1.0: # YUM has become relatively expensive compared to MCD.
                return 2          # Short (sell YUM)
            elif z < -1.0 : # YUM has become relatively cheap.
                return 1          # Long (sell MCD)
            else:
                return 0          # Hold
        else:
            if z >  1.0 and diff_score <= -0.1: 
                return 2        
            elif z < -1.0 and diff_score >= 0.1 : 
                return 1          
            else:
                return 0 
    
    def __init__(self, data,beta = beta):
        """
        data: pandas DataFrame
            - 'spread'     : Log price spread
            - 'spread_MA'  : Moving Average
            - 'spread_STD' : STD of spre
            - 'Z_score'    : Z-score (spread - MA) / STD)
            - 'price'      : pair price
        """
        super(PairTradingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.n_steps = len(self.data)
        self.current_step = 0
        self.beta =beta
        # action: 0-hold, 1-Long, 2-Short
        self.action_space = spaces.Discrete(3)
        
        # [spread, spread_MA, spread_STD, Z_score, price]
        low = -np.inf * np.ones(8)
        high = np.inf * np.ones(8)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Current Position: 0-hold, 1-Long, 2-Short
        self.position = 0
        self.entry_price = 0

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.data.iloc[self.current_step][['spread', 
                                                 'spread_MA', 
                                                 'spread_STD', 
                                                 'Z_score', 
                                                 'price',
                                                 'diff_score',
                                                 'MCD_closed_price',
                                                 'YUM_closed_price'
                                                ]].values
        return obs.astype(np.float32)
    
    def step(self, action):
        """
        Process one step. 
        action: int, {0: hold, 1: Long, 2: Short}
        """
        done = False
        reward = 0.0
        dollar_reward = 0.0
        info = {}
        row_now  = self.data.iloc[self.current_step]
        row_prev = self.data.iloc[self.current_step - 1]
        delta_real_price = ((row_now["YUM_closed_price"]- row_prev["YUM_closed_price"]) - self.beta*(row_now['MCD_closed_price']-row_prev['MCD_closed_price']))

        # Move next step
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True
        
        current_price = self.data.iloc[self.current_step]['price']
        
        if action == 1:  # Long position.
            if self.position < 0:
                reward += (self.entry_price - current_price)  # Reward using short.
                dollar_reward += (-delta_real_price)
                self.position = 0

            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
        elif action == 2:  # Short Position.
            if self.position > 0:
                reward += (current_price - self.entry_price)  # Reward using Long.
                dollar_reward += (delta_real_price)
                
                self.position = 0
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
        else:  # 0: Hold position
            if self.position != 0:
                dollar_reward += (-delta_real_price*(-self.position))
                if self.position == 1:
                    reward += (current_price - self.entry_price)
                    
                else:
                    reward += (self.entry_price - current_price)
                self.position = 0
                self.entry_price = 0
        
        # Update State
        obs = self._next_observation()
        
        info = {"dollar_reward": dollar_reward}
        
        return obs, reward, done, info    
        
    def render(self, mode='human', close=False):
        yum_stock = self.data.iloc[self.current_step]['YUM_closed_price']
        mcd_stock = self.data.iloc[self.current_step]['MCD_closed_price']
        pp = self.data.iloc[self.current_step]['price']

    
        action_txt = { 1: f"BUY 1 YUM & SELL {self.beta:.2f} MCD",
                      -1: f"SELL 1 YUM & BUY  {self.beta:.2f} MCD",
                       0: "HOLD"}[self.position]
    
        print(f"Step {self.current_step:3d} | Position: {self.position} | YUM ${yum_stock:,.2f} | MCD ${mcd_stock:,.2f} "
              f"| Pair ${pp:,.2f} | {action_txt:24s} ")


# In[463]:


# Class pairTradingEnv using dataFrame = df
env = PairTradingEnv(df)

state = env.reset()
print("Initial State:", state)
print("Beta: ",beta)

done = False
total_reward = 0
total_dollar = 0
while not done:
    action = env.sample_policy() #Determine the action using the current policy.
    state, reward, done,info = env.step(action)
    dollar_reward = info.get("dollar_reward", 0.0)
    total_reward += reward
    total_dollar +=dollar_reward
    print(total_dollar)
    print('====================================================================================================')
    env.render()


# In[465]:


print("Total Reward:", total_reward)
print(f"Total Dollar: ${round(total_dollar,3)}")


# In[409]:


from stable_baselines3 import DQN


# In[449]:


# env: PairTradingEnv instance
model = DQN(
    policy="MlpPolicy",      # 다층 퍼셉트론 정책 네트워크
    env=env,                 # 학습에 사용할 환경
    learning_rate=1e-4,      # 학습률
    buffer_size=10000,       # 리플레이 버퍼 크기
    learning_starts=1000,    # 학습 시작 전 최소 스텝
    target_update_interval=500, 
    gamma=0.99,              # 기본 할인 인자 (감성 반영 시 동적으로 변경 가능)
    verbose=1
)


# In[469]:


total_timesteps = 200000
model.learn(total_timesteps=total_timesteps)


# In[471]:


model.save("dqn_pairtrading")
# When we reuse this model.
model = DQN.load("dqn_pairtrading", env=env)


# In[474]:


obs = env.reset()
done = False
total_reward = 0.0
total_dollar = 0.0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    dollar_reward = info.get("dollar_reward", 0.0)
    total_reward += reward
    total_dollar  += dollar_reward
    env.render()
    print('-',total_dollar)


# In[476]:


print(total_dollar)

