# 페어 트레이딩 강화학습 MDP 환경
MDP와 뉴스 감성 통합 기능을 이용하여 트레이딩 전략을 구형하여 제 3자가 모든 결과를 재현할 수 있도록 합니다.

## 소프트웨어 및 하드웨어 요구사항

-소프트웨어-
 - Python 3.8 이상  
  - gymnasium ≥ 0.28  
  - numpy, pandas  
  - yfinance
  - nltk (VADER 감성 분석)  
  - beautifulsoup4 (웹 스크래핑)  
  - stable-baselines3 (DQN 구현)  
  - matplotlib, seaborn (평가 시각화)

## 데이터 소스
- 가격 데이터: Yahoo Financ
- 뉴스 헤드라인 yfinance.Ticker("Company").news
- 감성 모델: NLTK VADER (https://www.nltk.org/_modules/nltk/sentiment/vader.html)

## 목표
왜 페어 트레이딩인가?
 두 상관 자산 간 스프레드의 평균회귀를 이용한 시장 중립 전략 (https://arxiv.org/pdf/2407.16103)
 
왜 강화학습 + 뉴스 감성인가?
 전통적 임계치 규칙은 실간 뉴스 변동을 반영하지 않음. 뉴스 감성을 할인 인자나 보상에 반영하여 시장 환경 변화에 적응하도록 함.

목표:
 1.  MDP 환경 구현
 2.  뉴스 감성 기반 동적 할인 인자 통합
 3.  데이터들을 이용해 학습시켜 리스크 조정 수익 최대화
  
## 주요 성과
1. 환경 구현: `PairTradingEnv`  
   - **상태**: 8차원 벡터 (spread, MA, STD, Z-score, price, diff_score, MCD_closed_price, YUM_closed_price)  
   - **행동**: {0: 보유, 1: 롱, 2: 숏}  
   - **보상**: 거래 비용 차감 후 P&L, 감성 조정 γ 적용
  
2. 감성 통합:
   y_t = t_0 + a(=0.15)
   
4. 백 테스트: 2020/01/02
   - 누적수익: $199.9

## 성과 측정 지표:
누적수익








