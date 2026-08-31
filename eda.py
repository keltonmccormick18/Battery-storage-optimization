import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.optimize import minimize
from scipy.stats import norm

CISO = pd.read_csv('~/Desktop/energy-storage-control/data/prices_CISO.csv')
NYISO = pd.read_csv('~/Desktop/energy-storage-control/data/prices_NYIS.csv')

print(CISO.head())
print(NYISO.head())

CISO["hour"] = pd.to_datetime(CISO["hour"])
CISO["date"] = CISO["hour"].dt.date


daily_avg_ciso = CISO.groupby("date")['price_usd_mwh'].mean()
daily_avg_ciso.plot()
plt.show()

NYISO["hour"] = pd.to_datetime(NYISO["hour"])
NYISO["date"] = NYISO["hour"].dt.date
daily_avg_nyiso = NYISO.groupby("date")["price_usd_mwh"].mean()
daily_avg_nyiso.plot()
plt.show()
                                                    


#summary statistics
#CISO

CISO["price_usd_mwh"].describe()
NYISO["price_usd_mwh"].describe()


CISO["price_usd.mwh"].kurt()
CISO["price_usd.mwh"].skew()
CISO["price_usd.mwh"].median()
CISO["price_usd_mwh"].hist(bins=100)
plt.yscale("log")
plt.show()

CISO["avghour"] = CISO["hour"].dt.hour
perhour = CISO.groupby("avghour")["price_usd_mwh"].mean()
perhour.plot()
plt.show()

#NYISO


NYISO["price_usd.mwh"].kurt()
NYISO["price_usd.mwh"].skew()
NYISO["price_usd.mwh"].median()
NYISO["price_usd_mwh"].hist(bins=100)
plt.yscale("log")
plt.show()

NYISO["avghour"] = NYISO["hour"].dt.hour
perhour = NYISO.groupby("avghour")["price_usd_mwh"].mean()
perhour.plot()
plt.show()



#create features 

CISO["hour"] = pd.to_datetime(CISO["hour"])
CISO["hour_of_day"] = CISO["hour"].dt.hour
CISO["dow"] = CISO["hour"].dt.dayofweek
CISO["month"] = CISO["hour"].dt.month

NYISO["hour"] = pd.to_datetime(NYISO["hour"])
NYISO["hour_of_day"] = NYISO["hour"].dt.hour
NYISO["dow"] = NYISO["hour"].dt.dayofweek
NUISO["month"] = NYISO["hour"].dt.month



