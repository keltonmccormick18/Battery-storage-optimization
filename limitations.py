# For each walk-forward iteration, compute three things:
diagnostics = []

for r in results:
    start = r["eval_start"]
    week_prices = CISO["price_usd_mwh"].values[start:start + eval_window]
    
    # 1. Total price volatility (raw, before any model)
    total_vol = np.std(week_prices)
    
    # 2. Daily spread (max - min per day, averaged)
    daily_spreads = []
    for d in range(7):
        day_prices = week_prices[d * 24:(d + 1) * 24]
        daily_spreads.append(day_prices.max() - day_prices.min())
    avg_spread = np.mean(daily_spreads)
    
    r["total_vol"] = total_vol
    r["avg_spread"] = avg_spread

weeks = range(len(results))
total_vols = [r["total_vol"] for r in results]
avg_spreads = [r["avg_spread"] for r in results]
sigmas = [r["sigma"] for r in results]

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. Raw price vol vs residual sigma over time
axes[0].plot(weeks, total_vols, label="Total price std dev", alpha=0.7)
axes[0].plot(weeks, sigmas, label="OU residual σ", alpha=0.7)
axes[0].set_ylabel("$/MWh")
axes[0].set_title("Total Volatility vs Residual Volatility")
axes[0].legend()

# 2. Fraction of variance explained by seasonal model
# If sigma declines but total vol doesn't, the model is improving
variance_explained = [1 - (s**2 / tv**2) if tv > 0 else 0 
                      for s, tv in zip(sigmas, total_vols)]
axes[1].plot(weeks, variance_explained)
axes[1].set_ylabel("R² (fraction explained)")
axes[1].set_title("Seasonal Model Explanatory Power")
axes[1].axhline(y=np.mean(variance_explained), color='r', 
                linestyle='--', label=f"Mean: {np.mean(variance_explained):.2f}")
axes[1].legend()

# 3. Average daily spread over time
axes[2].plot(weeks, avg_spreads, alpha=0.7)
z = np.polyfit(weeks, avg_spreads, 1)
axes[2].plot(weeks, np.polyval(z, list(weeks)), 'r--', 
             label=f"Trend: {z[0]:+.2f} $/week")
axes[2].set_ylabel("$/MWh")
axes[2].set_xlabel("Week")
axes[2].set_title("Average Daily Price Spread (Max - Min)")
axes[2].legend()

plt.tight_layout()
plt.show()
