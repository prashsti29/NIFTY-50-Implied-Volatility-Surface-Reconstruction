readme

# NIFTY 50 Implied Volatility Surface Reconstruction

An institutional-grade, deterministic pricing engine for reconstructing missing Implied Volatility (IV) data across the NIFTY 50 options chain.

## Quantitative Philosophy: The Pure Strike-Linear Approach

In high-frequency options pricing, complex machine learning models (like Random Forests) and high-degree polynomials (like Cubic Splines) often suffer from overfitting and "Runge's phenomenon" (violent mathematical wobbling) when faced with noisy, near-expiry data. 

This repository discards black-box complexity in favor of **Raw Market Microstructure**. By relying 100% on piecewise linear interpolation in strike space ($K$-space), this engine assumes that the truest indicator of an option's volatility is the exact pricing of its immediate neighbors at that precise millisecond. 

### Key Architecture
1. **100% Cross-Sectional Reliance:** Eliminates the "temporal drag" of using historical time-series data. The model reacts instantly to underlying spot price shocks without lagging.
2. **Robust Extrapolation Flooring:** Deep Out-Of-The-Money (OTM) wings are extrapolated linearly but strictly floored at 0.5% IV to prevent impossible negative pricing and eliminate Butterfly Arbitrage.
3. **Emergency Temporal Fallback:** Time-axis interpolation is strictly reserved as an emergency net for complete cross-sectional liquidity failures (sparse rows with < 2 observed points).

## Performance Advantages
* **Computational Efficiency:** Executes the entire dataset in a fraction of the time required by iterative solvers or ML tree-builders.
* **Immunity to Overfitting:** The deterministic nature of the math ensures stable out-of-sample performance.
* **Preserves the Expiry Cliff:** Captures the violent "V" shape of 0-DTE (Zero Days to Expiry) options perfectly, whereas smooth splines artificially flatten the risk.

## Usage

### Installation
Ensure you have Python 3.8+ installed. Install the minimal dependencies:
```bash
pip install -r requirements.txt
m