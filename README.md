
# Black-Scholes-Merton model for Option Pricing
#### Video Demo: <https://youtu.be/L3EuAxM0MTA>

### Understanding Option Pricing Through Mathematics

This project presents a Python-based implementation of the Black-Scholes-Merton (BSM) model—one of the most widely recognized analytical tools for valuing European-style options. The objective is to illustrate the theoretical foundations of option pricing through a practical and interactive application.

The model is integrated with a Streamlit-based front end, allowing users to input financial parameters and instantly compute both the option price and the associated sensitivity measures (Greeks).

Developed as the final project for the CS50 Python course, this application is strictly intended for educational use. It demonstrates essential concepts in financial mathematics, such as stochastic modeling, volatility, time value, and risk sensitivity.

---

### Key Features

#### Core Pricing Engine

- Implements the Black-Scholes-Merton formulas to compute theoretical prices of European call and put options (defined in `project.py`).

- Calculates the five major option Greeks:
  - Delta: sensitivity to changes in the underlying asset price
  - Gamma: rate of change of Delta
  - Vega: sensitivity to changes in volatility
  - Theta: sensitivity to the passage of time (time decay)
  - Rho: sensitivity to changes in the risk-free interest rate

#### Interactive Dashboard

- Streamlit-based front end (contained in `project.py`)

- Accepts user-defined inputs including:
  - Ticker symbol (e.g., AAPL, TSLA)
  - Strike price
  - Start and expiration date
  - Risk-free interest rate and implied volatility

- Automatically retrieves the latest market price using the yfinance API

- Provides:
  - A tabular summary of the inputs and calculated outputs
  - Line plots showing how the Greeks vary with stock price
  - A 3D surface plot of option prices across a range of stock prices and maturities

#### Automated Testing

- Unit tests implemented using pytest (`test_project.py`)

- Validates:
  - Correct handling of invalid inputs (e.g., zero volatility, negative time)
  - Accurate calculation of option prices and Greeks
  - Proper error handling for unsupported option types

---

### Installation:
Ensure you have Python 3.7 or higher. To set up the project environment:
```bash
pip install --update pip
pip install -r requirements.txt
```
---
### Launching the Streamlit app:
```bash
streamlit run project_app.py
```
Once launched, the interface allows you to:

- Select the stock ticker and relevant parameters

- Define time to maturity using calendar dates

- Adjust the price range sliders for custom plotting

- View option price and Greek outputs instantly
---
### Testing:
To test `project.py`, we use
```bash
pytest test_project.py
```
Make sure all test cases pass before making significant changes to the model or interface.

---
### Educational Disclaimer
This project was developed as the capstone for the CS50 Python course and is intended solely for educational and demonstrative purposes. It simplifies several real-world complexities and should not be used for actual trading or investment decisions. Always consult financial professionals before applying financial models in practice.

---
### Acknowledgements
- Harvard University’s CS50 Python curriculum for conceptual grounding.

- Yahoo Finance for market data via the `yfinance` library.

- The open-source Python and Streamlit communities for tools and documentation that made this project possible.
---
### License
This project is licensed under the MIT License. You are free to use, modify, and distribute it, but only for non-commercial and educational purposes.
