import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
import mpl_toolkits.mplot3d as axe3d
import pandas as pd
from scipy.stats import norm

# use class for BSM model calculation and implementation 

def black_scholes(S, K, T, r, sigma, option_type):

    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2( d1, T, sigma)

    if option_type == "Call":
        return ((S*norm.cdf(d1)) - ((K*np.exp(-r*T)) * (norm.cdf(d2))))
    elif option_type == "Put":
        return ((K*np.exp(-r*T)) * (norm.cdf(-d2))) - ((S*norm.cdf(-d1)))
    else:
        raise ValueError("Invalid Input(s)")


def calculate_d1(S, K, T, r, sigma):

    S  = np.array(S, dtype=np.float64)
    if np.all(T) < 0:
        raise ValueError("Time has to be positive")
    if np.all(T) == 0 or  np.all(sigma) == 0 or np.all(K) == 0:
        raise ValueError("Time to maturity (T), strike price (K) and volatility (σ) cannot be ZERO. Please validate input.")
    return ((np.log(S / K)) + (r + (sigma**2)*0.5)*T)/(sigma*np.sqrt(T))

def calculate_d2( d1, T, sigma):

    if np.all(d1):
        return d1 - (sigma*np.sqrt(T))
    else:
        raise ValueError("The function is defined as calculate_d2( d1, T, sigma), you are seeing this error because the 'd1' parameter is missing.")

def calculate_greeks(S, K, T, r, sigma, option_type):

    S = np.array(S, dtype=np.float64)
    T = np.array(T, dtype=np.float64)

    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2( d1, T, sigma)

    # DELTA
    if option_type == "Call":
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)

    #GAMMA
    gamma = (norm.pdf(d1) /(S * sigma * np.sqrt(T)))

    #VEGA
    vega = (norm.pdf(d1) * S * np.sqrt(T)) / 100

    #THETA
    if option_type == "Call":
        theta = (-S * norm.pdf(d1)*sigma) / ((2 *np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1)*sigma) / ((2 *np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    #RHO
    if option_type == "Call":
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
    else:
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100

    return {"Name":["Delta", "Gamma","Vega","Theta","Rho"], "Value":[ delta , gamma, vega, theta, rho]}


# basic setup for page
def main():
    st.set_page_config(page_title="BSM model", page_icon="💲", layout="wide")
    st.title("Black-Scholes-Merton model for Option Pricing" )
    st.subheader("👋About the app:", divider="rainbow")
    st.text(" ")
    st.markdown('''

    This is a simple yet powerful **option pricing calculator** based on the **Black-Scholes model**.
    It allows users to:
    - Compute European call and put option prices
    - Analyze key **Greeks**:
        - Delta (Δ): Measures how much the option price changes with a $1 change in the underlying asset.
        - Gamma (Γ): Measures how much Delta changes with a $1 change in the underlying asset.
        - Vega (V): Measures how much the option price changes with a 1 percent change in implied volatility.
        - Theta (Θ): Measures how much the option price decreases per day as expiration approaches.
        - Rho (ρ): Measures how much the option price changes with a 1 percent change in the risk-free interest rate.
    - Visualize how these values evolve with changes in stock price and time to maturity


    Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/patelshlok)
    I’d love to hear your feedback.

    ''')
    st.text(" ")
    st.subheader("✌️Regarding the inputs:", divider="rainbow")
    col3, col4 = st.columns(2)
    with col3:
        st.text(" ")
        st.markdown("""
                    - Uses yfinance API to get the last equity market price.
                    - The slider generates upper and lower bounds for stock prices. If no value is assigned, the graph will not generated.
                    """)
    with col4:
        st.text(" ")
        st.markdown("""
                    - Please ensure that the ticker, strike price, dates, risk-free rate and volatility are valid; otherwise, it may raise a ValueError. And if any input is adjusted or changed, the app will re-run.
                    """)

    # user inputs
    with st.sidebar:
        st.text("Use desktop, for best results.😅")
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Enter Stock Ticker: ", value="AAPL")
            start_date = st.date_input("Start date:", value="today")
        with col2:
            option_type = st.selectbox("Option Type:",("Call", "Put"))
            end_date = st.date_input("Expiration date:", value="today")
        time_to_maturity = float(((end_date - start_date).days)/365)
        strike_price = float(st.number_input("Entre the Strike Price (K): ", value=0.00))
        col3, col4 = st.columns(2)
        with col3:
            risk_free_rate =(float(st.number_input("Risk-free Rate (as %): ", value=0.00))/ 100)
        with col4:
            volatility = (float(st.number_input("Volatility (as %): ", value=0.00)) / 100)
        slider = float(st.slider("Range of Stock Price (as %):", step=5) / 100)
        execute_op = st.button("Calculate", use_container_width=True)



    # fetch data with yfinance
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period="1d")
    if stock_data.empty:
        st.error("Data related to ticker not found. Please verify the inputs")
    else:
        latest_price = (stock_data["Close"].iloc[-1])


    if latest_price and execute_op:

    # return the findings of yfinance along with user input for data validation
        st.subheader("☺️Summary of the inputs:", divider="rainbow")
        data = {"Name" : ["Ticker","Current Price", "Strike Price (K)", "Time till maturity (T) in years", "Risk-free Rate (r)", "Volatility (σ)", "Option Type","Range of Stock Price (as %)"],
                "Value":[ticker, f"{latest_price:.2f}", f"{strike_price:.2f}", f"{time_to_maturity:.4f}", f"{risk_free_rate:.4f}", f"{volatility:.4f}", option_type, slider*100]}
        df = pd.DataFrame(data=data, index=[1,2,3,4,5,6,7,8])
        st.dataframe(df, hide_index=True)


    # first find the price of the option
        option_price = black_scholes(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)

    # second find the greeks
        greeks = calculate_greeks(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)

    # Present that important data to user:
        st.subheader("😁Required Output:", divider="rainbow")
        col_1 , col_2  = st.columns(2)
        with col_1:
            st.metric(label=f"{option_type} option price is:", value=f'{option_price:.2f}',  label_visibility="visible", border=True)
        with col_2:
            st.metric(label="Delta (Δ):",value=f"{(greeks["Value"][0]):.4f}", border=True)

        col_3 , col_4  = st.columns(2)
        with col_3:
            st.metric(label="Gamma (Γ):",value=f"{(greeks["Value"][1]):.4f}", border=True)
        with col_4:
            st.metric(label="Vega (V):",value=f"{(greeks["Value"][2]):.4f}", border=True)

        col_5 , col_6  = st.columns(2)
        with col_5:
            st.metric(label="Theta (Θ):",value=f"{(greeks["Value"][3]):.4f}", border=True)
        with col_6:
            st.metric(label="Rho (ρ):",value=f"{(greeks["Value"][4]):3f}", border=True)

        if slider:
        # graph of Greeks vs Stock Price
            st.subheader("💲Greeks vs Stock Price", divider="rainbow")
            Price_range = np.linspace(latest_price*(1 - slider), latest_price*(1 + slider), 150)
            delta_values = [float(calculate_greeks(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Value"][0]) for latest_price in Price_range]
            gamma_values = [float(calculate_greeks(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Value"][1]) for latest_price in Price_range]
            vega_values = [float(calculate_greeks(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Value"][2]) for latest_price in Price_range]
            theta_values = [float(calculate_greeks(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Value"][3]) for latest_price in Price_range]
            rho_values = [float(calculate_greeks(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Value"][4]) for latest_price in Price_range]

            plt.style.use("seaborn-v0_8-darkgrid")
            figure, axis = plt.subplots(figsize=(10,6))
            axis.plot(Price_range, delta_values, label='Delta', color='#1f77b4',lw=2, alpha=0.9 )
            axis.plot(Price_range, gamma_values, label='Gamma',color='#2ca02c', lw=2, alpha=0.9)
            axis.plot(Price_range, vega_values, label='Vega',color='#ff7f0e',lw=2, alpha=0.9 )
            axis.plot(Price_range, theta_values, label='Theta',color='#d62728',lw=2, alpha=0.9 )
            axis.plot(Price_range, rho_values, label='Rho',color='#9467bd',lw=2, alpha=0.9 )

            axis.axvline(x=strike_price, color="black", linestyle="-.", lw=1, alpha=0.7)
            axis.yaxis.set_major_locator(plt_ticker.AutoLocator())
            axis.set_facecolor("white")

            axis.set_xlabel("Stock Price", fontsize=12, fontweight="bold")
            axis.set_ylabel("Greeks",fontsize=12, fontweight="bold")
            axis.set_title("Option Greeks Sensitivity",fontsize=14,pad=20, fontweight="bold")
            axis.legend(frameon=True,framealpha=0.9, shadow=False,facecolor="white")

            axis.grid( True, alpha=0.3, linestyle="--" ,color="gray")
            for spine in ["top", "right"]:
                axis.spines[spine].set_visible(True)
                axis.spines[spine].set_color("#333333")

            axis.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

            st.pyplot(figure)

        # 3D plot: Option Price vs Stock Price over Time
            st.subheader("💸3D plot: Option Price vs Stock Price over Time" , divider="rainbow")

            Time_value =np.linspace(0.01,time_to_maturity,200)
            Price_grid ,Time_grid = np.meshgrid(Price_range, Time_value)
            chart_output = black_scholes(Price_grid, strike_price, Time_grid, risk_free_rate, volatility, option_type)

            figure3d = plt.figure(figsize=(10,8), dpi=100)
            axis3d = figure3d.add_subplot(111, projection="3d")
            surface_plot = axis3d.plot_surface(Price_grid, Time_grid, chart_output, cmap="viridis", edgecolor='none', alpha=0.9, rstride=1, cstride=1, antialiased=True, shade=True)

            cbar = figure3d.colorbar(surface_plot,shrink=0.6, aspect=20, location='left',pad=0.05)
            cbar.ax.yaxis.set_label_position("right")
            cbar.set_label("Option Price",fontsize=12, fontweight="bold", rotation= 90)

            axis3d.set_facecolor("white")
            axis3d.view_init(elev= 25, azim=45)

            axis3d.set_xlabel("Stock Price", fontsize=12, fontweight="bold")
            axis3d.set_ylabel("Time till Maturity (in years)", fontsize=12, fontweight="bold")
            axis3d.set_title("3D plot: Option Price vs Stock Price over Time", fontsize=14, fontweight="bold")
            st.pyplot(figure3d)

    else:
        st.text(" ")
        st.warning("It looks like the app isn't running at the moment, please enter your inputs to get things going. And if this warning keeps popping up or remains,"\
                    " feel free to give it another try later. If you need help, don't hesitate to reach out on LinkedIn!")


if __name__ == "__main__":
    main()



