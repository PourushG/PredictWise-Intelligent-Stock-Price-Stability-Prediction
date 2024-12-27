import streamlit as st
import yfinance as yf
import joblib
import pandas as pd
import numpy as np

# Load the ML model
@st.cache_resource
def load_model():
    return joblib.load("C:\\Users\\pouru\\Downloads\\nifty50_model_new.pkl")

model = load_model()

# Nifty50 stock names and symbols
nifty50_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Wipro": "WIPRO.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "ITC": "ITC.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Larsen & Toubro": "LT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "Oil and Natural Gas Corporation": "ONGC.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "UPL": "UPL.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports and SEZ": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Bharat Petroleum Corporation": "BPCL.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim Industries": "GRASIM.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Indian Oil Corporation": "IOC.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Coal India": "COALINDIA.NS",
    "BPCL": "BPCL.NS",
    "Tata Power": "TATAPOWER.NS",
    "Shree Cement": "SHREECEM.NS"
}

# List of stocks always classified as stable
always_stable_stocks = [
    "Reliance Industries",
    "NTPC",
    "Adani Enterprises",
    "Tata Motors",
    "Indian Oil Corporation",
    "BPCL",
    "Nestle India",
    "HDFC Bank",
    "TCS",
    "Infosys"
]

@st.cache_data
def fetch_stock_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="5y")

        if hist.empty or len(hist) < 1:
            raise ValueError("No valid stock data found.")

        latest_data = hist.iloc[-1]
        open_price = float(latest_data["Open"])
        close_price = float(latest_data["Close"])
        high_price = float(latest_data["High"])
        low_price = float(latest_data["Low"])
        volume = float(latest_data["Volume"])
        vwap = float((hist["Close"] * hist["Volume"]).sum() / hist["Volume"].sum())
        std = float(hist["Close"].std())

        current_data = {
            "open": open_price,
            "close": close_price,
            "high": high_price,
            "low": low_price,
            "volume": volume,
            "vwap": vwap,
            "std": std,
        }
        return current_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Prepare input for model
def prepare_input(stock_data):
    return np.array([
        stock_data["open"],
        stock_data["high"],
        stock_data["low"],
        stock_data["close"],
        stock_data["volume"],
        stock_data["vwap"],
        stock_data["std"],
    ]).reshape(1, -1)


# Streamlit UI
st.title("Predictwise: Stable Stock Prediction")
st.subheader("Your Intelligent Assistant for Smarter Investment Decisions")  # Short description
st.write("""
*Predictwise leverages the power of machine learning to analyze stock data and classify stocks as **Stable** or **Unstable**. 
With data-driven insights, we aim to empower investors to make informed decisions and navigate the market confidently.
Select a stock from the Nifty50 list, fetch its current data, and see the prediction in real-time! 📈*""")

st.write("**Select a stock from the Nifty50 list to fetch its current data and predict its stability.**")

# Search box for stock name
search_query = st.text_input("Search for a stock by name:", "", placeholder="Type to search for a stock...")

if search_query:
    matching_stocks = {name: symbol for name, symbol in nifty50_stocks.items() if search_query.lower() in name.lower()}
    
    if matching_stocks:
        selected_stock = st.selectbox("Select a matching stock:", list(matching_stocks.keys()))
    else:
        st.warning("The entered stock is not listed in the Nifty50.")
        selected_stock = None
else:
    # Dropdown for stock selection if no search query is provided
    selected_stock = st.selectbox("Or select a Stock:", list(nifty50_stocks.keys()))

# Session state to track results
if "results_cleared" not in st.session_state:
    st.session_state.results_cleared = False

# Create columns for buttons
col1, col2 = st.columns(2)

# Button to clear results
with col2:
    if st.button("**Clear Results**"):
        st.session_state.results_cleared = True

# Button to get prediction
with col1:
    if st.button("**Get Prediction**") and selected_stock:
        st.session_state.results_cleared = False  # Reset results_cleared state
        stock_symbol = nifty50_stocks.get(selected_stock)
        st.write(f"Fetching data for {selected_stock}...")
        
        # Fetch stock data
        stock_data = fetch_stock_data(stock_symbol)
        if stock_data:
            st.write("### Current Stock Data")
            st.write(stock_data)

            # Prepare input and predict
            input_data = prepare_input(stock_data)
            prediction = model.predict(input_data)
            result = "Stable" if prediction[0] == 1 else "Unstable"

            # Display result
            st.write("### Prediction Result")
            if result == "Stable":
                st.success(f"The stock {selected_stock} is Stable! 🚀")
            else:
                st.error(f"The stock {selected_stock} is Unstable! 📉")
        else:
            st.error("Failed to fetch stock data. Please try again.")



# Handle clearing of results
if st.session_state.results_cleared:
    st.write("Results cleared. You can start a new prediction.")

# Footer
st.write("---")
st.markdown("""
---

### 🚀 Built with ❤️ by the Predictwise Team
**Contributors**:  
- **Pourush Gupta** (`TCA2159031`) - *ML Model Development* 🧠  
- **Karishma Saxena** (`TCA2159022`) - *Frontend Development* 🎨  
- **Shruti Jain** (`TCA2159042`) - *Market Research and Data Collection* 📊  

---

💡 *Empowering better investment decisions with AI and data science!*  

""", unsafe_allow_html=True)

