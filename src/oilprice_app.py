import streamlit as st
import pandas as pd
import pickle
from src.data_handler import DataHandler
from src.predictor import Predictor

# Inject custom CSS
def load_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

            html, body, [class*="css"] {
                font-family: 'Poppins', sans-serif;
                background-color: #f1f3f6;
            }

            .main-title {
                font-size: 48px;
                font-weight: 700;
                background: -webkit-linear-gradient(45deg, #0072ff, #00c6ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }

            .section-title {
                font-size: 24px;
                font-weight: 600;
                color: #333333;
                margin-top: 40px;
                margin-bottom: 10px;
            }

            .card {
                background-color: #ffffff;
                padding: 25px;
                border-radius: 18px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.06);
                transition: 0.3s ease-in-out;
            }

            .card:hover {
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }

            .predict-btn button {
                background: linear-gradient(90deg, #0072ff, #00c6ff) !important;
                color: white !important;
                font-weight: bold;
                border-radius: 10px;
                padding: 0.6em 1.2em;
                margin-top: 20px;
                width: 100%;
            }

            .result-box {
                background-color: #e0f7fa;
                color: #006064;
                font-weight: bold;
                padding: 15px;
                border-radius: 12px;
                font-size: 18px;
                margin-top: 20px;
                text-align: center;
            }

            table {
                border-radius: 12px !important;
                overflow: hidden !important;
            }
        </style>
    """, unsafe_allow_html=True)

class Oil_price:
    """Manages the Streamlit app for oil price prediction."""

    def __init__(self, data_path='data/Crude Oil WTI Futures Historical Data.csv',
                 model_path='Model/svm_model.pkl',
                 required_columns={'Open', 'High', 'Low'}):
        try:
            self.data_handler = DataHandler(data_path, required_columns)
            self.predictor = Predictor(model_path)
            self.required_columns = required_columns
        except (FileNotFoundError, ValueError, pickle.PickleError) as e:
            st.error(str(e))
            st.stop()

    def run(self):
        load_custom_css()

        st.markdown('<h1 class="main-title">🛢️ Oil Price Predictor</h1>', unsafe_allow_html=True)

        st.markdown('<h3 class="section-title">📊 Recent Market Snapshot</h3>', unsafe_allow_html=True)
        st.container().markdown('<div class="card">', unsafe_allow_html=True)
        st.table(self.data_handler.get_head(4))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<h3 class="section-title">🧾 Enter Today\'s Price Data</h3>', unsafe_allow_html=True)

        with st.form("prediction_form"):
            st.markdown('<div class="card">', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                open_price = st.number_input('Open Price', min_value=0.0, step=0.01, format="%.2f")
            with col2:
                high_price = st.number_input('High Price', min_value=0.0, step=0.01, format="%.2f")
            with col3:
                low_price = st.number_input('Low Price', min_value=0.0, step=0.01, format="%.2f")

            st.markdown('</div>', unsafe_allow_html=True)

            submitted = st.form_submit_button("🔮 Predict", type="primary")
        
        if submitted:
            if open_price <= 0 or high_price <= 0 or low_price <= 0:
                st.error("❗ All prices must be greater than zero.")
            else:
                input_df = pd.DataFrame([[open_price, high_price, low_price]],
                                        columns=['Open', 'High', 'Low'])
                try:
                    prediction = self.predictor.predict(input_df)
                    st.markdown(f'<div class="result-box">📈 Predicted Closing Price: <br> <span style="font-size: 28px;">${prediction:.2f}</span></div>',
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
