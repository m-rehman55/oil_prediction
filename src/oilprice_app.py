import streamlit as st
import pandas as pd
import pickle
from src.data_handler import DataHandler
from src.predictor import Predictor

class Oil_price:
    """Manages the Streamlit app for oil price prediction."""

    def __init__(self, data_path='data/Crude Oil WTI Futures Historical Data.csv', model_path='Model/svm_model.pkl',
                 required_columns={'Open', 'High', 'Low'}):
        try:
            self.data_handler = DataHandler(data_path, required_columns)
            self.predictor = Predictor(model_path)
            self.required_columns = required_columns
        except (FileNotFoundError, ValueError, pickle.PickleError) as e:
            st.error(str(e))
            st.stop()

    def run(self):
        st.title('oil Price Prediction')
        st.subheader("Last 4 Open, High, Low Prices")
        st.table(self.data_handler.get_head(4))

        st.subheader("Enter Price Data")
        open_price = st.number_input('Open Price', min_value=0.0, step=0.01)
        high_price = st.number_input('High Price', min_value=0.0, step=0.01)
        low_price = st.number_input('Low Price', min_value=0.0, step=0.01)

        if st.button('Predict'):
            if open_price <= 0 or high_price <= 0 or low_price <= 0:
                st.error("All prices must be positive values.")
            else:
                input_df = pd.DataFrame([[open_price, high_price, low_price]],
                                        columns=['Open', 'High', 'Low'])
                try:
                    prediction = self.predictor.predict(input_df)
                    st.write(f'The predicted closing price is: {prediction:.2f}')
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
