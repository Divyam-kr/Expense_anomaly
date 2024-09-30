import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# Define category keywords for accurate classification
category_keywords = {
    'Groceries': ['grocery', 'supermarket', 'store', 'market'],
    'Entertainment': ['movie', 'cinema', 'concert', 'game', 'entertainment', 'theater', 'event'],
    'Bills': ['bill', 'utilities', 'rent', 'electricity', 'water', 'internet'],
    'Shopping': ['shop', 'clothing', 'electronics', 'fashion', 'purchase', 'mall', 'online'],
    'Dining': ['restaurant', 'dining', 'meal', 'food', 'dinner', 'lunch', 'breakfast', 'cafe', 'coffee'],
    'Travel': ['flight', 'hotel', 'travel', 'vacation', 'trip', 'airbnb', 'taxi', 'uber', 'train']
}

# Function to categorize based on keywords
def categorize_transaction(description):
    description = description.lower()  # Convert to lowercase for easier matching
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in description:
                return category
    return 'Other'  # Return 'Other' if no match is found

# Function to apply Isolation Forest and extract anomalies month-wise
def detect_anomalies_monthly(data):
    monthly_anomalies = {}
    data['Anomaly'] = None

    for period, group in data.groupby(data['Date'].dt.to_period('M')):
        # Prepare data for Isolation Forest
        model = IsolationForest(n_estimators=20, max_features=0.5, max_samples=0.75, contamination=0.03)
        group['Anomaly'] = model.fit_predict(group[['Amount']])
        
        # Extract anomalies (where Anomaly == -1)
        anomalies = group[group['Anomaly'] == -1]

        if not anomalies.empty:
            monthly_anomalies[period] = anomalies[['Date', 'Time', 'Amount', 'Description']]
    
    return monthly_anomalies

# Streamlit application
st.set_page_config(page_title="Financial Transaction Analyzer", layout="wide")

st.title("💰 Financial Transaction Analyzer")
st.write("This application helps you analyze financial transactions by detecting anomalies based on uploaded transaction data.")

# Sidebar for user input
st.sidebar.header("User Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Process uploaded file
if uploaded_file is not None:
    # Read CSV file
    transaction_dataa = pd.read_csv(uploaded_file)
    transaction_dataa['Date'] = pd.to_datetime(transaction_dataa['Date'])  # Convert 'Date' column to datetime
    
    st.subheader("")
    st.dataframe(transaction_dataa.head(), use_container_width=True)

    # Apply categorization
    transaction_data = transaction_dataa.copy()
    transaction_data['Category'] = transaction_data['Description'].apply(categorize_transaction)

    # Display categorized data
    st.subheader("Categorized Transaction Data")
    st.dataframe(transaction_data, use_container_width=True)

    # Anomaly detection
    if st.sidebar.button("Detect Anomalies"):
        st.sidebar.write("Analyzing the data for anomalies, please wait...")
        monthly_anomalies = detect_anomalies_monthly(transaction_data)
        
        # Display anomalies
        if monthly_anomalies:
            for month, anomalies in monthly_anomalies.items():
                st.subheader(f"📊 Anomalies Detected for {month}:")
                st.write(anomalies)
        else:
            st.warning("No anomalies detected in the uploaded data.")

# Footer
st.write("___")
st.write("© 2024 Financial Transaction Analyzer. All Rights Reserved.")
