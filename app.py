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

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

def feature_engineering(df):
    df = df.copy()
    # Encode category
    le = LabelEncoder()
    df['Category_enc'] = le.fit_transform(df['Category'])
    # Extract time features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    return df

def detect_anomalies_monthly(data, model_name='Isolation Forest', n_estimators=20, contamination=0.03, n_neighbors=20, z_thresh=3.0):
    monthly_anomalies = {}
    data['Anomaly'] = None
    data['Anomaly_Score'] = None

    for period, group in data.groupby(data['Date'].dt.to_period('M')):
        group = feature_engineering(group)
        features = group[['Amount', 'Category_enc', 'DayOfWeek', 'Hour']]
        if model_name == 'Isolation Forest':
            model = IsolationForest(n_estimators=n_estimators, max_features=0.8, max_samples=0.8, contamination=contamination, random_state=42)
            preds = model.fit_predict(features)
            scores = model.decision_function(features)
        elif model_name == 'Local Outlier Factor':
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            preds = model.fit_predict(features)
            scores = model.negative_outlier_factor_
        elif model_name == 'Z-Score':
            # Only use Amount for Z-score
            z_scores = zscore(group['Amount'])
            preds = np.where(np.abs(z_scores) > z_thresh, -1, 1)
            scores = z_scores
        
        else:
            preds = np.ones(len(group))
            scores = np.zeros(len(group))
        group['Anomaly'] = preds
        group['Anomaly_Score'] = scores
        anomalies = group[group['Anomaly'] == -1]
        if not anomalies.empty:
            monthly_anomalies[period] = anomalies[['Date', 'Time', 'Amount', 'Description', 'Category', 'Anomaly_Score']]
        # Visualization: plot Amount vs Anomaly Score
        plt.figure(figsize=(8,4))
        plt.scatter(group['Amount'], group['Anomaly_Score'], c=(group['Anomaly']==-1), cmap='coolwarm', label='Anomaly')
        plt.xlabel('Amount')
        plt.ylabel('Anomaly Score' if model_name != 'Z-Score' else 'Z-Score')
        plt.title(f'Anomaly Scores for {period}')
        plt.legend(['Normal','Anomaly'])
        plt.tight_layout()
        plt.savefig(f'anomaly_plot_{period}.png')
        plt.close()
    return monthly_anomalies

# Streamlit application
st.set_page_config(page_title="Financial Transaction Analyzer", layout="wide")

st.title("💰 Financial Transaction Analyzer")
st.write("This application helps you analyze financial transactions by detecting anomalies based on uploaded transaction data.")


# Sidebar for user input
st.sidebar.header("User Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])


# Advanced ML options
st.sidebar.header("ML Model & Parameters")
model_name = st.sidebar.selectbox("Select Anomaly Detection Model", ["Isolation Forest", "Local Outlier Factor", "Z-Score"])

# Show only relevant hyperparameters
contamination = None
n_estimators = None
n_neighbors = None
z_thresh = None
if model_name == "Isolation Forest":
    contamination = st.sidebar.slider("Contamination (expected anomaly %)", 0.01, 0.2, 0.03, step=0.01)
    n_estimators = st.sidebar.slider("n_estimators", 10, 100, 20, step=5)
elif model_name == "Local Outlier Factor":
    contamination = st.sidebar.slider("Contamination (expected anomaly %)", 0.01, 0.2, 0.03, step=0.01)
    n_neighbors = st.sidebar.slider("n_neighbors", 5, 50, 20, step=1)
elif model_name == "Z-Score":
    z_thresh = st.sidebar.slider("Z-Score Threshold", 2.0, 5.0, 3.0, step=0.1)

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
        # Pass only relevant params
        detect_kwargs = {"model_name": model_name}
        if model_name == "Isolation Forest":
            detect_kwargs["n_estimators"] = n_estimators
            detect_kwargs["contamination"] = contamination
        elif model_name == "Local Outlier Factor":
            detect_kwargs["n_neighbors"] = n_neighbors
            detect_kwargs["contamination"] = contamination
        elif model_name == "Z-Score":
            detect_kwargs["z_thresh"] = z_thresh
        monthly_anomalies = detect_anomalies_monthly(transaction_data, **detect_kwargs)
        # Display anomalies and plots
        if monthly_anomalies:
            for month, anomalies in monthly_anomalies.items():
                st.subheader(f"📊 Anomalies Detected for {month}:")
                st.write(anomalies)
                plot_path = f'anomaly_plot_{month}.png'
                st.image(plot_path, caption=f'Anomaly Score Plot for {month}')
        else:
            st.warning("No anomalies detected in the uploaded data.")


    # --- GenAI-powered Key Insights Box ---
    import io
    import textwrap
    import google.generativeai as genai
    st.markdown("""
        <style>
        .insight-box {
            background: linear-gradient(90deg, #e0eafc 0%, #cfdef3 100%);
            border-radius: 12px;
            padding: 1.5em 2em;
            margin-top: 2em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
            font-size: 1.1em;
        }
        .insight-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #1a237e;
            margin-bottom: 0.5em;
        }
        .genai-summary-box {
            background: linear-gradient(90deg, #f8ffae 0%, #43c6ac 100%);
            border-radius: 12px;
            padding: 1.5em 2em;
            margin-top: 1.5em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.09);
            font-size: 1.08em;
        }
        .genai-summary-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #00695c;
            margin-bottom: 0.5em;
        }
        </style>
    """, unsafe_allow_html=True)

    def generate_key_insights(df):
        # Simple GenAI-style summary (could be replaced with LLM API)
        total = df['Amount'].sum()
        avg = df['Amount'].mean()
        max_amt = df['Amount'].max()
        min_amt = df['Amount'].min()
        most_cat = df['Category'].value_counts().idxmax()
        most_cat_amt = df.groupby('Category')['Amount'].sum().idxmax()
        busiest_day = df['Date'].dt.day_name().value_counts().idxmax()
        n_tx = len(df)
        n_months = df['Date'].dt.to_period('M').nunique()
        return textwrap.dedent(f'''
            <div class=\"insight-title\">🔍 Key Insights</div>
            <ul>
                <li><b>Total Transactions:</b> {n_tx}</li>
                <li><b>Time Span:</b> {n_months} months</li>
                <li><b>Total Spent:</b> ₹{total:,.2f}</li>
                <li><b>Average Transaction:</b> ₹{avg:,.2f}</li>
                <li><b>Largest Transaction:</b> ₹{max_amt:,.2f}</li>
                <li><b>Smallest Transaction:</b> ₹{min_amt:,.2f}</li>
                <li><b>Most Frequent Category:</b> {most_cat}</li>
                <li><b>Highest Spending Category:</b> {most_cat_amt}</li>
                <li><b>Busiest Day:</b> {busiest_day}</li>
            </ul>
            <i>These insights are generated using GenAI-style analysis of your uploaded data.</i>
        </div>
        ''')


    # --- Real LLM (Gemini) Integration ---
    def get_gemini_insights(df, api_key, prompt_type="summary"):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        # Prepare a concise summary of the data for the LLM
        summary = f"""
        You are a financial data analyst. Here is a summary of the user's transaction data:
        - Total transactions: {len(df)}
        - Time span: {df['Date'].dt.to_period('M').nunique()} months
        - Total spent: ₹{df['Amount'].sum():,.2f}
        - Average transaction: ₹{df['Amount'].mean():,.2f}
        - Largest transaction: ₹{df['Amount'].max():,.2f}
        - Smallest transaction: ₹{df['Amount'].min():,.2f}
        - Most frequent category: {df['Category'].value_counts().idxmax()}
        - Highest spending category: {df.groupby('Category')['Amount'].sum().idxmax()}
        - Busiest day: {df['Date'].dt.day_name().value_counts().idxmax()}
        """
        if prompt_type == "summary":
            prompt = summary + "\n\nGenerate a natural language summary of the user's spending patterns, highlighting any interesting trends or anomalies."
        else:
            prompt = summary + "\n\nBased on this data, provide personalized recommendations to help the user manage their expenses better."
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"<i>LLM error: {e}</i>"

    # Use your Gemini API key here
    GEMINI_API_KEY = "AIzaSyB77fhJKNIQdIzTPQmGg8HBT7Tjdc3-TrY"

    st.markdown('<div class=\"insight-box\">' + generate_key_insights(transaction_data) + '</div>', unsafe_allow_html=True)

    with st.spinner("Generating LLM-powered summary..."):
        gemini_summary = get_gemini_insights(transaction_data, GEMINI_API_KEY, prompt_type="summary")
    st.markdown('<div class=\"genai-summary-box\">' + f'<div class="genai-summary-title">🧠 Gemini LLM Summary</div>' + gemini_summary + '</div>', unsafe_allow_html=True)

    with st.spinner("Generating LLM-powered recommendations..."):
        gemini_recs = get_gemini_insights(transaction_data, GEMINI_API_KEY, prompt_type="recommendations")
    st.markdown('<div class=\"genai-summary-box\">' + f'<div class="genai-summary-title">💡 Gemini LLM Recommendations</div>' + gemini_recs + '</div>', unsafe_allow_html=True)

# Footer
st.write("___")
st.write("© 2024 Financial Transaction Analyzer. All Rights Reserved.")
