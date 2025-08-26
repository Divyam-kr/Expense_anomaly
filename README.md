# 💸 Expense Anomaly & Insights Analyzer

## Overview
This project is a powerful Streamlit web application for analyzing financial transactions, detecting anomalies, and generating actionable insights using both advanced machine learning and real GenAI (LLM) technology (Google Gemini).

---

## ✨ Features
- **Upload & Analyze**: Upload your CSV of transactions and instantly visualize your data.
- **Automatic Categorization**: Transactions are categorized using smart keyword matching.
- **Anomaly Detection**: Detects unusual transactions using:
	- Isolation Forest
	- Local Outlier Factor (LOF)
	- Z-Score
- **Feature Engineering**: Uses amount, category, day of week, and hour for robust anomaly detection.
- **Interactive ML Controls**: Select model and tune hyperparameters from the sidebar.
- **Visualizations**: See anomaly scores and outliers in clear, monthly scatter plots.
- **GenAI Key Insights**: Get a beautiful summary of your spending patterns and statistics.
- **LLM-Powered Analysis**: Real Google Gemini LLM generates:
	- Natural language summary of your spending
	- Personalized recommendations for better expense management

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Expense_anomaly-main
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Set Up Gemini API Key
Edit `app.py` and set your Gemini API key in the `GEMINI_API_KEY` variable, or use environment variables for security.

### 4. Run the App
```bash
streamlit run app.py
```

---

## 📂 Data Format
Your CSV should have these columns:

| Date       | Time  | Amount  | Description           |
|------------|-------|---------|----------------------|
| 2023-09-30 | 00:37 | 1651.04 | Restaurant meal      |

---

## 🧠 Machine Learning & GenAI Details
- **ML Models**: Isolation Forest, LOF, Z-Score (with feature engineering)
- **GenAI**: Uses Google Gemini LLM via `google-generativeai` to:
	- Summarize your spending in natural language
	- Give actionable, personalized recommendations
- **Visualization**: Matplotlib plots for anomaly scores

---

## 🛡️ Security & Privacy
- Your data is processed locally and only the summary is sent to Gemini LLM for insights.
- Never share your API key publicly.

---

## 🙌 Credits
- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), [matplotlib](https://matplotlib.org/), [Google Gemini](https://ai.google.dev/), and [pandas](https://pandas.pydata.org/).

---

## 📞 Contact
For questions or suggestions, open an issue or contact the project maintainer.