# 📈 Influenza Weekly Analysis & Forecasting Dashboard

**Filename:** `Influenza_Weekly_Analysis_Streamlit.py`

A Streamlit-based interactive dashboard for analyzing and forecasting **weekly influenza cases** across countries.  
It combines **data cleaning, visualization, forecasting, alerts, and news integration** to help monitor flu trends worldwide.

---

## 🚀 Features

- **Data loading & cleaning**
  - Reads a CSV of weekly influenza case counts or search trends.
  - Normalizes column names, parses dates, maps countries → ISO3.

- **Interactive visualizations**
  - Country-wise weekly trends (line/bar).
  - Global choropleth heatmap (intensity by country).
  - Historical year-wise trend comparisons.
  - Top-5 leaderboard of most affected countries per week.

- **Forecasting**
  - Supports **Prophet** (preferred) and **ARIMA** (fallback).
  - Predicts the next **2–8 weeks**.
  - Detects and alerts **high-risk future weeks** using threshold methods (quantile, mean±std, median×2).

- **Early warning alerts**
  - Highlights periods exceeding thresholds.
  - Configurable alert methods via sidebar.

- **Exporting**
  - Download filtered dataset as CSV.
  - Export selected country’s trend as PNG.
  - Ready to extend for PDF/Excel reports.

- **Live News & Guidelines**
  - Fetches **real-time influenza / health news** using:
    - **NewsAPI / NewsData.io** (requires API key), OR
    - **Google News RSS (no key needed)**.
  - Sidebar keyword search (`influenza`, `bird flu`, `H5N1`, etc.).
  - Expandable panels showing title, date, summary, and link.

- **Notifications (placeholders)**
  - Email alerts (via SMTP).
  - SMS alerts (via Twilio).
  - Commented code with instructions for integration.

- **Customizable settings**
  - Country selection.
  - Date range filtering.
  - Year-wise comparison selection.
  - Forecast horizon and alert thresholds.

---

## 🛠️ Installation & Setup

1. **Clone the repository** and place your dataset (`influenza_weekly.csv`) in the root folder.  
   Or update the path in `read_and_prepare()`.

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate

3. **Install dependencies:**
pip install streamlit pandas numpy matplotlib plotly pycountry python-dateutil prophet pmdarima pycountry-convert requests feedparser

4. **Run the app:**
streamlit run Influenza_Weekly_Analysis_Streamlit.py

5. **Dataset Requirements**
Your CSV should have the following minimum columns (case-insensitive):

sdate (week start date, e.g., 2021-01-04)

country (country name)

all_inf (number of influenza cases or search volume)

If your dataset has different column names, update the mapping inside read_and_prepare().

**News Integration**

Option 1: API Key Method

Register at NewsAPI or NewsData.io
Set your key:

# Windows PowerShell
$env:NEWS_API_KEY="your_api_key"
# macOS/Linux
export NEWS_API_KEY="your_api_key"

Restart the app.

Option 2: No Key Needed
The app can use Google News RSS feeds.
Simply type a keyword (e.g., flu, H5N1, avian influenza) in the sidebar and fetch news.

* Example Workflow

1. Upload or use the default influenza_weekly.csv.
2. Select a country (e.g., India).
3. Adjust the date range to focus on recent years.
4. Compare trends across 2020–2024.
5. Run forecasting for the next 6 weeks.
6. Review alerts if high-risk weeks are detected.
7. Check related flu news in the sidebar.
8. Download filtered dataset or export charts.
9. Upload or use the default influenza_weekly.csv.
10. Select a country (e.g., India).
11. Adjust the date range to focus on recent years.
12. Compare trends across 2020–2024.
13. Run forecasting for the next 6 weeks.
14. Review alerts if high-risk weeks are detected.
15. Check related flu news in the sidebar.
16. Download filtered dataset or export charts.