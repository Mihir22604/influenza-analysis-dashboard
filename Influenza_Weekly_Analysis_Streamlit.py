"""
Influenza Weekly Analysis & Forecasting Streamlit App
Filename: Influenza_Weekly_Analysis_Streamlit.py

What this file includes:
- Data loading & cleaning for a CSV of weekly influenza search/case counts per country
- Country-wise weekly trends (interactive line/bar) + data table with filters
- World heatmap (choropleth) showing intensity by country (hover shows values)
- Top-5 leaderboard per selected week
- Historical trend comparison across years
- Forecasting (Prophet or ARIMA fallback) for next 2-4 weeks
- Early warning alerts for high-risk weeks
- Exporting (CSV, PNG of charts, PDF via matplotlib save)
- News API + WHO/CDC health tips integration (placeholder: needs API key)
- Email/SMS notification placeholders (SMTP / Twilio) — commented with instructions

How to run (step-by-step):
1) Put your dataset in the same folder with name `influenza_weekly.csv` or update the path below.
2) Create and activate a Python virtual environment (recommended):
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
3) Install dependencies (these are the required packages):
   pip install streamlit pandas numpy matplotlib plotly pycountry python-dateutil prophet pmdarima pycountry-convert
   # If `prophet` installation fails on your machine, use the ARIMA fallback included.
   # On some platforms, Prophet's package name is `prophet` (PyPI) or `fbprophet` older name.
4) Run the Streamlit app:
   streamlit run Influenza_Weekly_Analysis_Streamlit.py
5) In the app UI, select country, date range, compare years, run forecasting, and download reports.

Notes:
- This script assumes your CSV has at least the following columns (case-insensitive):
    - 'week' or 'date' (week identifier: e.g., '2021-01' or '2021-01-04' indicating week start)
    - 'country' (country name)
    - 'cases' or 'count' or 'value' (numeric value representing case count or search volume)
  If your column names differ, edit the `read_and_prepare()` function mapping.

- News API & SMS/email require API keys / account setup. See placeholders in the code.

"""

import io
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

import requests
import feedparser

# Prophet and ARIMA imports with safe fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

# utilities
import pycountry

# ----------------------
# Data loading & cleaning
# ----------------------

def read_and_prepare(path='influenza_weekly.csv'):
    df = pd.read_csv(path)

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Use specific columns from your dataset
    date_col = 'sdate'        # weekly start date
    country_col = 'country'
    value_col = 'all_inf'     # total influenza cases

    # Parse dates
    df['week_start'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['week_start'])

    # Clean country names
    df['country'] = df[country_col].astype(str).str.strip()

    # Ensure numeric cases
    df['cases'] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)

    # Aggregate to weekly-country level
    df = df.groupby(['week_start', 'country'], as_index=False)['cases'].sum()

    # Add year/week number
    df['year'] = df['week_start'].dt.year
    df['week_num'] = df['week_start'].dt.isocalendar().week

    # Manual corrections + ISO3 mapping
    country_corrections = {
        "United States of America": "USA",
        "Russian Federation": "RUS",
        "Iran (Islamic Republic of)": "IRN",
        "Viet Nam": "VNM",
        "Republic of Korea": "KOR",
        "United Kingdom": "GBR",
        "Bolivia (Plurinational State of)": "BOL",
        "Venezuela (Bolivarian Republic of)": "VEN",
        # add more if needed
    }

    def country_to_iso3(name):
        if name in country_corrections:
            return country_corrections[name]
        try:
            c = pycountry.countries.lookup(name)
            return c.alpha_3
        except Exception:
            return None
    df['iso3'] = df['country'].apply(country_to_iso3)
    return df


# ----------------------
# Forecasting helpers
# ----------------------

def forecast_prophet(df_country, periods=4):
    # df_country: columns ['week_start','cases'] weekly
    df_for = df_country[['week_start','cases']].rename(columns={'week_start':'ds','cases':'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_for)
    future = m.make_future_dataframe(periods=periods, freq='W-MON')
    forecast = m.predict(future)
    return m, forecast


def forecast_arima(df_country, periods=4):
    # Univariate ARIMA via pmdarima
    series = df_country.set_index('week_start')['cases'].asfreq('W-MON').fillna(0)
    model = pm.auto_arima(series, seasonal=True, m=52, suppress_warnings=True, error_action='ignore')
    fc, confint = model.predict(n_periods=periods, return_conf_int=True)
    last_date = series.index.max()
    future_idx = pd.date_range(start=last_date + pd.Timedelta(7, unit='D'), periods=periods, freq='W-MON')
    forecast = pd.DataFrame({'ds': future_idx, 'yhat': fc, 'yhat_lower': confint[:,0], 'yhat_upper': confint[:,1]})
    return model, forecast

# ----------------------
# Alerts & thresholds
# ----------------------

def compute_thresholds(df_country, method='quantile', q=0.95):
    if method == 'quantile':
        thr = df_country['cases'].quantile(q)
    elif method == 'mean_std':
        thr = df_country['cases'].mean() + 2*df_country['cases'].std()
    else:
        thr = df_country['cases'].median() * 2
    return thr

# ----------------------
# App UI (Streamlit)
# ----------------------

st.set_page_config(layout='wide', page_title='Influenza Weekly Trends & Forecast')

st.title('📈 Identifying Flu Symptoms using Google Search Queries — Weekly Dashboard')

# Sidebar: upload or use default file
st.sidebar.header('Data & Settings')
uploaded_file = st.sidebar.file_uploader('Upload influenza_weekly.csv', type=['csv'])

if uploaded_file is not None:
    df = read_and_prepare(uploaded_file)
else:
    if os.path.exists('influenza_weekly.csv'):
        df = read_and_prepare('influenza_weekly.csv')
        st.write("Available years:", df['year'].unique()) # added hear
        st.write("Available countries:", df['country'].unique()) # added hear
        st.write("Sample rows:", df.head())
    else:
        st.sidebar.error('No dataset found. Upload a CSV named influenza_weekly.csv or use the uploader.')
        st.stop()

# Controls
countries = sorted(df['country'].unique())
selected_country = st.sidebar.selectbox('Select Country', ['All'] + countries)

min_date = df['week_start'].min()
max_date = df['week_start'].max()
start_date, end_date = st.sidebar.date_input('Date range (week start)', [min_date, max_date])

if isinstance(start_date, list) or isinstance(start_date, tuple):
    start_date = start_date[0]
if isinstance(end_date, list) or isinstance(end_date, tuple):
    end_date = end_date[-1]

#--------------------------
years_available = sorted(df['year'].dropna().unique().tolist())

compare_years = st.sidebar.multiselect(
    'Compare years (hold Ctrl)', 
    years_available, 
    default=[min(years_available)] if years_available else [],
    key='compare_years_sidebar'   # <-- add a unique key
)

#--------------------------

#compare_years = st.sidebar.multiselect('Compare years (hold Ctrl)', sorted(df['year'].unique().tolist()), default=[min(df['year'].unique())])

forecast_weeks = st.sidebar.slider('Forecast horizon (weeks)', 2, 8, 4)
threshold_method = st.sidebar.selectbox('Threshold method for alerts', ['quantile','mean_std','median_mult'], index=0)
threshold_q = st.sidebar.slider('Quantile (if quantile method)', 0.80, 0.99, 0.95)

# Filter data by date & country
mask = (df['week_start'] >= pd.to_datetime(start_date)) & (df['week_start'] <= pd.to_datetime(end_date))
if selected_country != 'All':
    mask &= (df['country'] == selected_country)

df_view = df[mask].copy()

# Main layout
col1, col2 = st.columns([3,1])

with col1:
    st.subheader('Country-wise Weekly Flu Trends')
    if selected_country == 'All':
        st.info('Select a specific country to see its weekly trend. For all countries, see the map or leaderboard.')
    else:
        df_country = df_view[df_view['country'] == selected_country].sort_values('week_start')
        if df_country.empty:
            st.warning('No data for selected country & date range.')
        else:
            fig = px.line(df_country, x='week_start', y='cases', markers=True, title=f'{selected_country} — Weekly Cases')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.subheader('Historical Trend Comparison')

    # normalize years to avoid mismatch
    available_years = sorted(df['year'].unique().tolist())

    # 👇 Sidebar option to pick years
    compare_years = st.sidebar.multiselect(
    'Compare years (hold Ctrl)', 
    sorted(df['year'].unique().tolist()), 
    default=[min(df['year'].unique())]   # ⚠️ may break if df is empty
)
    # compare selected years for selected country (or global aggregate)
    comp_df = df.copy()
    if selected_country != 'All':
        comp_df = comp_df[comp_df['country']==selected_country]
    comp_df = comp_df[comp_df['year'].isin(compare_years)]
    if comp_df.empty:
        st.warning('No data for selected years/country')
    else:
        comp_pivot = comp_df.groupby(['year','week_num'], as_index=False)['cases'].mean()
        fig2 = px.line(comp_pivot, x='week_num', y='cases', color='year', markers=True, title='Week-of-year comparison (average cases by week number)')
        st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader('Top-5 Most Affected Countries (Leaderboard)')
    # compute latest week in the filtered window
    latest_week = df_view['week_start'].max()
    if pd.isna(latest_week):
        st.write('No data in range')
    else:
        leaderboard = df[df['week_start']==latest_week].nlargest(5,'cases')[['country','cases']]
        st.table(leaderboard.reset_index(drop=True))

    st.markdown('---')
    st.subheader('Download & Export')
    csv = df_view.to_csv(index=False)
    st.download_button('Download filtered data as CSV', csv, file_name='filtered_influenza_weekly.csv', mime='text/csv')

# World map heatmap
st.subheader('🌍 World Map — Flu Intensity Heatmap')
map_df = df_view.groupby(['country','iso3'], as_index=False)['cases'].sum()
map_df_iso = map_df.dropna(subset=['iso3'])
if map_df_iso.empty:
    st.warning('No ISO-mapped country data to show on map. Ensure country names are standard (United States, India, etc).')
else:
    fig_map = px.choropleth(map_df_iso, locations='iso3', color='cases', hover_name='country', projection='natural earth', title='Total cases in selected range')
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown('---')

# Forecasting section
st.header('🤖 Forecast & Early Warning')
if selected_country == 'All':
    st.info('Select a specific country for forecasting')
else:
    df_country_full = df[df['country']==selected_country].sort_values('week_start')
    if len(df_country_full) < 2:
        st.warning(f"Not enough data points for forecasting {selected_country}. Need at least 2, got {len(df_country_full)}.")
    else:
        st.write(f'Data points: {len(df_country_full)} weeks')
        # threshold
        thr = compute_thresholds(df_country_full, method=threshold_method, q=threshold_q)
        st.write(f'Computed alert threshold: {thr:.2f} ({threshold_method})')

        if PROPHET_AVAILABLE:
            st.write('Using Prophet for forecasting')
            model, forecast = forecast_prophet(df_country_full, periods=forecast_weeks)
            # show forecast tail
            fc_plot = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(forecast_weeks)
            st.dataframe(fc_plot.assign(ds=fc_plot['ds'].dt.date))
            # Plot
            figf = px.line(forecast, x='ds', y='yhat', title=f'Forecast next {forecast_weeks} weeks')
            figf.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='upper', mode='lines', line={'width':0})
            figf.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='lower', mode='lines', line={'width':0})
            st.plotly_chart(figf, use_container_width=True)

            # early warning detection in forecast
            high_risk = forecast[forecast['yhat'] > thr]
            if not high_risk.empty:
                st.error('High risk predicted in weeks:')
                st.table(high_risk[['ds','yhat']].assign(ds=high_risk['ds'].dt.date))
            else:
                st.success('No high risk weeks predicted in forecast horizon.')
        elif PMDARIMA_AVAILABLE:
            st.write('Prophet not available. Using ARIMA (pmdarima) for forecasting')
            model_ar, forecast_ar = forecast_arima(df_country_full, periods=forecast_weeks)
            st.dataframe(forecast_ar.assign(ds=forecast_ar['ds'].dt.date))
            figf = px.line(forecast_ar, x='ds', y='yhat', title=f'ARIMA Forecast next {forecast_weeks} weeks')
            st.plotly_chart(figf, use_container_width=True)

            high_risk = forecast_ar[forecast_ar['yhat'] > thr]
            if not high_risk.empty:
                st.error('High risk predicted in weeks:')
                st.table(high_risk[['ds','yhat']].assign(ds=high_risk['ds'].dt.date))
            else:
                st.success('No high risk weeks predicted in forecast horizon.')
        else:
            st.warning('Neither Prophet nor pmdarima available. Please install `prophet` or `pmdarima` to enable forecasting.')

#--------------------------------------------------------
# News & Health Guidelines integration (placeholder)
st.markdown('---')
st.header('📰 News & Health Guidelines')
# st.write('This section can fetch news via NewsAPI and show WHO/CDC guidance when risk is high. Provide NEWS_API_KEY in environment to enable.')
# NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
# if NEWS_API_KEY:
#     st.write('NEWS_API_KEY found — example fetch would appear here (requires requests & API usage).')
# else:
#     st.info('To enable real-time news, set environment variable NEWS_API_KEY and uncomment the News fetch block in the script.')
# Sidebar option: keyword for news search
keyword = st.sidebar.text_input("Search keyword for news", value="influenza")

# Google News RSS feed
rss_url = f"https://news.google.com/rss/search?q={keyword}&hl=en-IN&gl=IN&ceid=IN:en"

try:
    feed = feedparser.parse(requests.get(rss_url).content)

    if not feed.entries:
        st.warning("No news articles found for this keyword.")
    else:
        st.write(f"### Latest {len(feed.entries)} news articles about **{keyword}**:")

        for entry in feed.entries[:10]:  # show top 10
            with st.expander(f"📰 {entry.title}"):
                st.write(f"**Published at:** {entry.published}")
                st.write(entry.get("summary", "No description available."))
                st.markdown(f"[Read full article]({entry.link})")

except Exception as e:
    st.error(f"Failed to fetch news. Error: {e}")
#---------------------------------------------------------


# Save a simple PDF/PNG export example using matplotlib (non-interactive)
if st.button('Export Current Country Trend as PNG'):
    if selected_country == 'All':
        st.warning('Select a country first')
    else:
        df_country = df[df['country']==selected_country].sort_values('week_start')
        fig, ax = plt.subplots()
        ax.plot(df_country['week_start'], df_country['cases'])
        ax.set_title(f'{selected_country} — Weekly Cases')
        ax.set_xlabel('Week')
        ax.set_ylabel('Cases')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        st.download_button('Download PNG', buf, file_name=f'{selected_country}_trend.png', mime='image/png')

# End of app

# If running as script, Streamlit will serve the UI above

if __name__ == '__main__':
    pass
