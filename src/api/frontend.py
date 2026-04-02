# House Market Prediction Frontend UI web-interface powered by Streamlit

# Run the API locally using : 
# $ streamlit run src/api/frontend.py

import os
import streamlit as st
import requests, boto3, json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Paths
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", 'http://localhost:8000') # Environment variable is overwritten by ECS when deployed on AWS
ROOT_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_PATH / 'data'
PREPROCESSED_HOLDOUT_PATH = DATA_PATH / 'processed' / 'preprocessed_holdout.csv'
FE_HOLDOUT_PATH = DATA_PATH / 'feature_engineered' / 'fe_holdout.csv'

# Downloading data required by frontend from S3 bucket if not present
bucket = 'housemarket-ml-end-to-end'
region = 'eu-west-2'
s3 = boto3.client("s3", region_name=region)
def download_file(local_path: Path, s3_path: str):
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print("Saving to:", local_path.resolve())
        print("Exists dir:", local_path.parent.exists())
        s3.download_file(bucket, s3_path, str(local_path))
        print(f'📥  {local_path} downloaded')
    return None
download_file(PREPROCESSED_HOLDOUT_PATH, 'preprocessed_holdout.csv')
download_file(FE_HOLDOUT_PATH, 'fe_holdout.csv')


# Page config
st.set_page_config(
    page_title="House Market Prediction",
    page_icon="🏠",
    layout="wide",
)

# Styling 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

div[data-testid="stSelectbox"] label {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* Disable the geo drag rectangle — prevents map panning */
.js-plotly-plot .geo .drag {
    pointer-events: none !important;
}

/* Re-enable pointer events on the markers so clicks still work */
.js-plotly-plot .geo .geo-layers .trace {
    pointer-events: all !important;
}
</style>
""", unsafe_allow_html=True)

# Load all data using Streamlit's cache
@st.cache_data
def load_data():
    df_fe = pd.read_csv(FE_HOLDOUT_PATH)
    df_pre = pd.read_csv(PREPROCESSED_HOLDOUT_PATH)
    df_fe['city_full'] = df_pre['city_full']
    return df_fe


# Load available cities, years, and months from the holdout data
df_fe = load_data()
cities = sorted(df_fe['city_full'].unique())
years = sorted(df_fe['year'].unique())
months = sorted(df_fe['month'].unique())
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# Session state initialization
for key, default in [("selected_city", cities[3]),
                     ("selected_year",  years[0]),
                     ("selected_month", months[-5]),
                     ("show_pred", False),
                     ("highlighted_month", months[-1])]:
    if key not in st.session_state:
        st.session_state[key] = default


# Process map click BEFORE selectboxes render
map_state = st.session_state.get("us_map", {})
if map_state:
    points = (map_state.get("selection") or {}).get("points", [])
    if points:
        clicked = points[0].get("customdata")
        if clicked and clicked != st.session_state.selected_city:
            print(clicked)
            st.session_state.city_selector = clicked


# Page title
st.markdown("# 🏠 House Market Prediction")
st.markdown("---")

# Selection buttons
col_city, col_year, col_month = st.columns(3)
with col_city:
    selected_city = st.selectbox(
        "Select City",
        options=cities,
        # index=cities.index(st.session_state.selected_city),
        key="city_selector",
    )
with col_year:
    selected_year = st.selectbox(
        "Select Year",
        options=years,
        index=years.index(st.session_state.selected_year),
        key="year_selector",
    )
with col_month:
    selected_month = st.selectbox(
        "Select Month",
        options=months,
        format_func=lambda m: MONTH_NAMES[m],
        index=months.index(st.session_state.selected_month),
        key="month_selector",
    )

# US Map for city selection
df_map = df_fe.copy()
city_coords = df_map.drop_duplicates("city_full")[["city_full","lat","lng"]].reset_index(drop=True)
others  = city_coords[city_coords["city_full"] != st.session_state.city_selector]
sel_row = city_coords[city_coords["city_full"] == st.session_state.city_selector].iloc[0]

# Display non-select cities
fig_map = go.Figure()
fig_map.add_trace(go.Scattergeo(
    lon=others["lng"], lat=others["lat"],
    text=others["city_full"], customdata=others["city_full"],
    mode="markers",
    marker=dict(size=10, color="#60a5fa", opacity=0.7,
                line=dict(width=1, color="rgba(0,0,0,0.2)")),
    hovertemplate="<b>%{text}</b><extra></extra>",
    name="Cities",
))
# Display select cities
fig_map.add_trace(go.Scattergeo(
    lon=[sel_row["lng"]], lat=[sel_row["lat"]],
    text=[sel_row["city_full"]], customdata=[sel_row["city_full"]],
    mode="markers+text",
    marker=dict(size=16, color="#f97316", opacity=1.0,
                line=dict(width=2, color="#ffffff")),
    textfont=dict(color="#f97316", size=12, family="DM Mono"),
    textposition="top center",
    hovertemplate="<b>%{text}</b><extra></extra>",
    name="Selected",
))

fig_map.update_layout(
    geo=dict(
        scope="usa", projection_type="albers usa",
        showland=True, landcolor="rgba(200,210,200,0.35)",
        showlakes=True, lakecolor="rgba(100,150,220,0.2)",
        showcoastlines=True, coastlinecolor="rgba(128,128,128,0.4)",
        subunitcolor="rgba(128,128,128,0.3)",
        showframe=True, bgcolor="rgba(0,0,0,0)",
        lataxis=dict(range=[24, 50]),
        lonaxis=dict(range=[-125, -66]),
    ),
    dragmode=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
    height=400,
    legend=dict(font=dict(family="DM Mono", size=11), borderwidth=1),
    hoverlabel=dict(font=dict(family="DM Mono", size=12)),
)

st.plotly_chart(
    fig_map,
    use_container_width=True,
    on_select="rerun",
    key="us_map",
    config={
        "scrollZoom": False,
        "displayModeBar": False,
        "doubleClick": False,
    }
)


# Predictions button
st.markdown("")
if st.button("🚀 Show Predictions", type="primary", use_container_width=True):

    st.session_state.selected_city = st.session_state.city_selector
    selected_city = st.session_state.selected_city
    st.session_state.selected_year = selected_year
    st.session_state.selected_month = selected_month

    # From the feature engineered data, only keep the selected city and the selected year
    df_selected = df_fe.copy()
    df_selected = df_selected[df_selected['city_full'] == selected_city]
    df_selected = df_selected[df_selected['year'] == selected_year]
    st.session_state.highlighted_month = selected_month

    # Remove 'city_full' and convert to list[dict]
    df_to_predict = df_selected.copy()
    df_to_predict.drop(columns=['city_full'], inplace=True)
    list_to_predict = df_to_predict.to_dict('records')

    # Send request to Backend API as list[dict]
    response = requests.post(f'{BACKEND_API_URL}/predict', json=list_to_predict)
    predictions = response.json()

    # Check response status code
    if response.status_code != 200:
        print('❌ invalid status code')

    # Check if length from input and response match
    st.session_state.valid_prediction = len(predictions) == len(df_selected['price'])
    if st.session_state.valid_prediction:
        print('✅ Valid prediction')
        st.session_state.show_pred = True
    else:
        print('❌ Invalid prediction')
        st.session_state.show_pred = False

    # Update prevision DataFrame to display
    df_selected = df_selected.rename(columns={'price': 'actual_price'})
    df_selected['predicted_price'] = predictions
    df_selected = df_selected.sort_values(by=['month'])

    st.session_state.df_to_plot = df_selected


# Display results
if st.session_state.show_pred:

    df = st.session_state.df_to_plot

    # Prepare month feature
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    monthly_avg = df.groupby("month")[["actual_price", "predicted_price"]].mean().reset_index()

    # Graph title 
    st.markdown(f"#### 📈 {st.session_state.selected_city}, Yearly Trend — {st.session_state.selected_year}")
    fig = px.line(
        monthly_avg,
        x="month",
        y=["actual_price", "predicted_price"],
        markers=True,
        labels={"value": "Price", "month": "Month"},
    )

    # Add highlight with background shading for the selected month
    fig.add_vrect(
        x0=st.session_state.selected_month - 0.5,
        x1=st.session_state.selected_month + 0.5,
        fillcolor="green",
        opacity=0.1,
        layer="below",
        line_width=0,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plot Yearly accuracy metrics 
    st.markdown("#### 📊 Yearly Accuracy Metrics")
    y_pred = df['predicted_price']
    y_eval = df['actual_price']
    mae = mean_absolute_error(y_eval.values, y_pred)
    rmse = np.sqrt(mean_squared_error(y_eval.values, y_pred))
    avg_error = 100 * np.mean( np.abs(y_eval.values - y_pred) / y_eval.values)

    def metric_block(label, value):
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h4>{label} : {value}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_block("Average Error", f"{avg_error:.2f}%")
    with col2:
        metric_block("MAE", f"${mae:,.0f}")
    with col3:
        metric_block("RMSE", f"${rmse:,.0f}")
    

    # Display prediction table
    st.markdown("#### 🚀 Monthly Predictions")
    st.dataframe(monthly_avg, use_container_width=True)

else:
    st.info("Choose filters and click **Show Predictions** to compute.")