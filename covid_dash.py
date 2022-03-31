import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from requests import get
import re
import os
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import datetime
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pandasql import sqldf 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
try:
    import streamlit_analytics
except ImportError:
    # Install streamlit-analytics on first run (not included in requirements.txt).
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "streamlit_analytics"]
    )
    import streamlit_analytics
    
    
######################
## Streamlit Design // Panel
#####################

st.set_page_config(
    page_title="COVID-Dash",
    page_icon=":bar_chart:",
    layout="wide"
)
streamlit_analytics.start_tracking()

######################
## Case Data Import & Cleaning
#####################

@st.cache
def get_data(url, parsenum):
    df = pd.read_csv(url,  parse_dates=parsenum)
    return df

cat_url ='https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv'
case_url = 'https://data.ontario.ca/datastore/dump/ed270bb8-340b-41f9-a7c6-e8ef587e6d11?bom=True'
outbreak_url = 'https://data.ontario.ca/datastore/dump/66d15cce-bfee-4f91-9e6e-0ea79ec52b3d?bom=True'


case_df = get_data(case_url, [1])

cat_df = get_data(cat_url, [3])


######################
## Cat Data & Cleaning
#####################

case_df= case_df[['Reported Date', 'Confirmed Positive', 'Resolved', 'Deaths',
       'Deaths_New_Methodology', 'Total Cases',
       'Percent positive tests in last day', 
       'Number of patients hospitalized with COVID-19',
       'Number of patients in ICU due to COVID-19',
       'Number of patients in ICU, testing positive for COVID-19',
       'Number of patients in ICU on a ventilator due to COVID-19',
       'Num. of patients in ICU on a ventilator testing positive',
       'Num. of patients in ICU on a ventilator testing negative']]
case_df = case_df.fillna(0)
case_df['Death_Count'] =  case_df['Deaths']+ case_df['Deaths_New_Methodology']
case_df['Deaths'] = case_df['Death_Count'].diff().replace(-411.0, 20)
case_df['Deaths Rolling Avg'] = case_df['Deaths'].rolling(10, min_periods=1).mean()

case_df = case_df[case_df['Reported Date']>'2020-04-18']


cat_df = cat_df[[ 
       'Test_Reported_Date',  'Age_Group', 'Client_Gender',
        'Outcome1', 
        'Reporting_PHU_City', 
        'Reporting_PHU_Latitude',
       'Reporting_PHU_Longitude']]

cut_date = case_df['Reported Date'].tail(1) - datetime.timedelta(days=15)
cat_df = cat_df[cat_df['Test_Reported_Date']>cut_date.to_string(index=False)]
cat_df.rename(columns = {'Reporting_PHU_Latitude':'lat'}, inplace = True)
cat_df.rename(columns = {'Reporting_PHU_Longitude':'lng'}, inplace = True)


#st.write('Last Reporting Date: ' +'**'+case_df['Reported Date'].tail(1).dt.strftime("%B %d, %Y").to_string(index=False)+'**')
st.write('Joel McInnis Data Analyst Challenge')

# cat_df = cat_df[['Row_ID', 
#        'Test_Reported_Date',  'Age_Group', 'Client_Gender',
#         'Outcome1', 'Outbreak_Related',
#         'Reporting_PHU_City', 
#         'Reporting_PHU_Latitude',
#        'Reporting_PHU_Longitude']]

#cat_df = cat_df[cat_df['Test_Reported_Date']>'2022-03-10']


#######################
## Title
#######################

st.title('Ontario COVID-19 Dashboard')
st.write('Last Reporting Date: ' +'**'+case_df['Reported Date'].tail(1).dt.strftime("%B %d, %Y").to_string(index=False)+'**')


#######################
## Function for Mapping
#######################
import pydeck as pdk

geo_df = cat_df[['lng', 'lat']].reset_index(drop=True)

layer = pdk.Layer(
    "HeatmapLayer",
    data=geo_df,
    get_position=['lng', 'lat'],
    radiusPixels=175,
    auto_highlight=True,
    elevation_scale=50,
    pickable=True,
    elevation_range=[0, 2000],
    extruded=True, 
    opacity=0.4,                
    coverage=10)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=-79.4,
    latitude=43.6,
    zoom=8,
    min_zoom=5,
    max_zoom=15,
    pitch=40.5,
    bearing=-27.36)

# Render
map = pdk.Deck(layers=[layer], initial_view_state=view_state)
st.pydeck_chart(map, use_container_width=True)

# HEXAGON_LAYER_DATA = (
#     "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/3d-heatmap/heatmap-data.csv"  # noqa
# )

# # Define a layer to display on a map
# layer = pdk.Layer(
#     "HexagonLayer",
#     HEXAGON_LAYER_DATA,
#     get_position=["lng", "lat"],
#     auto_highlight=True,
#     elevation_scale=50,
#     pickable=True,
#     elevation_range=[0, 3000],
#     extruded=True,
#     coverage=1,
# )

# # Set the viewport location
# view_state = pdk.ViewState(
#     longitude=-1.415, latitude=52.2323, zoom=6, min_zoom=5, max_zoom=15, pitch=40.5, bearing=-27.36,
# )

# # Render
# map = pdk.Deck(layers=[layer], initial_view_state=view_state)

# st.pydeck_chart(map, use_container_width=True)
st.subheader('GTA COVID Case Heatmap')


with st.expander("Explanation & Data Sources"):
     st.write("""
         The chart above is a heatmap of COVID-19 Cases in Ontario for the 15 days from the latest reporting date. The heatmap data comes from the Confirmed positive cases of COVID19 in Ontario file fetched from https://data.ontario.ca/ on load.
         The file size exceeded the system's parameters to handle all records (detailed individual case data is 150 MB+), so the data 
         was reduced by variables and date to a more manageable -- and meaningful -- 15-day period. The script reads the last date and yeilds the date from 15-day prior to subset the file.

The heatmap is placed at the top because it is an efficiently digestible visual. The information is easily absorbed by the user and they get aboard understanding for the recent COVID spread in the GTA, where the TRU LUV offices are located.
     """)
     st.image("https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_256,w_256,f_auto,q_auto:eco,dpr_1/yt6im9kseccytscjrc3b")

#######################
## Widget for Snapshot
#######################

c_today= int(case_df['Confirmed Positive'].values[-1])
d_today= int(case_df['Deaths'].values[-1])
icu_today= int(case_df['Number of patients in ICU due to COVID-19'].values[-1])

c_yest= int(case_df['Confirmed Positive'].values[-2])
d_yest= int(case_df['Deaths'].values[-2])
icu_yest= int(case_df['Number of patients in ICU due to COVID-19'].values[-2])

col0, col1, col2, col3 = st.columns(4)
col0.write('\n\n| Snapshot Metrics for: \n '+'**'+case_df['Reported Date'].tail(1).dt.strftime("%B %d, %Y").to_string(index=False)+'**')
col1.metric("Reporting Date Cases", c_today, c_today-c_yest, delta_color="inverse")
col2.metric("Reporting Date Deaths", d_today, d_today-d_yest,delta_color="inverse")
col3.metric("Reporting ICU Cases", icu_today, icu_today-icu_yest, delta_color="inverse")

with st.expander("Snapshot Explanation"):
     st.write("""Knowing current metrics for daily cases, deaths, and hospitalizations has become a staple for the pandemic last two years.
              The values are pulled and formated from the latest report date record from Daily change in cases by PHU file.
              The delta in the graphic shows the change from the previous day.
         
     """)
#######################
## Cases & Deaths
#######################

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=case_df['Reported Date'], y=case_df['Confirmed Positive'], name="Confirmed Cases", line=dict(color="#b444f4")),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=case_df['Reported Date'], y=case_df['Deaths Rolling Avg'], name="Test Positive Rate", line=dict(color="#44f4b4")),
    secondary_y=True
)


fig.update_xaxes(color="#fff",linecolor="#fff")
fig.update_yaxes(title_text="<b>Confirmed Cases</b>", secondary_y=False, linecolor="#fff")
fig.update_yaxes(title_text="Deaths 10-day Rolling Avg.", secondary_y=True,linecolor="#fff")
fig.update_yaxes(color="#fff")
fig.update_layout(
    title_text="COVID Cases & Deaths",
     template="simple_white",
     font_color="#fff",    
    legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.1,
    font_color="#444"    
))

fig.update_layout({
'plot_bgcolor': 'rgba(33, 33, 33, .1)',
'paper_bgcolor': 'rgba(33, 33, 33, 1)',
'legend_bgcolor': 'rgba(245, 245, 245, 1)'

})


config = {'displayModeBar': False}
st.plotly_chart(fig, use_container_width=True, config=config)

st.subheader('COVID Cases & Deaths')

with st.expander("See explanation"):
     st.write("""
         The chart above uses chronological data of daily covid cases and deaths throughout the pandemic. 
         The large differences in values required the use of two y-axises to demonstrate the interaction between the two metrics and their current trends. 
         There was some data munging required to calculate the death count because of a change in methodology and record keeping. Then the death data recording 
         data seemed to be intermittent since there are many short-term spikes. To correct for this, and provide for a more readable chart, I used a rolling average of 10-days.  The data was calculated from the Daily change in cases by PHU file.

        The chart is fully interactive. You can hover over points on either line to see the value, click on the legend to isolate the metric, and you can crop/zoom 
        in over the chart for user-defined periods.
     """)

#######################
## Bar Chart Outbreak
#######################
out_df = get_data(outbreak_url, [1])
out_df = out_df[out_df['date']>cut_date.to_string(index=False)]

q1 = """SELECT outbreak_group, SUM(number_ongoing_outbreaks) As Cases 
        FROM out_df
        GROUP BY outbreak_group """

q1df = sqldf(q1)

q1df.outbreak_group = q1df.outbreak_group.str.strip("123456")
#color_discrete_map = {'MALE': '#71fcf7', 'FEMALE':'#b444f4'}
fig2 = px.bar(q1df, x='outbreak_group', y='Cases', color='Cases')

fig2.update_xaxes(color="#fff")
fig2.update_yaxes(title_text="<b>Confirmed Cases</b>", linecolor="#fff")

fig2.update_yaxes(color="#fff")
fig2.update_layout(
    title_text="Outbreaks by Environments",
     template="simple_white",
     font_color="#fff",    
    legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99,
    font_color="#444"    
))

fig2.update_layout({
'plot_bgcolor': 'rgba(33, 33, 33, .1)',
'paper_bgcolor': 'rgba(33, 33, 33, 1)',
'legend_bgcolor': 'rgba(245, 245, 245, 1)'

})

st.plotly_chart(fig2, use_container_width=True, config=config)
st.subheader('Ontario Outbreaks for the past 15-days')

with st.expander("See explanation"):
     st.write("""
         The bar chart above represents outbreaks of COVID cases by setting for the trailing 15-day period. There are four primary categories, including the workplace which could be a guiding principle to anser the question of "when can we return to the office?" 

The data comes from the file Ongoing outbreaks on
 https://data.ontario.ca/. The data was subset for the 15-day period and aggregated by summing the value of cases by day per group.

Now the issue with this data, is we don't have the population of the workforce that has returned to the office -- as this is not data tracked daily as these outbreaks are recorded.
     """)
     
     
#######################
## Indicator Index
#######################    

#Adding feature for Case fatality rate
case_df['CFR'] = case_df['Death_Count']/case_df['Confirmed Positive'].cumsum()
scaler = StandardScaler()
scale_df = case_df[20:]
scale_df = pd.DataFrame(scaler.fit_transform(scale_df[['CFR','Percent positive tests in last day','Number of patients in ICU due to COVID-19']]),
                 index=scale_df['Reported Date'], columns= ['CFR','Percent positive tests in last day','Number of patients in ICU due to COVID-19'])

scaler2 = MinMaxScaler()
index_value = (pd.DataFrame(scaler2.fit_transform(pd.DataFrame(scale_df.mean(axis=1))))*100).tail(1)
index_value = int(index_value.values)


fig3 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = index_value,
            title = {'text': "Composite Index"},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': "black"},
                     'steps' : [
                        {'range': [0, 25], 'color': "#66e53b"},
                        {'range': [25, 75], 'color': "#f2f2f2"},   
                        {'range': [75, 100], 'color': "darkred"}],},
            domain = {'x': [0.2, 1], 'y': [0, 1]}
        ))
fig3.update_layout(width=500, height=500, 
         margin={'l':0})

st.markdown('<br />​<hr />​​​​​​​​​​​​​​​​​​​<br />', unsafe_allow_html=True)
fig4 = px.line((pd.DataFrame(scaler2.fit_transform(pd.DataFrame(scale_df.mean(axis=1))), index=scale_df.index)*100))



fig4.update_xaxes(color="#fff")
fig4.update_yaxes(title_text="<b>Index Value</b>", linecolor="#fff")

fig4.update_yaxes(color="#fff")
fig4.update_layout(
    title_text="Composite Index Time Series",
     template="simple_white",
     font_color="#fff",    
    legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99,
    font_color="#444"    
))
fig4.update_layout({
'plot_bgcolor': 'rgba(33, 33, 33, .1)',
'paper_bgcolor': 'rgba(33, 33, 33, 1)',


})
col0, col1 = st.columns(2)


col1.plotly_chart(fig4, use_container_width=True, config=config)

col0.plotly_chart(fig3, use_container_width=True, config=config)

st.subheader("Composite Risk Index")
with st.expander('Details & Formula'):
    st.write("""My intent here was derive a single index value metric as a guide for risk of COVID in Ontario. I choose 'Daily Positive Rate', 'Case Fatality Rate' (calucated metric), 
                and 'ICU Cases with COVID' as my variables. Test positive rate represents the spread in the community, CFR represents the mortal threat to infection, and ICU numbers 
                are a measure of pressure on the healtcare system. I first transformed the values to be on the same scale, then combined by the finding the mean value for each date.   
                Then I used used a MinMax scaler and a multiplier to get the composite value on  measured on a scale of 0-100.\n\n            
                I .""")
st.markdown('<br />​<hr />​​​​​​​​​​​​​​​​​​​<br />', unsafe_allow_html=True)
