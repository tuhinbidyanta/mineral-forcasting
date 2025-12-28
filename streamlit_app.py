import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from domestic import domestic_page,hybrid

### Copper Data Load
st.set_page_config(
    page_title="Mineral Forecasting Dashboard",
    layout="wide"
)
st.title("🔶 Team Critical Thinker")


st.subheader("Domestic Data production and foreign trade")
copper = pd.read_excel("./mineral_data.xlsx",sheet_name="Copper")
years  = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
copper['Year'] = years

# --------------------------------
# PAGE CONFIG
# --------------------------------

# Dash Board Secction
st.markdown("Domestic Data production and foreign trade of Copper in India")
col1, col2 = st.columns(2)
with col1:
    option = st.selectbox(
        "Select Year",
        years,
        key="copper_year"
    )

    st.write("You selected:", option)
with col2:
    result = eval(
    copper['State of Production'].values[0],
    {"array": np.array, "object": object, "__builtins__": {}}
    )
    # st.write(list(result.keys()))
    option_state = st.selectbox(
        "Select State",
        list(result.keys()),
        key="copper_state"
    )
    st.write("You selected:", option_state)
fig = domestic_page(copper,option_state,option)
col1,col2,col3 = st.columns(3)
with col1:
    st.plotly_chart(fig[0],width='stretch')
with col2:
    st.plotly_chart(fig[1],width='stretch')
with col3:
    st.plotly_chart(fig[2],width='stretch')






### Tin
tin = pd.read_excel("./mineral_data.xlsx",sheet_name="Tin")
years  = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
tin['Year'] = years
st.markdown("Domestic Data production and foreign trade of Tin in India")
col1, col2 = st.columns(2)
with col1:
    option_tin = st.selectbox(
        "Select Year",
        tin['Year'].tolist(),
        key="tin_year"
    )

    st.write("You selected:", option_tin)
with col2:
    result = eval(
    tin['State of Production'].values[0],
    {"array": np.array, "object": object, "__builtins__": {}}
    )
    # st.write(list(result.keys()))
    option_state_tin = st.selectbox(
        "Select State",
        list(result.keys()),
        key="tin_state"
    )
    st.write("You selected:", option_state_tin)
fig = domestic_page(tin,option_state_tin,option_tin)
col1,col2,col3 = st.columns(3)
with col1:
    st.plotly_chart(fig[0],width='stretch')
with col2:
    st.plotly_chart(fig[1],width='stretch')
with col3:
    st.plotly_chart(fig[2],width='stretch')




### Graphite
graphite = pd.read_excel("./mineral_data.xlsx",sheet_name="Graphite")
years  = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
graphite['Year'] = years
st.markdown("Domestic Data production and foreign trade of Graphite in India")
col1, col2 = st.columns(2)
with col1:
    option_g = st.selectbox(
        "Select Year",
        graphite['Year'].tolist(),
        key="Graphite_year"
    )

    st.write("You selected:", option_g)
with col2:
    result = eval(
    graphite['State of Production'].values[0],
    {"array": np.array, "object": object, "__builtins__": {}}
    )
    # st.write(list(result.keys()))
    option_state_g = st.selectbox(
        "Select State",
        list(result.keys()),
        key="graphite_state"
    )
    st.write("You selected:", option_state_g)
fig = domestic_page(graphite,option_state_g,option_g)
col1,col2,col3 = st.columns(3)
with col1:
    st.plotly_chart(fig[0],width='stretch')
with col2:
    st.plotly_chart(fig[1],width='stretch')
with col3:
    st.plotly_chart(fig[2],width='stretch')




### Phosphorus
phosphorus = pd.read_excel("./mineral_data.xlsx",sheet_name="Phosphorus")
# years  = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
phosphorus['Year'] = years
st.markdown("Domestic Data production and foreign trade of Phosphorus in India")
col1, col2 = st.columns(2)
with col1:
    option_p = st.selectbox(
        "Select Year",
        phosphorus['Year'].tolist(),
        key="Phos_year"
    )

    st.write("You selected:", option_p)
with col2:
    result = eval(
    phosphorus['State of Production'].values[0],
    {"array": np.array, "object": object, "__builtins__": {}}
    )
    # st.write(list(result.keys()))
    option_state_p = st.selectbox(
        "Select State",
        list(result.keys()),
        key="phos_state"
    )
    st.write("You selected:", option_state_p)
fig = domestic_page(phosphorus,option_state_p,option_p)
col1,col2,col3 = st.columns(3)
with col1:
    st.plotly_chart(fig[0],width='stretch')
with col2:
    st.plotly_chart(fig[1],width='stretch')
with col3:
    st.plotly_chart(fig[2],width='stretch')







### mineral Forecasting
df1 = pd.read_excel('./Data/Yaerwise EX-IM trade.xlsx')
dat  = pd.DataFrame()
# df1.head()
imp = []
exp = []
for i in range(1,len(df1.iloc[1:,1:])+1):
    exp.append((df1.iloc[i,2::2].values).astype('float32'))
    imp.append((df1.iloc[i,3::2].values).astype('float32'))

dat['Mineral'] = (df1.iloc[1:,1].values).astype('str')
dat['Export'] = exp
dat['Import'] = imp

st.subheader("mineral Forcasting")
# dat = pd.read_csv('./processed_data.csv')
col1,col2 = st.columns(2)
with col1:

    mineral = st.selectbox("Select Mineral",dat['Mineral'].unique(),key="mineral_forecast")
with col2:
    year_count = st.selectbox("Select prediction Year Count", [2,4,6],key="year_count")
# st.write(hybrid(dat[dat["Mineral"]==mineral],step=year_count))
st.write(dat[dat["Mineral"]==mineral]['Export'])
st.write(dat[dat["Mineral"]==mineral]['Import'].apply(lambda x: np.round(x, 2)))
export_forecast, import_forecast = hybrid(dat[dat["Mineral"]==mineral], step=year_count)

# Set negative values in export/import lists to zero
export_forecast = [np.where(np.array(f) < 0, 0, f) for f in export_forecast]
import_forecast = [np.where(np.array(f) < 0, 0, f) for f in import_forecast]

# st.write(f"Export forecast for next {year_count} years using (ARIMA): {export_forecast[1]}")
# st.write(f"Export forecast for next {year_count} years using hybrid(ARIMA+LSTM): {export_forecast[0]}")
# st.write(f"Import forecast for next {year_count} years using (ARIMA): {import_forecast[1]}")
# st.write(f"Import forecast for next {year_count} years using hybrid(ARIMA+LSTM): {import_forecast[0]}")

# st.write(f"Import forecast for next {yaear_count} years: {import_forecast}")
fig = go.Figure()
# Plot historical Export data
fig.add_scatter(
        x=[i+2017 for i in range(len(dat[dat["Mineral"]==mineral]['Export'].values[0]))],
        y=dat[dat["Mineral"]==mineral]['Export'].values[0],
        mode="markers+lines+text",
        name="Export (Historical)",
        fill='tozeroy',
        fillcolor="rgba(100, 200, 100, 0.3)"
)
# Plot Export Forecast (hybrid)
fig.add_scatter(
        x=[i for i in range(2025, 2025+year_count)],
        y=export_forecast[0],
        mode="markers+lines+text",
        name="Export Forecast (Hybrid)",
        fill='tozeroy',
        fillcolor="rgba(100, 200, 100, 0.15)"
)
# Plot Export Forecast (ARIMA)
fig.add_scatter(
        x=[i for i in range(2025, 2025+year_count)],
        y=export_forecast[1],
        mode="markers+lines+text",
        name="Export Forecast (ARIMA)",
        fill='tozeroy',
        fillcolor="rgba(200, 100, 100, 0.15)"
)

fig.update_layout(height=450, showlegend=True)
fig.update_yaxes(title_text = "Export Million Dollar")

fig2 = go.Figure()
# Plot historical Export data
fig2.add_scatter(
        x=[i+2017 for i in range(len(dat[dat["Mineral"]==mineral]['Import'].values[0]))],
        y=dat[dat["Mineral"]==mineral]['Import'].values[0],
        mode="markers+lines+text",
        name="Import (Historical)",
        fill='tozeroy',
        fillcolor="rgba(100, 200, 100, 0.3)"
)
# Plot Export Forecast (hybrid)
fig2.add_scatter(
        x=[i for i in range(2025, 2025+year_count)],
        y=import_forecast[0],
        mode="markers+lines+text",
        name="Import Forecast (Hybrid)",
        fill='tozeroy',
        fillcolor="rgba(100, 200, 100, 0.15)"
)
# Plot Export Forecast (ARIMA)
fig2.add_scatter(
        x=[i for i in range(2025, 2025+year_count)],
        y=import_forecast[1],
        mode="markers+lines+text",
        name="Import Forecast (ARIMA)",
        fill='tozeroy',
        fillcolor="rgba(200, 100, 100, 0.15)"
)

fig2.update_layout(height=450, showlegend=True)
fig2.update_yaxes(title_text = "Import Million Dollar")
col1,col2 = st.columns(2)
with col1:
    st.plotly_chart(fig, width='stretch')
with col2:
    st.plotly_chart(fig2, width='stretch')



st.subheader("Country Wise Import Export Data  and Dependency Analysis")
c_data_import = pd.read_excel("./countrywise_import_export_data.xlsx",sheet_name="Import")
c_data_export = pd.read_excel("./countrywise_import_export_data.xlsx",sheet_name="Export")
# st.dataframe(c_data)
# miner = eval(
# c_data['minerals'].values[0],
# {"array": np.array, "object": object, "__builtins__": {}}
# )
# st.write()

miner  = st.selectbox("Select Mineral",c_data_export['minerals'].unique(),key="country_mineral")
year_country = st.selectbox("Select Year Count for Forecast",range(2017,2026),key="year_count_country")
# st.dataframe(c_data_import['minerals']==miner)
def country_page(df,mineral):
    data_mineral = df[df["minerals"] == mineral]
    country = eval(
        data_mineral['countries'].values[0],
        {"array": np.array, "object": object, "__builtins__": {}}
        )
    usmillion = eval(
        data_mineral['usmillion'].values[0],
        {"array": np.array, "object": object, "__builtins__": {}}
        )
    volume = eval(
        data_mineral['volume'].values[0],
        {"array": np.array, "object": object, "__builtins__": {}}
        )
    return country,usmillion,volume
# st.dataframe(country_page(c_data_import,miner)[0])
col1,col2 = st.columns(2)
with col1:
    fig1_ = go.Figure()
    fig1_.add_bar(
        x=country_page(c_data_import,miner)[0][year_country-2017],
        y=country_page(c_data_import,miner)[1][year_country-2017],
        name="Import US Million"
    )
    fig1_.update_layout(title_text=f"Country Wise Import Data of {miner} in {year_country}")
    fig1_.update_yaxes(title_text = "Import US Million Dollar")   
    st.plotly_chart(fig1_, width='stretch')
with col2:
    fig2_ = go.Figure()
    fig2_.add_bar(
        x=country_page(c_data_export,miner)[0][year_country-2017],
        y=country_page(c_data_export,miner)[1][year_country-2017],
        name="Export US Million"
    )
    fig2_.update_layout(title_text=f"Country Wise Export Data of {miner} in {year_country}")
    fig2_.update_yaxes(title_text = "Export US Million Dollar")   
    st.plotly_chart(fig2_, width='stretch')
col1,col2 = st.columns(2)
with col1:
    fig3_ = go.Figure()
    fig3_.add_bar(
        x=country_page(c_data_import,miner)[0][year_country-2017],
        y=country_page(c_data_import,miner)[2][year_country-2017],
        name="Import Volume"
    )
    fig3_.update_layout(title_text=f"Country Wise Import Volume Data of {miner} in {year_country}")
    fig3_.update_yaxes(title_text = "Import Volume")   
    st.plotly_chart(fig3_, width='stretch')
with col2:
    fig4_ = go.Figure()
    fig4_.add_bar(
        x=country_page(c_data_export,miner)[0][year_country-2017],
        y=country_page(c_data_export,miner)[2][year_country-2017],
        name="Export Volume"
    )
    fig4_.update_layout(title_text=f"Country Wise Export Volume Data of {miner} in {year_country}")
    fig4_.update_yaxes(title_text = "Export Volume")   
    st.plotly_chart(fig4_, width='stretch')

st.markdown(f'#### The maximum Export of {miner} is from {country_page(c_data_export,miner)[0][year_country-2017][np.argmax(country_page(c_data_export,miner)[1][year_country-2017])]}')
st.markdown(f'#### The maximum Import of {miner} is from {country_page(c_data_import,miner)[0][year_country-2017][np.argmax(country_page(c_data_import,miner)[1][year_country-2017])]}')

# --------------------------------
# LOAD DATA (REPLACE WITH YOUR NOTEBOOK CODE)
# --------------------------------
# Example structure – replace with your real data source
# data = pd.read_csv("copper_data.csv")

# Dummy placeholders (remove when using real data)
# years = [2020, 2021, 2022]
# states = ["Odisha", "Jharkhand", "Chhattisgarh"]

# --------------------------------
# SIDEBAR CONTROLS
# --------------------------------
# st.sidebar.header("Controls")

# selected_year = st.sidebar.selectbox("Select Year", years)
# selected_state = st.sidebar.selectbox("Select State", states)

# --------------------------------
# DATA PROCESSING (ADAPT FROM NOTEBOOK)
# --------------------------------
# Replace these with your actual calculations
# total_import = 120
# total_export = 95

# months = ['jan','feb','mar','apr','may','jun',
#           'jul','aug','sep','oct','nov','dec']
# total_production = [10,15,14,18,20,22,25,23,21,19,16,14]
# state_production = [4,6,5,7,8,9,10,9,8,7,6,5]

# # --------------------------------
# # MAIN DASHBOARD
# # --------------------------------
# col1, col2 = st.columns(2)

# # ---- Import / Export + Monthly Production ----
# with col1:
#     fig1 = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=[
#             f"Import & Export ({selected_year})",
#             f"Monthly Production ({selected_year})"
#         ]
#     )

#     fig1.add_trace(
#         go.Bar(
#             x=["Import", "Export"],
#             y=[total_import, total_export],
#             name="Trade"
#         ),
#         row=1, col=1
#     )

#     fig1.add_trace(
#         go.Scatter(
#             x=months,
#             y=total_production,
#             mode="markers+lines+text",
#             fill="tozeroy",
#             name="Production"
#         ),
#         row=1, col=2
#     )

#     fig1.update_layout(height=400)
#     st.plotly_chart(fig1, width='stretch')

# # ---- State-wise Production ----
# with col2:
#     fig2 = go.Figure(
#         go.Bar(
#             x=months,
#             y=state_production,
#             marker_color="orange"
#         )
#     )

#     fig2.update_layout(
#         title=f"Monthly Production in {selected_state} ({selected_year})",
#         yaxis_title="Quantity",
#         height=400
#     )

#     st.plotly_chart(fig2, width='stretched')

# # --------------------------------
# # FOOTER
# # --------------------------------
# st.markdown("---")
# st.caption("📊 Interactive dashboard built with Streamlit & Plotly")
