import streamlit as st
import pandas as pd
import numpy as np
import ast  # To safely evaluate the string representation of lists
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="Demand & Inventory Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title ---
st.title('📦 SKU Demand & Inventory Dashboard')
st.markdown("This dashboard visualizes the demand forecast and inventory recommendations from the forecasting model.")

# --- Load Data ---
@st.cache_data
def load_data(filepath):
    """Loads the forecast results CSV, returns a DataFrame."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at '{filepath}'. Please run the main forecasting script first to generate the results CSV.")
        return None

# Use the absolute path for your Vertex AI environment
output_csv_path = "./multi_sku_forecasting_final.csv"

# --- Load Data with Spinner ---
with st.spinner("Loading data..."):
    data_df = load_data(output_csv_path)

if data_df is not None:
    # Print the columns to debug
    st.write("Columns in the loaded data:", data_df.columns.tolist())

    # Check if the expected columns are in the DataFrame
    if 'forecast_summary' not in data_df.columns:
        st.error("Missing 'forecast_summary' column in the data.")
    else:
        # --- Parse the forecast_summary column ---
        all_forecasts = []
        for index, row in data_df.iterrows():
            forecasts = ast.literal_eval(row['forecast_summary'])  # Convert string to list of dicts
            for forecast in forecasts:
                all_forecasts.append(forecast)

        # Create a DataFrame from the parsed forecasts
        forecast_df = pd.DataFrame(all_forecasts)

        # --- Aggregate Daily Forecasts to Monthly ---
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])  # Ensure 'ds' is datetime
        forecast_df.set_index('ds', inplace=True)
        monthly_forecasts = forecast_df.resample('M').sum()  # Aggregate to monthly totals

        # --- Calculate Monthly Statistics ---
        monthly_stats = monthly_forecasts.agg(
            highest=('yhat', 'max'),
            lowest=('yhat', 'min'),
            average=('yhat', 'mean')
        )

        # --- Display Monthly Statistics Table ---
        st.subheader("Monthly Forecast Statistics Table")
        st.dataframe(monthly_stats)

        # --- Visualize Monthly Statistics ---
        st.subheader("Monthly Forecast Visualization")
        st.line_chart(monthly_stats, use_container_width=True)

    # --- Sidebar for SKU Selection ---
    st.sidebar.header("Select Product SKU")
    
    sku_list = sorted(data_df['sku_id'].unique())
    
    selected_sku = st.sidebar.selectbox(
        "Choose an SKU to analyze:",
        sku_list,
        index=0 
    )

    st.header(f"Analysis for SKU: `{selected_sku}`")
    
    sku_data = data_df[data_df['sku_id'] == selected_sku].iloc[0]

    # --- Display Key Inventory Metrics ---
    st.subheader("Inventory Policy Recommendations")

    if pd.notna(sku_data.get('error')) or pd.notna(sku_data.get('inventory_forecast_error')):
        error_message = sku_data.get('error', sku_data.get('inventory_forecast_error', 'An unknown error occurred.'))
        st.warning(f"Could not generate inventory metrics for this SKU. Reason: **{error_message}**")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="**Reorder Point** (Units)", value=f"{sku_data.get('reorder_point', 0):.0f}", help="When on-hand stock hits this level, place a new order.")
        with col2:
            st.metric(label="**Validated EOQ** (Order Qty)", value=f"{sku_data.get('validated_eoq', 0):.0f}", help="The most cost-effective quantity to order.")
        with col3:
             st.metric(label="**Safety Stock** (Units)", value=f"{sku_data.get('safety_stock', 0):.0f}", help="Buffer stock to prevent stockouts.")
        
        st.info(f"The model used for this forecast was: **{sku_data.get('final_forecast_model_type', 'N/A')}**", icon="🤖")

    # --- Display Demand Forecasts ---
    st.subheader("Demand Forecast (Next 6 Months)")
    
    if pd.notna(sku_data.get('error')) or pd.notna(sku_data.get('inventory_forecast_error')):
        st.warning("Could not generate demand forecasts for this SKU.")
    else:
        col_fcst_1, col_fcst_2, col_fcst_3 = st.columns(3)
        with col_fcst_1:
            st.metric(label="**Next 1 Month** Forecast", value=f"{sku_data.get('forecast_next_1_month', 0):.0f} units")
        with col_fcst_2:
            st.metric(label="**Next 3 Months** Forecast", value=f"{sku_data.get('forecast_next_3_months', 0):.0f} units")
        with col_fcst_3:
            st.metric(label="**Next 6 Months** Forecast", value=f"{sku_data.get('forecast_next_6_months', 0):.0f} units")

    # --- **NEW**: Display Real Forecast Chart ---
    st.subheader("Forecast Visualization")
    
    # if daily_forecasts_df is not None:
    #     sku_forecast_data = daily_forecasts_df[daily_forecasts_df['sku_id'] == selected_sku]
    #     
    #     if not sku_forecast_data.empty:
    #         
    #         base = alt.Chart(sku_forecast_data).encode(x='ds:T')
    #         
    #         # Create the uncertainty band
    #         band = base.mark_area(opacity=0.3, color='#57A44C').encode(
    #             y=alt.Y('yhat_lower', title='Forecasted Sales'),
    #             y2='yhat_upper'
    #         ).properties(
    #             title=f'6-Month Demand Forecast for {selected_sku}'
    #         )
    #
    #         # Create the main forecast line
    #         line = base.mark_line(color='#0068C9').encode(
    #             y=alt.Y('yhat', title='Forecasted Sales')
    #         )
    #         
    #         # Combine the chart layers
    #         chart = (band + line).interactive()
    #         
    #         st.altair_chart(chart, use_container_width=True)
    #
    #     else:
    #         st.warning("No daily forecast data available for this SKU.")
    # else:
    #     st.info("Run the main forecasting script to generate the data needed for this chart.")

    # --- Show Raw Data for the SKU ---
    with st.expander("Show Raw Data for this SKU"):
        st.dataframe(sku_data.to_frame().T)

else:
    st.info("Waiting for the forecasting data file to be generated...")

