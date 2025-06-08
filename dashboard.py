# --- Imports ---
import streamlit as st
import pandas as pd
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
st.markdown("This dashboard visualizes demand forecasts and provides inventory policy recommendations for each product SKU.")

# --- Data Loading ---
@st.cache_data(show_spinner="Loading summary data...")
def load_summary_data(filepath):
    """Loads the main forecast summary results."""
    try:
        df = pd.read_csv(filepath)
        df['last_processed_date'] = pd.to_datetime(df['last_processed_date'])
        return df
    except FileNotFoundError:
        st.error(f"❌ Summary file not found: `{filepath}`. Please run the forecasting pipeline first.", icon="🔥")
        return None

@st.cache_data(show_spinner="Loading daily forecast data...")
def load_daily_data(filepath):
    """Loads the detailed daily forecast data."""
    try:
        df = pd.read_csv(filepath)
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    except FileNotFoundError:
        return None # Don't show an error, as the summary file is the primary one.

# --- Helper Functions for Display ---

def display_overview(df):
    """Displays high-level metrics for all SKUs."""
    st.header("Dashboard Overview", divider="rainbow")
    
    total_skus = df['sku_id'].nunique()
    successful_skus = df[df['error'].isna()]['sku_id'].nunique()
    ensemble_models = df[df['final_forecast_model_type'] == 'Ensemble']['sku_id'].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total SKUs Processed", f"{total_skus}")
    col2.metric("SKUs with Successful Forecasts", f"{successful_skus}", delta=f"{((successful_skus/total_skus)*100):.1f}%")
    col3.metric("SKUs using Ensemble Model", f"{ensemble_models}", delta=f"{((ensemble_models/total_skus)*100):.1f}%")


def display_inventory_metrics(sku_data):
    """Renders the inventory policy recommendation card."""
    st.subheader("🔑 Inventory Policy")
    with st.container(border=True):
        if pd.notna(sku_data.get('error')):
            st.warning(f"Could not generate inventory metrics. **Reason:** {sku_data['error']}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                label="Reorder Point (Units)",
                value=f"{sku_data.get('reorder_point', 0):,.0f}",
                help="When on-hand stock reaches this level, place a new order."
            )
            col2.metric(
                label="Order Quantity (EOQ)",
                value=f"{sku_data.get('validated_eoq', 0):,.0f}",
                help="The recommended order size, optimized for cost and MOQ."
            )
            col3.metric(
                label="Safety Stock (Units)",
                value=f"{sku_data.get('safety_stock', 0):,.0f}",
                help="Buffer stock to prevent stockouts from demand or supply variability."
            )
            col4.metric(
                label="Avg. Lead Time (Days)",
                value=f"{sku_data.get('avg_lead_time', 'N/A')}",
                help="The average supplier lead time used for calculations."
            )
            
            # Display model type and last processed date
            model_type = sku_data.get('final_forecast_model_type', 'Prophet')
            last_date = sku_data['last_processed_date'].strftime('%d %b %Y')
            st.info(f"**Model Used:** `{model_type}` | **Last Processed Date:** `{last_date}`", icon="🤖")

            with st.expander("Learn about these inventory terms"):
                st.markdown("""
                - **Reorder Point (ROP):** The inventory level that triggers a replenishment order. It's calculated as `(Forecasted Daily Demand × Lead Time) + Safety Stock`.
                - **Economic Order Quantity (EOQ):** The ideal order quantity a company should purchase to minimize inventory costs such as holding costs, shortage costs, and order costs.
                - **Safety Stock:** Extra quantity of a product which is stored in the warehouse to prevent an out-of-stock situation.
                """)

def display_forecast_totals(sku_data):
    """Renders the demand forecast totals card."""
    st.subheader("📈 Demand Forecast")
    with st.container(border=True):
        if pd.notna(sku_data.get('error')):
            st.warning("Could not generate demand forecasts.")
        else:
            # Use st.container() to create a row layout
            with st.container():
                st.metric("Next 1 Month Forecast", f"{sku_data.get('forecast_next_1_month', 0):,.0f} units")
                st.metric("Next 3 Months Forecast", f"{sku_data.get('forecast_next_3_months', 0):,.0f} units")
                st.metric("Next 6 Months Forecast", f"{sku_data.get('forecast_next_6_months', 0):,.0f} units")

def display_forecast_chart(daily_df):
    """Renders the interactive forecast chart."""
    st.subheader("📊 Forecast Visualization")
    
    if daily_df is None or daily_df.empty:
        st.info("No detailed daily forecast data is available to plot for this SKU.")
        return

    with st.container(border=True):
        time_agg = st.radio(
            "View forecast by:",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Resample data based on selection
        resample_rule = {'Daily': 'D', 'Weekly': 'W-MON', 'Monthly': 'MS'}[time_agg]
        chart_df = daily_df.set_index('ds').resample(resample_rule).sum().reset_index()

        # Create Altair chart
        base = alt.Chart(chart_df).encode(x=alt.X('ds:T', title='Date'))
        
        forecast_line = base.mark_line(point=True, color='#0068c9').encode(
            y=alt.Y('yhat:Q', title='Forecasted Demand (Units)'),
            tooltip=[
                alt.Tooltip('ds:T', title='Date'),
                alt.Tooltip('yhat:Q', title='Forecast', format=',.0f')
            ]
        )
        
        # In this example, yhat_lower/upper are not in daily data, so we only plot the main forecast
        # If they were present, a confidence band could be added like this:
        # band = base.mark_area(opacity=0.3).encode(y='yhat_lower:Q', y2='yhat_upper:Q')
        # chart = band + forecast_line
        
        st.altair_chart(forecast_line.interactive(), use_container_width=True)


# --- Main Application ---
summary_df = load_summary_data("./home/jupyter/multi_sku_forecasting_final.csv")
daily_df_all = load_daily_data("./home/jupyter/daily_forecasts_final.csv")

if summary_df is not None:
    # --- Sidebar for SKU Selection ---
    st.sidebar.header("⚙️ Select SKU")
    sku_list = sorted(summary_df['sku_id'].unique())
    selected_sku = st.sidebar.selectbox(
        "Choose an SKU to analyze:", sku_list, index=0
    )
    
    # --- Main Content Area ---
    display_overview(summary_df)
    
    st.header(f"Analysis for SKU: `{selected_sku}`", divider="rainbow")

    # Filter data for the selected SKU
    sku_data = summary_df[summary_df['sku_id'] == selected_sku].iloc[0]
    
    if daily_df_all is not None:
        daily_data_filtered = daily_df_all[daily_df_all['sku_id'] == selected_sku]
    else:
        daily_data_filtered = None

    # Display components
    col_left, col_right = st.columns([1, 1])
    with col_left:
        display_inventory_metrics(sku_data)
    with col_right:
        display_forecast_totals(sku_data)

    st.markdown("---") # Visual separator
    display_forecast_chart(daily_data_filtered)

else:
    st.info("Awaiting the generation of the forecasting data file...")
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2RtcXBmNWp5N2VlZ3RsbDVud2Z2OWs0c2hsZDNtMm5pZzY5d2FndCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0HlBOJaAgDqfKAfS/giphy.gif")