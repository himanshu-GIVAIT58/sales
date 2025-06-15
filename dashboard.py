# --- Imports ---
import streamlit as st
import pandas as pd
import altair as alt
from typing import Optional, Dict, Any, List
import re
import google.generativeai as genai
from streamlit_chat import message # You might need to run: pip install streamlit-chat

# --- Page Configuration ---
st.set_page_config(
    page_title="Demand & Inventory Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chatbot Configuration ---
# IMPORTANT: Replace with your actual Gemini API Key.
# For deployment, use Streamlit Secrets: st.secrets["GEMINI_API_KEY"]
try:
    GEMINI_API_KEY = "AIzaSyAnxwxedQhBa07RJSVq6r6BlfpsypVY4nM" # <--- PASTE YOUR KEY HERE
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
    CHATBOT_ENABLED = True
except Exception as e:
    st.warning(f"⚠️ Gemini API key not configured. Chatbot will be disabled. Error: {e}", icon="🔑")
    CHATBOT_ENABLED = False

# --- Constants ---
SUMMARY_FILEPATH = "./home/jupyter/multi_sku_forecasting_final (1).csv"
DAILY_FILEPATH = "./home/jupyter/daily_forecasts_final (1).csv"
NUMERIC_COLUMNS = [
    'rmse', 'mape', 'wape', 'accuracy', 'historical_avg_sales',
    'tuned_prophet_rmse', 'tuned_prophet_wape', 'lstm_rmse', 'lstm_wape',
    'xgboost_rmse', 'xgboost_wape'
]
MODEL_PREFIXES = {
    'Tuned Prophet': 'tuned_prophet',
    'LSTM': 'lstm',
    'XGBoost': 'xgboost'
}

# --- Data Loading ---
@st.cache_data(show_spinner="Loading summary data...")
def load_summary_data(filepath: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(filepath)
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['last_processed_date'] = pd.to_datetime(df['last_processed_date'], errors='coerce')
        for col in ['top_features', 'top_importances']:
            if col in df.columns and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and '[' in x else x)
        return df
    except FileNotFoundError:
        st.error(f"❌ **File Not Found:** `{filepath}`. Ensure the data file is in the same directory.", icon="🔍")
        return None
    except Exception as e:
        st.error(f"⚠️ An error occurred while loading summary data: {str(e)}", icon="🚨")
        return None

@st.cache_data(show_spinner="Loading daily forecast data...")
def load_daily_data(filepath: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(filepath)
        df['ds'] = pd.to_datetime(df['ds'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        invalid_rows_mask = df['ds'].isna()
        if invalid_rows_mask.any():
            st.warning(f"⚠️ Found {invalid_rows_mask.sum()} rows with unparseable dates, which were excluded.", icon="📅")
            df = df.dropna(subset=['ds'])
        if not df.empty:
            df['ds'] = df['ds'].dt.normalize()
        return df
    except FileNotFoundError:
        st.info("ℹ️ Daily forecast file not found. Proceeding without daily data.", icon="📂")
        return None
    except Exception as e:
        st.error(f"⚠️ Failed to load daily forecast data: {str(e)}", icon="📉")
        return None

# --- Chatbot Helper Functions ---
def find_sku_in_question(question: str, available_skus: List[str]) -> Optional[str]:
    """Finds the first valid SKU ID mentioned in the user's question."""
    question_upper = question.upper()
    for sku in available_skus:
        if sku.upper() in question_upper:
            return sku
    return None

def get_data_context(sku_id: str, df_summary: pd.DataFrame, df_daily: Optional[pd.DataFrame], forecast_days: int = 30) -> str:
    """Retrieves and formats all relevant data for a given SKU into a string for the LLM."""
    summary_data = df_summary[df_summary['sku_id'] == sku_id]
    if summary_data.empty:
        return f"Error: No summary data found for SKU '{sku_id}'."
    summary_data = summary_data.iloc[0]

    daily_data = df_daily[df_daily['sku_id'] == sku_id].head(forecast_days) if df_daily is not None else pd.DataFrame()

    context = f"""
    Here is the available data for SKU '{sku_id}':
    - Champion Model: {summary_data.get('final_forecast_model_type', 'N/A')}
    - Model Accuracy (100 - WAPE): {summary_data.get('accuracy', 0):.2f}%
    - Reorder Point: {summary_data.get('reorder_point', 'N/A')} units
    - Safety Stock: {summary_data.get('safety_stock', 'N/A')} units
    - Next 30-Day Total Forecast: {summary_data.get('forecast_next_1_month', 0):,.0f} units
    - Next 90-Day Total Forecast: {summary_data.get('forecast_next_3_months', 0):,.0f} units
    - Daily Forecast for the next {forecast_days} days:
    {daily_data[['ds', 'yhat']].to_string(index=False) if not daily_data.empty else 'Not available.'}
    """
    return context

def get_chatbot_response(question: str, df_summary: pd.DataFrame, df_daily: Optional[pd.DataFrame]) -> str:
    """Generates a response from the LLM based on the user's question and retrieved data."""
    available_skus = df_summary['sku_id'].unique().tolist()
    sku_id = find_sku_in_question(question, available_skus)

    if not sku_id:
        return f"I'm sorry, I couldn't identify a valid SKU in your question. Please include one of the available SKUs, like `{available_skus[0]}`."

    data_context = get_data_context(sku_id, df_summary, df_daily)
    if "Error:" in data_context:
        return data_context

    prompt = f"""
    You are an expert sales and inventory analyst. Your task is to answer the user's question based *only* on the data provided below.
    Be concise and clear.

    --- DATA FOR SKU {sku_id} ---
    {data_context}
    --- END DATA ---

    User Question: "{question}"

    Your Answer:
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, an error occurred with the AI model: {e}"

# --- UI Helper Functions ---
def get_model_performance(sku_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    performance = {}
    for model_name, prefix in MODEL_PREFIXES.items():
        rmse_col, wape_col = f'{prefix}_rmse', f'{prefix}_wape'
        if rmse_col in sku_data and pd.notna(sku_data[rmse_col]):
            performance[model_name] = {'RMSE': sku_data[rmse_col], 'WAPE': sku_data[wape_col]}
    return performance

def display_main_metrics(sku_data: Dict[str, Any]) -> None:
    st.subheader("Key Recommendations", divider="blue")
    cols = st.columns(4)
    cols[0].metric("Reorder Point", f"{sku_data.get('reorder_point', '0')}")
    cols[1].metric("Order Quantity (EOQ)", f"{sku_data.get('validated_eoq', '0')}")
    cols[2].metric("Safety Stock", f"{sku_data.get('safety_stock', '0')}")
    cols[3].metric("Next 30-Day Forecast", f"{sku_data.get('forecast_next_1_month', 0):,.0f} units")

def display_forecast_chart(daily_df: Optional[pd.DataFrame], hist_avg: float) -> None:
    # This function remains unchanged...
    if daily_df is None or daily_df.empty:
        st.info("ℹ️ No daily forecast data available for visualization.", icon="📊")
        return

    time_agg = st.radio("Aggregate by:", ["Daily", "Weekly", "Monthly"], horizontal=True, index=1, key="time_agg")
    
    resample_rule = {'Daily': 'D', 'Weekly': 'W-MON', 'Monthly': 'MS'}[time_agg]
    chart_df = daily_df.set_index('ds').resample(resample_rule).sum().reset_index()

    base = alt.Chart(chart_df).encode(x=alt.X('ds:T', title='Date'))
    
    band = base.mark_area(opacity=0.3, color='#6B7280').encode(
        y=alt.Y('yhat_lower:Q', title='Forecasted Demand (Units)'),
        y2=alt.Y2('yhat_upper:Q'),
        tooltip=[alt.Tooltip('ds:T', title='Date', format='%Y-%m-%d'), alt.Tooltip('yhat:Q', title='Forecast', format=',.0f'), alt.Tooltip('yhat_lower:Q', title='Lower Bound', format=',.0f'), alt.Tooltip('yhat_upper:Q', title='Upper Bound', format=',.0f')]
    ).interactive()

    forecast_line = base.mark_line(color='#1F2937', size=2).encode(y=alt.Y('yhat:Q'))
    
    avg_line = alt.Chart(pd.DataFrame({'y': [hist_avg]})).mark_rule(color='#9CA3AF', strokeDash=[5, 5]).encode(y='y')
    avg_text = avg_line.mark_text(align='left', dx=5, dy=-10, text='Historical Avg', color='#4B5563').encode(y='y')

    chart = alt.layer(band, forecast_line, avg_line, avg_text).properties(title=f'{time_agg} Demand Forecast', height=400).configure_axis(labelFontSize=12, titleFontSize=14).configure_title(fontSize=16, anchor='start')

    st.altair_chart(chart, use_container_width=True)


def display_model_deep_dive(sku_data: Dict[str, Any]) -> None:
    # This function remains unchanged...
    performance_data = get_model_performance(sku_data)
    
    if not performance_data:
        st.info("ℹ️ No model performance data available for this SKU.", icon="🎯")
        return

    perf_df = pd.DataFrame.from_dict(performance_data, orient='index').reset_index().rename(columns={'index': 'Model'})
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance Comparison", divider="gray")
        perf_melted = perf_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
        chart = alt.Chart(perf_melted).mark_bar().encode(
            x=alt.X('Value:Q', title=None),
            y=alt.Y('Model:N', sort='-x', title=None),
            color=alt.Color('Model:N', scale=alt.Scale(scheme='category10'), legend=None),
            row=alt.Row('Metric:N', title=None, header=alt.Header(labelAngle=0, labelAlign='left'))
        ).properties(height=150).configure_facet(spacing=10)
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.subheader("Feature Importance (XGBoost)", divider="gray")
        if sku_data.get('final_forecast_model_type') == 'XGBoost' and sku_data.get('top_features'):
            imp_df = pd.DataFrame({'Feature': sku_data['top_features'], 'Importance': sku_data['top_importances']})
            imp_chart = alt.Chart(imp_df).mark_bar().encode(
                x=alt.X('Importance:Q', title='Importance Score'),
                y=alt.Y('Feature:N', sort='-x', title=None),
                color=alt.Color(value='#4B5563')
            ).properties(height=340)
            st.altair_chart(imp_chart, use_container_width=True)
        else:
            st.info("ℹ️ Feature importance available only for XGBoost champion model.", icon="📋")

# --- Main Application ---
def main():
    """Main function to orchestrate the Streamlit dashboard."""
    st.title('🔮 SKU Demand & Inventory Dashboard')
    st.markdown("This dashboard visualizes forecasts from a champion-challenger pipeline and provides an AI assistant for analysis.")

    summary_df = load_summary_data(SUMMARY_FILEPATH)
    daily_df_all = load_daily_data(DAILY_FILEPATH)

    if summary_df is None:
        st.info("ℹ️ Awaiting generation of forecasting data. Please ensure the pipeline has run.", icon="⏳")
        return

    # --- NEW: Chatbot UI Section ---
    if CHATBOT_ENABLED:
        with st.expander("💬 Chat with your Data Assistant", expanded=False):
            # Initialize chat history in session state
            if "messages" not in st.session_state:
                st.session_state.messages = []
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Hi! How can I help you analyze the forecast data today?"}
                )

            # Display chat messages from history
            for msg in st.session_state.messages:
                message(msg["content"], is_user=(msg["role"] == "user"))

            # Get user input
            if prompt := st.chat_input("Ask about a specific SKU, e.g., 'What is the reorder point for ER01118?'"):
                # Add user message to history and display
                st.session_state.messages.append({"role": "user", "content": prompt})
                message(prompt, is_user=True)

                # Generate and display assistant response
                with st.spinner("Analyzing..."):
                    response = get_chatbot_response(prompt, summary_df, daily_df_all)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    message(response)
    
    st.markdown("---") # Visual separator

    # --- SKU Selection and Details Section (Original Logic) ---
    with st.sidebar:
        st.header("⚙️ SKU Selection")
        sku_list = sorted(summary_df['sku_id'].unique())
        search_term = st.text_input("Search SKU:", placeholder="Enter SKU ID...")
        filtered_sku_list = [sku for sku in sku_list if search_term.lower() in str(sku).lower()] if search_term else sku_list
        
        selected_sku = st.radio("Select SKU:", filtered_sku_list, index=0, key="sku_select") if filtered_sku_list else None

    if selected_sku:
        st.header(f"Detailed Analysis for SKU: `{selected_sku}`", divider="rainbow")
        sku_data = summary_df[summary_df['sku_id'] == selected_sku].iloc[0].to_dict()
        daily_data_filtered = daily_df_all[daily_df_all['sku_id'] == selected_sku] if daily_df_all is not None else None

        if pd.notna(sku_data.get('error')):
            st.error(f"⚠️ Processing error for this SKU: **{sku_data['error']}**", icon="🚨")
        else:
            display_main_metrics(sku_data)
            tab1, tab2, tab3 = st.tabs(["📈 Forecast Visualization", "🧠 Model Insights", "📋 Data Details"])
            with tab1:
                display_forecast_chart(daily_data_filtered, sku_data.get('historical_avg_sales', 0))
            with tab2:
                display_model_deep_dive(sku_data)
            with tab3:
                st.subheader("Forecast Summary", divider="gray")
                st.dataframe(pd.DataFrame([sku_data]), use_container_width=True)
                st.subheader("Daily Forecast Data", divider="gray")
                st.dataframe(daily_data_filtered, use_container_width=True)
    elif filtered_sku_list:
        st.info("Please select a SKU from the sidebar to see detailed analysis.", icon="👈")


if __name__ == "__main__":
    main()
