"""DemandCast dashboard: historical sales + a scenario-driven forecast, plus
an embedded chat with the AI Analyst agent.

    streamlit run src/demandcast/dashboard/app.py   # or: make dashboard

The Forecast tab predicts in-process (same functions the API uses under
the hood) - no server needed just to see a chart. The AI Analyst tab's
forecast tool calls the live HTTP API instead, since that's what actually
demonstrates agent tool-use across a service boundary; it needs
`make serve` running in another terminal, plus ANTHROPIC_API_KEY set.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandcast.agent.agent import ask, build_agent
from demandcast.data import get_sale_date_bounds, list_store_ids, load_fct_sales
from demandcast.registry import load_latest_model
from demandcast.serving.inference import get_store_history, predict_sales
from demandcast.serving.schemas import PredictRequest

st.set_page_config(page_title="DemandCast", layout="wide")
st.title("DemandCast")

FORECAST_HORIZON_DAYS = 42  # matches the test_weeks=6 holdout used everywhere else


@st.cache_resource
def _cached_model():
    return load_latest_model()


@st.cache_data
def _cached_store_ids() -> list[int]:
    return list_store_ids()


@st.cache_data
def _cached_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    return get_sale_date_bounds()


@st.cache_data(show_spinner=False)
def _cached_history(store_id: int) -> pd.DataFrame:
    return load_fct_sales(store_id=store_id)


tab_forecast, tab_chat = st.tabs(["Forecast", "AI Analyst"])

with tab_forecast:
    store_ids = _cached_store_ids()
    default_index = store_ids.index(262) if 262 in store_ids else 0
    store_id = st.selectbox("Store", options=store_ids, index=default_index)

    history_df = _cached_history(store_id)
    fig = go.Figure()
    fig.add_scatter(
        x=history_df["sale_date"], y=history_df["sales"], mode="lines", name="Actual sales"
    )
    fig.update_layout(
        title=f"Store {store_id} - historical sales", xaxis_title="Date", yaxis_title="Sales"
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Scenario")
    min_date, max_date = _cached_date_bounds()
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_date = st.date_input(
            "Forecast date",
            value=(max_date + pd.Timedelta(days=1)).date(),
            min_value=(min_date + pd.Timedelta(days=45)).date(),
            max_value=(max_date + pd.Timedelta(days=FORECAST_HORIZON_DAYS)).date(),
            help=f"History only runs through {max_date.date()}; dates beyond that "
            "extrapolate using the same trailing window, so this is capped at "
            f"{FORECAST_HORIZON_DAYS} days past it.",
        )
    with col2:
        is_promo = st.toggle("Simulate promo running", value=False)
        is_school_holiday = st.toggle("Simulate school holiday", value=False)
    with col3:
        manual_adjustment_pct = st.slider(
            "Manual adjustment (%)",
            min_value=-50,
            max_value=50,
            value=0,
            help="A manual override applied on top of the model's prediction - "
            "not model-driven, unlike the promo/holiday toggles above.",
        )

    if st.button("Forecast", type="primary"):
        try:
            model = _cached_model()
            store_history = get_store_history(store_id, forecast_date)
            request = PredictRequest(
                store_id=store_id,
                date=forecast_date,
                is_promo=is_promo,
                is_school_holiday=is_school_holiday,
            )
            base_prediction = predict_sales(model, store_history, request)
        except ValueError as exc:
            st.error(str(exc))
        else:
            metric_cols = st.columns(2)
            metric_cols[0].metric("Model prediction", f"{base_prediction:,.0f}")
            if manual_adjustment_pct != 0:
                adjusted = base_prediction * (1 + manual_adjustment_pct / 100)
                metric_cols[1].metric(
                    f"Manually adjusted ({manual_adjustment_pct:+d}%)", f"{adjusted:,.0f}"
                )

with tab_chat:
    st.caption(
        "Needs the forecast API running separately (`make serve`) and "
        "ANTHROPIC_API_KEY set in the environment."
    )

    if "agent" not in st.session_state:
        try:
            st.session_state.agent = build_agent()
            st.session_state.agent_error = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.session_state.agent = None
            st.session_state.agent_error = str(exc)

    if st.session_state.agent_error:
        st.error(f"AI Analyst agent unavailable: {st.session_state.agent_error}")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(content)

        question = st.chat_input("Ask about DemandCast's models, metrics, or a store's sales")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask(st.session_state.agent, question)
                    except Exception as exc:  # noqa: BLE001 - shown to the user, not swallowed
                        answer = f"Something went wrong answering that: {exc}"
                st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))
