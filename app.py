import streamlit as st
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("📊 Vendor Invoice Intelligence Portal")

st.markdown("""
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This portal leverages machine learning to:

- Forecast freight costs accurately
- Detect potentially risky invoices
- Reduce manual invoice reviews
- Improve financial operations
""")

st.divider()

# =====================================================
# KPI DASHBOARD
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Freight Model R²",
        "96.99%"
    )

with col2:
    st.metric(
        "Invoice Model Accuracy",
        "96.66%"
    )

with col3:
    st.metric(
        "Invoices Analysed",
        "5,543"
    )

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

selected_module = st.sidebar.radio(
    "Select Module",
    [
        "🚚 Freight Cost Prediction",
        "🚨 Invoice Risk Flagging"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Business Benefits

✅ Cost Forecasting

✅ Invoice Risk Detection

✅ Reduced Manual Reviews

✅ Faster Finance Operations
""")

# =====================================================
# FREIGHT COST PREDICTION
# =====================================================

if selected_module == "🚚 Freight Cost Prediction":

    st.subheader("🚚 Freight Cost Prediction")

    st.info(
        "Predict freight cost using invoice dollar amount."
    )

    with st.form("freight_form"):

        dollars = st.number_input(
            "Invoice Dollars",
            min_value=1.0,
            value=18500.0,
            step=100.0
        )

        submit_freight = st.form_submit_button(
            "Predict Freight Cost"
        )

    if submit_freight:

        try:

            input_data = {
                "Dollars": [dollars]
            }

            result = predict_freight_cost(
                input_data
            )

            prediction = float(
                result["Predicted_Freight"].iloc[0]
            )

            st.success(
                "Prediction completed successfully."
            )

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    "Invoice Value",
                    f"${dollars:,.2f}"
                )

            with metric_col2:
                st.metric(
                    "Predicted Freight",
                    f"${prediction:,.2f}"
                )

            st.dataframe(
                result,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

# =====================================================
# INVOICE RISK FLAGGING
# =====================================================

else:

    st.subheader(
        "🚨 Invoice Risk Flagging"
    )

    st.warning(
        "Predict whether an invoice should be flagged for manual approval."
    )

    with st.form("invoice_flag_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )

        with col2:

            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:

            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )

        submit_flag = st.form_submit_button(
            "Evaluate Invoice Risk"
        )

    if submit_flag:

        try:

            input_data = {
                "invoice_quantity": [
                    invoice_quantity
                ],
                "invoice_dollars": [
                    invoice_dollars
                ],
                "Freight": [
                    freight
                ],
                "total_item_quantity": [
                    total_item_quantity
                ],
                "total_item_dollars": [
                    total_item_dollars
                ]
            }

            result = predict_invoice_flag(
                input_data
            )

            probability = float(
                result[
                    "Risk_Probability"
                ].iloc[0]
            )

            status = result[
                "Risk_Status"
            ].iloc[0]

            is_flagged = bool(
                result[
                    "Predicted_Flag"
                ].iloc[0]
            )

            st.metric(
                "Risk Probability",
                f"{probability:.2f}%"
            )

            st.progress(
                min(
                    int(probability),
                    100
                )
            )

            if probability >= 80:

                st.error(
                    "🔴 HIGH RISK"
                )

            elif probability >= 50:

                st.warning(
                    "🟠 MEDIUM RISK"
                )

            else:

                st.success(
                    "🟢 LOW RISK"
                )

            if is_flagged:

                st.error(
                    f"🚨 {status}"
                )

            else:

                st.success(
                    f"✅ {status}"
                )

            st.dataframe(
                result,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

# =====================================================
# MODEL INFORMATION
# =====================================================

st.divider()

with st.expander("📘 Model Information"):

    st.markdown("""
### Freight Prediction Model

- Algorithm: Linear Regression
- Features: Dollars
- R² Score: 96.99%

### Invoice Risk Model

- Algorithm: Random Forest Classifier
- Accuracy: 88%

#### Features

- invoice_quantity
- invoice_dollars
- Freight
- total_item_quantity
- total_item_dollars

### Database

SQLite Inventory Database

#### Tables

- vendor_invoice
- purchases
- purchase_prices
- begin_inventory
- end_inventory
""")