# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ARIMA helper
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="African Health Burden Dashboard (2003-2023)",
    layout="wide"
)

# ---------- constants / palettes ----------
PALETTE = px.colors.qualitative.Plotly  # colorful default palette
PALETTE_ALT = px.colors.qualitative.T10

HORIZON_DEFAULT = 5

# ---------- ARIMA helper with safe fallbacks ----------
def fit_arima_with_fallback(y, horizon=HORIZON_DEFAULT):
    y = pd.Series(y, dtype="float64").dropna()
    n = len(y)
    if n < 6:
        last = float(y.iloc[-1]) if n else 0.0
        pred = np.full(horizon, last, dtype=float)
        return pred, pred * 0.9, pred * 1.1
    for order in [(1,1,1), (0,1,1), (1,1,0), (0,1,0)]:
        try:
            m = ARIMA(y.values, order=order, enforce_stationarity=False, enforce_invertibility=False)
            fit = m.fit()
            fc = fit.get_forecast(steps=horizon)
            pred = fc.predicted_mean.to_numpy()
            conf = fc.conf_int(alpha=0.05)
            lower = conf.iloc[:, 0].to_numpy()
            upper = conf.iloc[:, 1].to_numpy()
            return pred, lower, upper
        except Exception:
            continue
    try:
        x = np.arange(n)
        b1, b0 = np.polyfit(x, y.values, 1)
        x_f = np.arange(n, n + horizon)
        pred = (b1 * x_f) + b0
        resid = y.values - (b1 * x + b0)
        s = np.std(resid) if len(resid) > 1 else 0.1 * (abs(y.iloc[-1]) + 1)
        lower = pred - 1.96 * s
        upper = pred + 1.96 * s
        return pred, lower, upper
    except Exception:
        last = float(y.iloc[-1])
        pred = np.full(horizon, last, dtype=float)
        return pred, pred * 0.9, pred * 1.1

# ---------- data loading helpers ----------
@st.cache_data
def load_main():
    df = pd.read_csv("IHME-GBD_2023_DATA-b7fb5d99-1.csv")
    df["measure"] = df["measure"].replace({"DALYs (Disability-Adjusted Life Years)": "DALYs"})
    agg = df.groupby(["year", "location", "measure"], as_index=False)["val"].sum()
    simple = agg.pivot(index=["year", "location"], columns="measure", values="val").reset_index()
    simple = simple.rename(columns={"DALYs": "val_DALYs", "Deaths": "val_Deaths"})
    simple["year"] = pd.to_numeric(simple["year"], errors="coerce")
    if "val_DALYs" in simple.columns:
        simple["val_DALYs"] = pd.to_numeric(simple["val_DALYs"], errors="coerce")
    if "val_Deaths" in simple.columns:
        simple["val_Deaths"] = pd.to_numeric(simple["val_Deaths"], errors="coerce")
    simple = simple.dropna(subset=["year"])
    return simple

@st.cache_data
def load_age():
    df = pd.read_csv("IHME-GBD_2023_DATA-4e80f6ce-1_age_gender.csv")
    df["measure"] = df["measure"].replace({"DALYs (Disability-Adjusted Life Years)": "DALYs"})
    dalys = df[df["measure"] == "DALYs"].copy()
    dalys = dalys.groupby(["year", "location", "age"], as_index=False)[["val"]].sum().rename(columns={"val": "val_DALYs"})
    dalys["year"] = pd.to_numeric(dalys["year"], errors="coerce")
    dalys["val_DALYs"] = pd.to_numeric(dalys["val_DALYs"], errors="coerce")
    dalys = dalys.dropna(subset=["year", "val_DALYs"])
    return dalys

@st.cache_data
def load_cause():
    df = pd.read_csv("IHME-GBD_2023_DATA-b1520254-1_cause_.csv")
    df["measure"] = df["measure"].replace({"DALYs (Disability-Adjusted Life Years)": "DALYs"})
    dalys = df[df["measure"] == "DALYs"].copy()
    dalys = dalys.groupby(["year", "location", "cause"], as_index=False)[["val"]].sum().rename(columns={"val": "val_DALYs"})
    dalys["year"] = pd.to_numeric(dalys["year"], errors="coerce")
    dalys["val_DALYs"] = pd.to_numeric(dalys["val_DALYs"], errors="coerce")
    dalys = dalys.dropna(subset=["year", "val_DALYs"])
    return dalys

@st.cache_data
def load_cause_sex():
    # cause broken down by sex (for cause x gender comparison)
    df = pd.read_csv("IHME-GBD_2023_DATA-b1520254-1_cause_.csv")
    df["measure"] = df["measure"].replace({"DALYs (Disability-Adjusted Life Years)": "DALYs"})
    dalys = df[df["measure"] == "DALYs"].copy()
    # group by year, location, cause, sex
    dalys = dalys.groupby(["year", "location", "sex", "cause"], as_index=False)[["val"]].sum().rename(columns={"val": "val_DALYs"})
    dalys["year"] = pd.to_numeric(dalys["year"], errors="coerce")
    dalys["val_DALYs"] = pd.to_numeric(dalys["val_DALYs"], errors="coerce")
    dalys = dalys.dropna(subset=["year", "val_DALYs"])
    return dalys

@st.cache_data
def load_sex():
    # aggregated DALYs by year/location/sex (for Tab 1 gender lines)
    df = pd.read_csv("IHME-GBD_2023_DATA-4e80f6ce-1_age_gender.csv")
    df["measure"] = df["measure"].replace({"DALYs (Disability-Adjusted Life Years)": "DALYs"})
    dalys = df[df["measure"] == "DALYs"].copy()
    dalys = dalys.groupby(["year", "location", "sex"], as_index=False)[["val"]].sum().rename(columns={"val": "val_DALYs"})
    dalys["year"] = pd.to_numeric(dalys["year"], errors="coerce")
    dalys["val_DALYs"] = pd.to_numeric(dalys["val_DALYs"], errors="coerce")
    dalys = dalys.dropna(subset=["year", "val_DALYs"])
    return dalys

# load datasets
simple_df_total = load_main()
by_age = load_age()
by_cause = load_cause()
by_cause_sex = load_cause_sex()
by_sex = load_sex()

# Sidebar filters
st.sidebar.title("Filters")
countries = sorted(simple_df_total["location"].dropna().unique())
default_country = "Kenya" if "Kenya" in countries else (countries[0] if countries else None)
country = st.sidebar.selectbox("Country", countries, index=countries.index(default_country) if default_country else 0)

min_year = int(simple_df_total["year"].min())
max_year = int(simple_df_total["year"].max())
year_range = st.sidebar.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Use the tabs above the charts to switch views.")

# Page header
st.title("African Health Burden Dashboard (2003 - 2023)")
st.markdown(f"This dashboard explores Deaths and DALYs across African countries using IHME GBD 2023 data. Current view: `{country}`, years {year_range[0]} - {year_range[1]}.")

# Filter for country & years (for tabs 1-3)
main_sub = simple_df_total[(simple_df_total["location"] == country) & (simple_df_total["year"].between(year_range[0], year_range[1]))].sort_values("year")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Country Time Trends", "Age Patterns", "Cause Patterns", "Forecasting (RF, XGBoost, ARIMA)"])

# ----------------- TAB 1: Time trends + By gender -----------------
with tab1:
    st.subheader("1. Time trends for DALYs and Deaths")
    colA, colB = st.columns([2,1])
    with colB:
        view_mode = st.radio("View mode", ["Overall", "By gender"], index=0)
    col1, col2 = st.columns(2)
    # colorful DALYs/Deaths lines
    with col1:
        st.markdown("**DALYs over time**")
        fig_dalys = px.line(main_sub, x="year", y="val_DALYs", markers=True, title=f"DALYs trend - {country}", labels={"val_DALYs":"DALYs","year":"Year"}, color_discrete_sequence=PALETTE)
        fig_dalys.update_layout(template="plotly_white", hovermode="x unified")
        fig_dalys.update_xaxes(dtick=1)
        st.plotly_chart(fig_dalys, use_container_width=True)
    with col2:
        st.markdown("**Deaths over time**")
        fig_deaths = px.line(main_sub, x="year", y="val_Deaths", markers=True, title=f"Deaths trend - {country}", labels={"val_Deaths":"Deaths","year":"Year"}, color_discrete_sequence=PALETTE_ALT)
        fig_deaths.update_layout(template="plotly_white", hovermode="x unified")
        fig_deaths.update_xaxes(dtick=1)
        st.plotly_chart(fig_deaths, use_container_width=True)

    st.markdown("**DALYs vs Deaths**")
    if not main_sub.empty and "val_Deaths" in main_sub.columns and "val_DALYs" in main_sub.columns:
        fig_scatter = px.scatter(main_sub, x="val_Deaths", y="val_DALYs", text="year",
                                 labels={"val_Deaths":"Deaths","val_DALYs":"DALYs"},
                                 title=f"DALYs vs Deaths - {country}",
                                 color_discrete_sequence=PALETTE)
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Not enough data to show DALYs vs Deaths scatter for this selection.")

    # By gender view
    if view_mode == "By gender":
        st.markdown("**DALYs by gender**")
        sex_sub = by_sex[(by_sex["location"]==country) & (by_sex["year"].between(year_range[0], year_range[1]))].copy()
        if sex_sub.empty:
            st.info("No sex-disaggregated DALYs data available for this selection.")
        else:
            fig_sex = px.line(sex_sub, x="year", y="val_DALYs", color="sex", markers=True,
                              labels={"val_DALYs":"DALYs","year":"Year","sex":"Gender"},
                              title=f"DALYs over time by gender - {country}",
                              color_discrete_sequence=px.colors.qualitative.Bold)
            fig_sex.update_layout(template="plotly_white", hovermode="x unified")
            fig_sex.update_xaxes(dtick=1)
            st.plotly_chart(fig_sex, use_container_width=True)

# ----------------- TAB 2: Age patterns -----------------
with tab2:
    st.subheader("2. Age patterns of DALYs")
    age_sub = by_age[(by_age["location"]==country) & (by_age["year"].between(year_range[0], year_range[1]))].copy()
    if age_sub.empty:
        st.info("No age-specific data available for this selection.")
    else:
        years_available = sorted(age_sub["year"].unique())
        year_age = st.selectbox("Select year for age distribution", years_available, index=len(years_available)-1)
        age_year_sub = age_sub[age_sub["year"]==year_age].sort_values("age")
        st.markdown(f"DALYs by age group in {country}, {year_age}")
        fig_age = px.bar(age_year_sub, x="age", y="val_DALYs", labels={"age":"Age group","val_DALYs":"DALYs"},
                         title=f"DALYs by age group - {country}, {year_age}", color_discrete_sequence=PALETTE)
        fig_age.update_layout(template="plotly_white", xaxis_tickangle=45)
        st.plotly_chart(fig_age, use_container_width=True)

        st.markdown(f"Heatmap of DALYs by age and year - {country}")
        heat = age_sub.pivot_table(index="age", columns="year", values="val_DALYs", aggfunc="sum").fillna(0)
        fig_hm = px.imshow(heat, labels=dict(x="Year", y="Age group", color="DALYs"), aspect="auto", title=f"DALYs heatmap - {country}", color_continuous_scale="Turbo")
        fig_hm.update_layout(template="plotly_white")
        st.plotly_chart(fig_hm, use_container_width=True)

# ----------------- TAB 3: Cause patterns + Cause x Gender -----------------
with tab3:
    st.subheader("3. Leading causes of DALYs")
    cause_sub = by_cause[(by_cause["location"]==country) & (by_cause["year"].between(year_range[0], year_range[1]))].copy()
    if cause_sub.empty:
        st.info("No cause-specific data available for this selection.")
    else:
        years_cause = sorted(cause_sub["year"].unique())
        year_cause = st.selectbox("Select year for cause breakdown", years_cause, index=len(years_cause)-1)
        cause_year_sub = cause_sub[cause_sub["year"]==year_cause].copy()
        top_causes = cause_year_sub.sort_values("val_DALYs", ascending=False).head(8)
        st.markdown(f"Top causes of DALYs in {country}, {year_cause}")
        fig_cause = px.bar(top_causes, x="cause", y="val_DALYs", labels={"cause":"Cause","val_DALYs":"DALYs"},
                           title=f"Top causes of DALYs - {country}, {year_cause}", color="cause", color_discrete_sequence=PALETTE)
        fig_cause.update_layout(template="plotly_white", xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_cause, use_container_width=True)

        st.markdown("Time trend for a selected cause")
        cause_list = sorted(cause_sub["cause"].unique())
        chosen_cause = st.selectbox("Select cause to view trend", cause_list)
        cause_trend = cause_sub[cause_sub["cause"]==chosen_cause].sort_values("year")
        fig_ct = px.line(cause_trend, x="year", y="val_DALYs", markers=True, labels={"val_DALYs":"DALYs","year":"Year"},
                         title=f"DALYs trend for '{chosen_cause}' - {country}", color_discrete_sequence=PALETTE_ALT)
        fig_ct.update_layout(template="plotly_white", hovermode="x unified")
        fig_ct.update_xaxes(dtick=1)
        st.plotly_chart(fig_ct, use_container_width=True)

        # Cause x Gender comparison
        st.markdown("Cause × Gender comparison")
        cs_sub = by_cause_sex[(by_cause_sex["location"]==country) & (by_cause_sex["year"].between(year_range[0], year_range[1]))].copy()
        if cs_sub.empty:
            st.info("No cause-by-sex data available for this selection.")
        else:
            # user picks year and top N causes
            years_cs = sorted(cs_sub["year"].unique())
            year_cs = st.selectbox("Select year for cause x gender view", years_cs, index=len(years_cs)-1, key="cs_year")
            n_causes = st.slider("Number of top causes to show (per sex)", min_value=3, max_value=12, value=6, step=1, key="n_causes")
            cs_year_sub = cs_sub[cs_sub["year"]==year_cs].copy()
            # find top causes by combined val
            top_c = cs_year_sub.groupby("cause")["val_DALYs"].sum().sort_values(ascending=False).head(n_causes).index.tolist()
            cs_plot = cs_year_sub[cs_year_sub["cause"].isin(top_c)].copy()

            # stacked bar by sex for chosen causes
            fig_cs = px.bar(cs_plot, x="cause", y="val_DALYs", color="sex", barmode="group",
                            labels={"cause":"Cause","val_DALYs":"DALYs","sex":"Gender"},
                            title=f"DALYs by Cause and Gender - {country}, {year_cs}",
                            color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_cs.update_layout(template="plotly_white", xaxis_tickangle=45)
            st.plotly_chart(fig_cs, use_container_width=True)

            # small multiples: trend of top causes by sex
            st.markdown("Trends of selected causes by gender")
            trend_df = cs_sub[cs_sub["cause"].isin(top_c)].copy()
            fig_trends = px.line(trend_df, x="year", y="val_DALYs", color="sex", facet_col="cause", facet_col_wrap=3, markers=True,
                                 labels={"val_DALYs":"DALYs","year":"Year","sex":"Gender"},
                                 title=f"Trends for top causes by gender - {country}")
            fig_trends.update_layout(template="plotly_white", hovermode="x unified")
            fig_trends.update_xaxes(dtick=1)
            st.plotly_chart(fig_trends, use_container_width=True)

# ----------------- TAB 4: Forecasting -----------------
with tab4:
    st.subheader("4. Forecasting with Random Forest, XGBoost and ARIMA")
    full_sub = simple_df_total[simple_df_total["location"]==country].sort_values("year").copy()
    if len(full_sub) < 6 or "val_DALYs" not in full_sub.columns:
        st.info("Not enough historical years (or DALYs missing) to build forecasts. Need at least ~6 years.")
    else:
        st.markdown("We fit three models and show future-only forecasts: Random Forest, XGBoost and ARIMA. Only future forecasts are plotted (no numeric performance metrics shown).")
        horizon = st.slider("Forecast horizon (years ahead)", min_value=3, max_value=10, value=5, step=1)
        years = full_sub["year"].values.reshape(-1,1)
        dalys = full_sub["val_DALYs"].values
        last_year = int(full_sub["year"].max())
        future_years = np.arange(last_year+1, last_year+1+horizon)
        X_future = future_years.reshape(-1,1)

        rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42)
        rf.fit(years, dalys)
        preds_rf_future = rf.predict(X_future)

        xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42)
        xgb.fit(years, dalys)
        preds_xgb_future = xgb.predict(X_future)

        pred_arima, lower_arima, upper_arima = fit_arima_with_fallback(dalys, horizon=horizon)

        df_rf_future = pd.DataFrame({"year": future_years, "DALYs": preds_rf_future, "Model": "Random Forest"})
        df_xgb_future = pd.DataFrame({"year": future_years, "DALYs": preds_xgb_future, "Model": "XGBoost"})
        df_arima_future = pd.DataFrame({"year": future_years, "DALYs": pred_arima, "Model": "ARIMA(1,1,1)", "lower": lower_arima, "upper": upper_arima})

        df_forecast = pd.concat([df_rf_future, df_xgb_future, df_arima_future.drop(columns=["lower","upper"])], ignore_index=True)

        st.markdown("### Forecasted DALYs (future years only)")
        fig_forecast = px.line(df_forecast, x="year", y="DALYs", color="Model", markers=True,
                               title=f"Future DALYs forecast - {country}", labels={"year":"Year","DALYs":"Forecast DALYs"},
                               color_discrete_sequence=px.colors.qualitative.Alphabet)
        fig_forecast.update_layout(template="plotly_white", hovermode="x unified")
        fig_forecast.update_xaxes(dtick=1)

        ar = df_arima_future.sort_values("year")
        if ar["DALYs"].notna().any():
            fig_forecast.add_trace(go.Scatter(
                x=list(ar["year"]) + list(ar["year"][::-1]),
                y=list(ar["upper"]) + list(ar["lower"][::-1]),
                fill="toself",
                fillcolor="rgba(0,0,0,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name="ARIMA 95% CI"
            ))

        st.plotly_chart(fig_forecast, use_container_width=True)
        st.caption("Each line shows projected DALYs for future years. ARIMA includes a 95% CI band when available.")
