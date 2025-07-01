import streamlit as st
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Function to load and prepare main data_raw
@st.cache_data
def load_data():
    df = pd.read_excel('data_processed/df_final.xlsx')
    df['Period'] = pd.to_datetime(df['Period'])
    return df

# Function to load data_raw with predictions
@st.cache_data
def load_predictions():
    df_pred = pd.read_excel('outputs/Data_w_Predictions.xlsx')
    df_pred['period'] = pd.to_datetime(df_pred['period'], dayfirst=True)
    return df_pred

# Function to load merged dataframe with integrated variables for correlation
@st.cache_data
def load_merged_data():
    df_merged = pd.read_excel('data_processed/df_merged_year.xlsx')
    return df_merged

# Function to load corrected HVO data_raw
@st.cache_data
def load_hvo():
    df_hvo = pd.read_excel('data_raw/HVO España.xlsx')
    df_hvo.columns = df_hvo.columns.str.strip()  # Clean spaces in column names
    df_hvo['HVO'] = df_hvo['HVO'].astype(str).str.replace(',', '.').astype(float)  # Convert to float
    df_hvo['Fecha'] = pd.to_datetime(dict(year=df_hvo['Año'], month=df_hvo['Mes'], day=1))  # Create date
    return df_hvo

# Function to load best models and MAPE data_raw
@st.cache_data
def load_best_models():
    df_models = pd.read_excel('outputs/BestModels_MAPE.xlsx')
    return df_models

# Function to plot PACF and ACF with white letters
def plot_pacf_acf_white(series, lags=40):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_pacf(series, lags=lags, ax=axes[0], zero=False)
    axes[0].set_title('PACF', color='white')
    axes[0].grid(True)
    axes[0].set_facecolor('none')
    axes[0].spines['bottom'].set_color('white')
    axes[0].spines['left'].set_color('white')
    axes[0].axhline(0, linestyle='--', color='white')
    axes[0].tick_params(axis='x', colors='white')
    axes[0].tick_params(axis='y', colors='white')
    axes[0].xaxis.label.set_color('white')
    axes[0].yaxis.label.set_color('white')
    ylim = max(abs(axes[0].get_ylim()[0]), abs(axes[0].get_ylim()[1]))
    axes[0].set_ylim(-ylim, ylim)

    plot_acf(series, lags=lags, ax=axes[1], zero=False)
    axes[1].set_title('ACF', color='white')
    axes[1].grid(True)
    axes[1].set_facecolor('none')
    axes[1].spines['bottom'].set_color('white')
    axes[1].spines['left'].set_color('white')
    axes[1].axhline(0, linestyle='--', color='white')
    axes[1].tick_params(axis='x', colors='white')
    axes[1].tick_params(axis='y', colors='white')
    axes[1].xaxis.label.set_color('white')
    axes[1].yaxis.label.set_color('white')
    ylim = max(abs(axes[1].get_ylim()[0]), abs(axes[1].get_ylim()[1]))
    axes[1].set_ylim(-ylim, ylim)

    fig.patch.set_alpha(0)

    st.pyplot(fig)

df = load_data()
df_pred = load_predictions()
df_merged_year = load_merged_data()
df_hvo = load_hvo()
df_models = load_best_models()

st.title('Macro Demand for Fuels & Eco-fuels in Spain')

tab1, tab2 = st.tabs(["Data", "Forecast"])

with tab1:
    regions = sorted(df['CCAA'].unique())
    products = sorted(df['Product'].unique())

    selected_region = st.selectbox('Select Autonomous Community', regions)

    if selected_region == "España":
        if "HVO" not in products:
            products_with_hvo = products + ["HVO"]
        else:
            products_with_hvo = products
    else:
        products_with_hvo = [p for p in products if p != "HVO"]

    selected_product = st.selectbox('Select Product', sorted(products_with_hvo))

    if selected_region == "España" and selected_product == "HVO":

        st.subheader("HVO Consumption by Year")

        hvo_yearly = df_hvo.groupby('Año')['HVO'].sum().reset_index()
        fig_hvo = go.Figure()
        fig_hvo.add_trace(go.Scatter(
            x=hvo_yearly['Año'],
            y=hvo_yearly['HVO'],
            mode='lines+markers',
            name='HVO (tonnes)',
        ))
        fig_hvo.update_layout(
            title='Annual HVO Consumption in Spain',
            xaxis_title='Year',
            yaxis_title='Tonnes',
            xaxis=dict(dtick=1)
        )
        st.plotly_chart(fig_hvo, use_container_width=True)


        st.subheader("Integrated Correlation Heatmap")

        correlation_columns = [
            'Gasolina 95 I.O.', 'Gasolina 98 I.O.', 'Gasóleo A', 'Gasóleo B',
            'MUJERES', 'VARONES', 'TOTAL_POPULATION',
            'Autobuses_gasolina', 'Autobuses_gasoil',
            'Turismos_gasolina', 'Turismos_gasoil',
            'Motocicletas_gasolina', 'Motocicletas_gasoil',
            'Tractores Industriales_gasolina', 'Tractores Industriales_gasoil',
            'Otros vehículos_gasolina', 'Otros vehículos_gasoil',
            'Camiones y Furgonetas_gasolina', 'Camiones y Furgonetas_gasoil',
            'Total_gasolina', 'Total_gasoil',
            'GDP_month', 'Diesel A', 'Diesel B', 'Gasolina 95', 'Gasolina 98',
            'Matriculaciones Gasolina 95', 'Matriculaciones Gasolina 98',
            'Matriculaciones Diesel A',
        ]

        existing_cols = [col for col in correlation_columns if col in df_merged_year.columns]

        corr_data = df_merged_year[existing_cols].dropna()
        corr_matrix = corr_data.corr()

        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(18, 16))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={"shrink": 0.8},
            annot_kws={"alpha": 0.8, "fontsize": 10, "weight": "bold"},
            ax=ax,
        )
        plt.title("Integrated Correlation Matrix", fontsize=18, weight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    else:

        df_filtered = df[(df['CCAA'] == selected_region) & (df['Product'] == selected_product)].copy()
        df_filtered = df_filtered.sort_values('Period')
        df_filtered.set_index('Period', inplace=True)

        st.write(f"Data for {selected_product} in {selected_region}")
        st.dataframe(df_filtered[['Tonnes', 'Precio']])

        fig_consumption = go.Figure()
        fig_consumption.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Tonnes'], mode='lines+markers', name='Tonnes'))
        fig_consumption.update_layout(title='Consumption in Tonnes', xaxis_title='Date', yaxis_title='Tonnes')
        st.plotly_chart(fig_consumption, use_container_width=True)

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Precio'], mode='lines+markers', name='Price'))
        fig_price.update_layout(title='Price', xaxis_title='Date', yaxis_title='Euros')
        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader('PACF and ACF Plots')
        if len(df_filtered) > 30:
            plot_pacf_acf_white(df_filtered['Tonnes'], lags=30)
        else:
            plot_pacf_acf_white(df_filtered['Tonnes'], lags=len(df_filtered) - 1)

        st.subheader("Integrated Correlation Heatmap")

        correlation_columns = [
            'Gasolina 95 I.O.', 'Gasolina 98 I.O.', 'Gasóleo A', 'Gasóleo B',
            'MUJERES', 'VARONES', 'TOTAL_POPULATION',
            'Autobuses_gasolina', 'Autobuses_gasoil',
            'Turismos_gasolina', 'Turismos_gasoil',
            'Motocicletas_gasolina', 'Motocicletas_gasoil',
            'Tractores Industriales_gasolina', 'Tractores Industriales_gasoil',
            'Otros vehículos_gasolina', 'Otros vehículos_gasoil',
            'Camiones y Furgonetas_gasolina', 'Camiones y Furgonetas_gasoil',
            'Total_gasolina', 'Total_gasoil',
            'GDP_month', 'Diesel A', 'Diesel B', 'Gasolina 95', 'Gasolina 98',
            'Matriculaciones Gasolina 95', 'Matriculaciones Gasolina 98',
            'Matriculaciones Diesel A',
        ]

        existing_cols = [col for col in correlation_columns if col in df_merged_year.columns]

        corr_data = df_merged_year[existing_cols].dropna()
        corr_matrix = corr_data.corr()

        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(18, 16))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={"shrink": 0.8},
            annot_kws={"alpha": 0.8, "fontsize": 10, "weight": "bold"},
            ax=ax,
        )
        plt.title("Integrated Correlation Matrix", fontsize=18, weight='bold')
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.header("Consumption Predictions")

    regions_pred = sorted(df_pred['CCAA'].unique())
    products_all = sorted(df_pred['Product'].unique())

    selected_region_pred = st.selectbox('Select Autonomous Community for Prediction', regions_pred, key='region_pred')

    # Si la región es España, incluir HVO en productos, si no, excluirlo
    if selected_region_pred == "España":
        products_pred = products_all
    else:
        products_pred = [p for p in products_all if p != "HVO"]

    selected_product_pred = st.selectbox('Select Product for Prediction', products_pred, key='prod_pred')

    st.subheader(f"Best Model and MAPE for {selected_product_pred} in {selected_region_pred}")
    df_models_filtered = df_models[
        (df_models['CCAA'].str.contains(selected_region_pred, case=False, na=False)) &
        (df_models['Product'].str.contains(selected_product_pred, case=False, na=False))
    ]
    st.dataframe(df_models_filtered[['Product', 'MAPE', 'Model']].reset_index(drop=True))

    df_pred_filtered = df_pred[(df_pred['CCAA'] == selected_region_pred) & (df_pred['Product'] == selected_product_pred)].copy()
    df_pred_filtered = df_pred_filtered.sort_values('period')
    df_pred_filtered.set_index('period', inplace=True)

    st.write(f"Historical Data and Predictions for {selected_product_pred} in {selected_region_pred}")
    st.dataframe(df_pred_filtered[['Tonnes', 'LOWER', 'UPPER', 'Average']])

    fig_pred = go.Figure()

    df_pred_filtered.reset_index(inplace=True)

    if selected_product_pred == "HVO" and selected_region_pred == "España":

        fig_pred.add_trace(go.Scatter(
            x=df_pred_filtered['period'].loc[df_pred_filtered['Average'] == 0],
            y=df_pred_filtered['Tonnes'].loc[df_pred_filtered['Average'] == 0],
            mode='lines+markers',
            name='Tonnes',
            line=dict(color='blue')
        ))

        fig_pred.add_trace(go.Scatter(
            x=df_pred_filtered['period'].loc[df_pred_filtered['Average'] != 0],
            y=df_pred_filtered['Tonnes'].loc[df_pred_filtered['Average'] != 0],
            mode='lines+markers',
            name='Predicted Tonnes',
            line=dict(color='green')
        ))

        fig_pred.add_trace(go.Scatter(
            x=df_pred_filtered['period'].loc[df_pred_filtered['Average'] != 0],
            y=df_pred_filtered['Average'].loc[df_pred_filtered['Average'] != 0],
            mode='lines+markers',
            name='Average',
            line=dict(color='orange')
        ))

    else:
        fig_pred.add_trace(go.Scatter(
            x=df_pred_filtered['period'].loc[df_pred_filtered['Average'] == 0],
            y=df_pred_filtered['Tonnes'].loc[df_pred_filtered['Average'] == 0],
            mode='lines+markers',
            name='Actual Tonnes',
            line=dict(color='blue')
        ))

        fig_pred.add_trace(go.Scatter(
            x=df_pred_filtered['period'].loc[df_pred_filtered['Average'] != 0],
            y=df_pred_filtered['Tonnes'].loc[df_pred_filtered['Average'] != 0],
            mode='lines+markers',
            name='Predicted Tonnes',
            line=dict(color='green')
        ))

        fig_pred.add_trace(go.Scatter(
            x=df_pred_filtered['period'].loc[df_pred_filtered['Average'] != 0],
            y=df_pred_filtered['Average'].loc[df_pred_filtered['Average'] != 0],
            mode='lines+markers',
            name='Average',
            line=dict(color='orange')
        ))

        fig_pred.add_trace(go.Scatter(
            x=pd.Series(list(df_pred_filtered['period'].loc[df_pred_filtered['Average'] != 0].to_list()) + list(df_pred_filtered['period'].loc[df_pred_filtered['Average'] != 0].to_list()[::-1])),
            y=pd.concat([df_pred_filtered['UPPER'].loc[df_pred_filtered['Average'] != 0], df_pred_filtered['LOWER'].loc[df_pred_filtered['Average'] != 0][::-1]]),
            fill='toself',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,165,0,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))

    min_tonnes = df_pred_filtered['Tonnes'][df_pred_filtered['Tonnes'] > 0].min()
    min_average = df_pred_filtered['Average'][df_pred_filtered['Average'] > 0].min() if any(df_pred_filtered['Average'] > 0) else min_tonnes
    y_min = min(min_tonnes, min_average) * 0.95

    fig_pred.update_layout(
        title='Actual Consumption and Prediction with Intervals',
        xaxis_title='Date',
        yaxis_title='Tonnes',
        yaxis=dict(range=[y_min, df_pred_filtered[['Tonnes', 'UPPER']].max().max() * 1.05]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig_pred, use_container_width=True)