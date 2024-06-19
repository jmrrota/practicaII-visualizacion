
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import statsmodels.api as sm
import numpy as np

# Función para convertir ISO2 a ISO3
file_path = 'shopping_trends.csv'
data = pd.read_csv(file_path)

# Configurar el estado de sesión
if 'page' not in st.session_state:
    st.session_state.page = 0

# Convertir las columnas necesarias a numéricas
data['Previous Purchases'] = pd.to_numeric(data['Previous Purchases'])
data['Review Rating'] = pd.to_numeric(data['Review Rating'])

# Crear la variable 'Índice de Lealtad del Cliente'
max_frequency = data['Frequency of Purchases'].map({
    'Weekly': 4, 'Fortnightly': 2, 'Monthly': 1, 'Annually': 0.0833
}).max()
data['Frequency Value'] = data['Frequency of Purchases'].map({
    'Weekly': 4, 'Fortnightly': 2, 'Monthly': 1, 'Annually': 0.0833
})
max_previous_purchases = data['Previous Purchases'].max()

data['Índice de Lealtad'] = (data['Frequency Value'] / max_frequency) * (data['Previous Purchases'] / max_previous_purchases)

# Crear la variable 'Puntuación de Satisfacción Ajustada'
data['Puntuación de Satisfacción Ajustada'] = data['Review Rating'] * (1 + (data['Previous Purchases'] / 10))

# Crear la variable 'Segmentación de Clientes Basada en el Gasto'
purchase_amount_percentiles = data['Purchase Amount (USD)'].quantile([0.25, 0.75])
low_threshold = purchase_amount_percentiles[0.25]
high_threshold = purchase_amount_percentiles[0.75]

def categorize_purchase_amount(amount):
    if amount < low_threshold:
        return 'Bajo'
    elif amount > high_threshold:
        return 'Alto'
    else:
        return 'Medio'

data['Segmentación de Clientes'] = data['Purchase Amount (USD)'].apply(categorize_purchase_amount)

# Crear la variable 'Factor de Enganche con Promociones'
data['Promoción Aplicada'] = data[['Discount Applied', 'Promo Code Used']].apply(
    lambda x: 1 if 'Yes' in x.values else 0, axis=1)
factor_enganche = (data['Promoción Aplicada'].sum() / len(data)) * 100

# Crear la variable 'Diversidad de Productos'
diversidad_productos = data.groupby('Customer ID')['Category'].nunique().reset_index()
diversidad_productos.columns = ['Customer ID', 'Diversidad de Productos']
data = pd.merge(data, diversidad_productos, on='Customer ID')


st.set_page_config(layout="wide")

# Definir las funciones para los gráficos
def ventana1():
    st.header("Distribución de la Edad por Género")

    col1, col2, col3 = st.columns([1, 1, 1])


    # 1. Distribución de la edad por género
    fig_age_gender = px.histogram(data, x='Age', color='Gender', nbins=20, title='Distribución de la Edad por Género')
    fig_age_gender.update_layout(bargap=0.2)
    #st.plotly_chart(fig_age_gender)

    with col1:
      st.plotly_chart(fig_age_gender)


    # 2. Preferencias de compra por género
    purchase_preferences = data.groupby(['Gender', 'Category']).size().reset_index(name='Count')
    fig_purchase_preferences = px.bar(purchase_preferences, x='Category', y='Count', color='Gender', barmode='group', title='Preferencias de Compra por Género')
    #st.plotly_chart(fig_purchase_preferences)

    with col2:
      st.plotly_chart(fig_purchase_preferences)

    # 3. Métodos de pago preferidos por edad
    age_bins = pd.cut(data['Age'], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66+'])
    data['Age Group'] = age_bins
    fig_payment_methods = px.box(data, x='Preferred Payment Method', y='Age', color='Preferred Payment Method', title='Métodos de Pago Preferidos por Edad')
    #st.plotly_chart(fig_payment_methods)

    with col3:
      st.plotly_chart(fig_payment_methods)

def ventana2():
    st.header("Correlación entre Promociones y Lealtad del Cliente")

    # Calcular el uso de promociones por cliente
    promociones_por_cliente = data.groupby('Customer ID')['Promoción Aplicada'].sum().reset_index(name='Promociones Usadas')
    lealtad_por_cliente = data[['Customer ID', 'Índice de Lealtad']].drop_duplicates()

    # Unir las tablas
    analisis_promociones = pd.merge(promociones_por_cliente, lealtad_por_cliente, on='Customer ID')

    col1, col2 = st.columns([1, 1])
    # 1. Gráfico de dispersión
    fig_scatter = px.scatter(analisis_promociones, x='Promociones Usadas', y='Índice de Lealtad', title='Relación entre Promociones Usadas e Índice de Lealtad')
    #st.plotly_chart(fig_scatter)
    with col1:
      st.plotly_chart(fig_scatter)

    # 2. Gráfico de cajas
    analisis_promociones['Grupo de Promociones'] = pd.cut(analisis_promociones['Promociones Usadas'], bins=[-1, 0, 5, 10, 20, 50], labels=['0', '1-5', '6-10', '11-20', '21+'])
    fig_box = px.box(analisis_promociones, x='Grupo de Promociones', y='Índice de Lealtad', title='Índice de Lealtad por Uso de Promociones')
    #st.plotly_chart(fig_box)
    with col2:
      st.plotly_chart(fig_box)


def ventana3():
    st.header("Influencias de las Estaciones en el Comportamiento de Compra")

    col1, col2 = st.columns([1, 1])
    # 1. Patrones de gasto por estación
    fig_box_season = px.box(data, x='Season', y='Purchase Amount (USD)', title='Distribución de los Montos de Compra por Estación')
    #st.plotly_chart(fig_box_season)
    with col1:
      st.plotly_chart(fig_box_season)

    # 2. Tipos de productos comprados por estación
    productos_por_estacion = data.groupby(['Season', 'Category']).size().reset_index(name='Count')
    fig_bar_season = px.bar(productos_por_estacion, x='Season', y='Count', color='Category', barmode='group', title='Categorías de Productos Comprados por Estación')
    #st.plotly_chart(fig_bar_season)
    with col2:
      st.plotly_chart(fig_bar_season)

def ventana4():
    st.header("Relación entre la Satisfacción del Cliente y las Compras Futuras")
    # Calcular el total de compras futuras por cliente
    data['Future Purchases'] = data.groupby('Customer ID')['Previous Purchases'].transform('sum')

    col1, col2 = st.columns([1, 1])
    # 1. Gráfico de correlación interactivo
    correlation_matrix = data[['Review Rating', 'Future Purchases']].corr()
    fig_corr = go.Figure(data=go.Heatmap(
                       z=correlation_matrix.values,
                       x=correlation_matrix.columns,
                       y=correlation_matrix.columns,
                       colorscale='Blues'))
    fig_corr.update_layout(title='Matriz de Correlación: Satisfacción vs Compras Futuras')
    #st.plotly_chart(fig_corr)
    with col1:
      st.plotly_chart(fig_corr)

    # 2. Regresión lineal mejorada
    X = data['Review Rating']
    y = data['Future Purchases']
    X = sm.add_constant(X)  # Agregar una constante para el término independiente
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    confidence_interval = model.get_prediction(X).conf_int()

    # Crear la figura de regresión lineal
    fig_regression = go.Figure()

    # Añadir puntos de datos
    fig_regression.add_trace(go.Scatter(
        x=data['Review Rating'], y=data['Future Purchases'],
        mode='markers', name='Datos',
        marker=dict(color='blue', opacity=0.6)
    ))

    # Añadir línea de regresión
    fig_regression.add_trace(go.Scatter(
        x=data['Review Rating'], y=predictions,
        mode='lines', name='Regresión Lineal',
        line=dict(color='red')
    ))

    # Añadir intervalos de confianza
    fig_regression.add_trace(go.Scatter(
        x=data['Review Rating'], y=confidence_interval[:, 0],
        mode='lines', name='Confianza Inferior',
        line=dict(color='lightgrey'), fill=None
    ))
    fig_regression.add_trace(go.Scatter(
        x=data['Review Rating'], y=confidence_interval[:, 1],
        mode='lines', name='Confianza Superior',
        line=dict(color='lightgrey'), fill='tonexty'
    ))

    fig_regression.update_layout(
        title='Regresión Lineal Mejorada: Calificación de los Productos vs Compras Futuras',
        xaxis_title='Calificación de los Productos',
        yaxis_title='Compras Futuras'
    )
    #st.plotly_chart(fig_regression)
    with col2:
      st.plotly_chart(fig_regression)


# Lista de gráficos
visualizations = [
    ventana1,
    ventana2,
    ventana3,
    ventana4
]

# Mostrar gráfico actual
visualizations[st.session_state.page]()

# Botones de navegación
col1, col2, col3 = st.columns([1, 1, 1])

if col1.button('Previous'):
    if st.session_state.page > 0:
        st.session_state.page -= 1

if col3.button('Next'):
    if st.session_state.page < len(visualizations) - 1:
        st.session_state.page += 1

st.write(f"Visualization {st.session_state.page + 1} of {len(visualizations)}")
