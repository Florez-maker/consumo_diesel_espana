import pandas as pd

# 1. Cargar consumo
df_consumption = pd.read_excel('data_raw/consumos-pp-ccaa-provincias.xlsx', sheet_name='Consumos', skiprows=5)
df_consumption.columns = df_consumption.columns.str.strip()
month_map = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
    'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}
df_consumption['Mes'] = df_consumption['Mes'].str.lower().map(month_map)

# 2. Cargar matriculaciones y renombrar columnas
df_matriculaciones = pd.read_excel('data_raw/matriculaciones.xlsx')
df_matriculaciones.rename(columns={'Comunidad': 'CCAA'}, inplace=True)
df_matriculaciones.rename(columns={
    'Gasolina 95': 'Matriculaciones Gasolina 95',
    'Gasolina 98': 'Matriculaciones Gasolina 98',
    'Diesel A': 'Matriculaciones Diesel A',
    'Diesel B': 'Matriculaciones Diesel B'
}, inplace=True)

# Normalizar claves para merge
for df_ in [df_consumption, df_matriculaciones]:
    df_['CCAA'] = df_['CCAA'].str.strip().str.lower()
    df_['Año'] = df_['Año'].astype(int)
    df_['Mes'] = df_['Mes'].astype(int)

# Filtrar df_consumption para que solo tenga CCAA y años que existen en matriculaciones
ccaa_validos = df_matriculaciones['CCAA'].unique()
anos_validos = df_matriculaciones['Año'].unique()
df_consumption_filtered = df_consumption[
    (df_consumption['CCAA'].isin(ccaa_validos)) &
    (df_consumption['Año'].isin(anos_validos))
]

print(f"Filas en df_consumption antes del filtro: {len(df_consumption)}")
print(f"Filas en df_consumption después del filtro: {len(df_consumption_filtered)}")

# 3. Cargar PIB trimestral y convertir a mensual
df_gdp = pd.read_excel('data_raw/GDP_REGIONS_Quaterly.xlsx', sheet_name='PIB trim CCAA', skiprows=1)
df_gdp.rename(columns={'Unnamed: 0': 'period'}, inplace=True)
df_gdp['period'] = df_gdp['period'].astype(str)
df_gdp = pd.melt(df_gdp, id_vars='period', var_name='region', value_name='GDP_Q')
df_gdp['YEAR'] = df_gdp['period'].str[:4].astype(int)
df_gdp['quarter'] = df_gdp['period'].str[4:].astype(int)
df_gdp['CCAA'] = df_gdp['region'].str.strip().str.lower()

quarter_to_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}
df_gdp_monthly = df_gdp.loc[df_gdp.index.repeat(3)].copy()
df_gdp_monthly['month_idx'] = df_gdp_monthly.groupby(['YEAR', 'CCAA', 'quarter']).cumcount()
df_gdp_monthly['Mes'] = df_gdp_monthly.apply(lambda r: quarter_to_months[r['quarter']][r['month_idx']], axis=1)
df_gdp_monthly['GDP_month'] = df_gdp_monthly['GDP_Q'] / 3
df_gdp_monthly = df_gdp_monthly[['YEAR', 'Mes', 'CCAA', 'GDP_month']]
df_gdp_monthly.rename(columns={'YEAR': 'Año'}, inplace=True)
df_gdp_monthly['CCAA'] = df_gdp_monthly['CCAA'].str.strip().str.lower()

# 4. Cargar precios y pivotear
def prep_price_df(file, product_name):
    df_price = pd.read_excel(file, index_col=0).fillna(0)
    df_price.rename(columns={
        'Toda España': 'España',
        'Andalucia': 'Andalucía',
        'Aragon': 'Aragón',
        'Comunidad Valencia': 'Comunidad Valenciana',
        'Castilla la Mancha': 'Castilla-La Mancha',
        'Rioja (La)': 'La Rioja'
    }, inplace=True)
    df_long = df_price.reset_index().melt(id_vars='Fecha', var_name='CCAA', value_name='Precio')
    df_long['Period'] = pd.to_datetime(df_long['Fecha'], format='%Y-%m')
    df_long['Año'] = df_long['Period'].dt.year
    df_long['Mes'] = df_long['Period'].dt.month
    df_long['CCAA'] = df_long['CCAA'].str.strip().str.lower()
    df_long['Product'] = product_name
    return df_long[['Año', 'Mes', 'CCAA', 'Precio', 'Product']]

prices_diesel_A = prep_price_df('../data_raw/Prices_Gasoleo_A.xlsx', 'Diesel A')
prices_diesel_B = prep_price_df('../data_raw/Prices_Gasoleo_B.xlsx', 'Diesel B')
prices_gasoline_95 = prep_price_df('../data_raw/Prices_Gasoline_95.xlsx', 'Gasolina 95')
prices_gasoline_98 = prep_price_df('../data_raw/Prices_Gasoline_98.xlsx', 'Gasolina 98')

df_prices = pd.concat([prices_diesel_A, prices_diesel_B, prices_gasoline_95, prices_gasoline_98], ignore_index=True)
df_prices_pivot = df_prices.pivot_table(index=['Año', 'Mes', 'CCAA'], columns='Product', values='Precio').reset_index()

# 5. Merge datasets
df_merged = df_consumption_filtered.merge(df_matriculaciones, how='left', on=['CCAA', 'Año', 'Mes'])
df_merged = df_merged.merge(df_gdp_monthly, how='left', on=['CCAA', 'Año', 'Mes'])
df_merged = df_merged.merge(df_prices_pivot, how='left', on=['CCAA', 'Año', 'Mes'])

# Verificar columnas matriculaciones
print(df_merged[['Matriculaciones Gasolina 95', 'Matriculaciones Diesel A']].info())
print(df_merged[['Matriculaciones Gasolina 95', 'Matriculaciones Diesel A']].head(10))

# Guardar resultado
df_merged.to_excel('data_processed/df_merged_year.xlsx', index=False)
print("Archivo df_merged_year.xlsx creado con éxito.")