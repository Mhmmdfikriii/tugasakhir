import streamlit as st
import numpy as np
import plotly
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title='Dashboard DBD Tangerang', layout='wide')
st.title('Dashboard DBD Tangerang - Versi Final Skripsi')
st.caption('Analisis spasial, visualisasi, prediksi, dan dashboard siap sidang')
with st.container():
    st.markdown('### Dashboard Monitoring Demam Berdarah Dengue Kota Tangerang')

st.sidebar.header('Kontrol Dashboard')
geo = st.sidebar.file_uploader('Upload GeoJSON Kelurahan (opsional)', type=['geojson','json'])
csv = st.sidebar.file_uploader('Upload CSV Kasus (opsional)', type=['csv','xlsx'])

kecamatan = st.sidebar.text_input('Filter Kecamatan (opsional)')
tahun = st.sidebar.selectbox('Pilih Tahun', ['Semua',2022,2023,2024,2025,2026])
default_geo = 'map.geojson'
default_xls = 'data.xlsx'
@st.cache_data
def load_geo(src):
    g = gpd.read_file(src).to_crs(epsg=4326)
    name_col = 'NAME_4' if 'NAME_4' in g.columns else g.columns[0]
    g['KELURAHAN'] = g[name_col].astype(str).str.upper().str.strip()
    return g

@st.cache_data
def load_data(src):
    if src:
        d = pd.read_excel(src) if str(src.name).endswith('xlsx') else pd.read_csv(src)
    else:
        raw = pd.read_excel(default_xls, header=None)
        raw.columns = raw.iloc[0]
        d = raw[1:].copy()
    d = d.rename(columns={d.columns[1]: 'KELURAHAN'})
    d['KELURAHAN'] = d['KELURAHAN'].astype(str).str.upper().str.strip()
    for c in d.columns:
        if str(c).isdigit():
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0)
    return d

gdf = load_geo(geo if geo else default_geo)
df_raw = load_data(csv)
if 'kecamatan' in df_raw.columns and kecamatan:
    df_raw = df_raw[df_raw['kecamatan'].astype(str).str.contains(kecamatan, case=False, na=False)]
if 'tahun' in df_raw.columns and tahun != 'Semua':
    df_raw = df_raw[df_raw['tahun'] == tahun]

df = gdf.merge(df_raw, on='KELURAHAN', how='left')
year_cols = [c for c in df.columns if str(c).isdigit()]

center = [df.geometry.centroid.y.mean(), df.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap')
if year_cols:
    value_col = sorted(year_cols)[-1]
    folium.Choropleth(geo_data=df.to_json(), data=df, columns=['KELURAHAN', value_col], key_on='feature.properties.KELURAHAN', fill_color='YlOrRd', legend_name=f'Kasus {value_col}').add_to(m)
folium.GeoJson(df, tooltip=folium.GeoJsonTooltip(fields=['KELURAHAN'] + year_cols)).add_to(m)
folium.LayerControl().add_to(m)
st_folium(m, width=1200, height=650)

if not df.empty:
    st.subheader('Visualisasi Data Asli')
    if year_cols:
        yearly = pd.DataFrame({'tahun': year_cols, 'kasus': [df[c].fillna(0).sum() for c in year_cols]})
        st.plotly_chart(px.line(yearly, x='tahun', y='kasus', markers=True, title='Tren Kasus per Tahun'), use_container_width=True)
        target = sorted(year_cols)[-1]
        if len(year_cols) > 1:
            X = df[year_cols[:-1]].fillna(0)
            y = df[target].fillna(0)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            df['prediksi'] = model.predict(X)
            st.subheader('Prediksi Random Forest')
            st.dataframe(df[['KELURAHAN', target, 'prediksi']].head(20), use_container_width=True)

st.subheader('Ringkasan Dashboard')
c1, c2, c3 = st.columns(3)
c1.metric('Jumlah Wilayah', len(df))
total_kasus = int(df[year_cols].sum().sum()) if year_cols else 0
c2.metric('Total Kasus', total_kasus)
c3.metric('Rata-rata Kasus', round(total_kasus / len(df), 2) if len(df) else 0)
st.markdown('---')
st.subheader('Kesimpulan Otomatis')
st.success('Dashboard menampilkan sebaran kasus DBD per kelurahan, tren tahunan, dan analisis prediksi berbasis data asli Kota Tangerang.')
st.download_button('Download Data CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='hasil_dashboard.csv', mime='text/csv')
