import streamlit as st
import numpy as np
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
default_geo = '/mnt/data/PETA KELURAHAN KOTA TANGERANG.geojson'
default_xls = '/mnt/data/DBD PER KLURAHAN 2023,2024.xlsx'
if geo or True:
    gdf = gpd.read_file(geo if geo else default_geo).to_crs(epsg=4326)
    gdf['KELURAHAN']=gdf['NAME_4'].astype(str).str.upper().str.strip()
    if csv:
        df = pd.read_excel(csv) if str(csv.name).endswith('xlsx') else pd.read_csv(csv)
    else:
        raw = pd.read_excel(default_xls, header=None)
        raw.columns = raw.iloc[0]
        df = raw[1:].copy()
        if 'kecamatan' in df.columns and kecamatan:
            df = df[df['kecamatan'].astype(str).str.contains(kecamatan, case=False, na=False)]
        if 'tahun' in df.columns and tahun != 'Semua':
            df = df[df['tahun']==tahun]
        df = df.rename(columns={df.columns[1]:'KELURAHAN'})
    df['KELURAHAN']=df['KELURAHAN'].astype(str).str.upper().str.strip()
    for c in df.columns:
        if str(c).isdigit(): df[c]=pd.to_numeric(df[c], errors='coerce').fillna(0)
    df = gdf.merge(df, on='KELURAHAN', how='left')
    center=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m=folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap')
    value_col = None
    year_cols = [c for c in gdf.columns if str(c).isdigit()]
    if year_cols:
        value_col = sorted(year_cols)[-1]
        folium.Choropleth(geo_data=gdf.to_json(), data=gdf, columns=['KELURAHAN', value_col], key_on='feature.properties.KELURAHAN', fill_color='YlOrRd', legend_name=f'Kasus {value_col}').add_to(m)
    folium.GeoJson(gdf, name='Kelurahan', tooltip=folium.GeoJsonTooltip(fields=['KELURAHAN'] + year_cols)).add_to(m)
    if not df.empty and {'lat','lon'}.issubset(df.columns):
        pts=df[['lat','lon']].dropna().values.tolist()
        HeatMap(pts, radius=15).add_to(m)
        for _,r in df.iterrows():
            folium.CircleMarker([r['lat'],r['lon']], radius=5, popup=str(r.to_dict())).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width=1200, height=650)
    if not df.empty:
        st.subheader('Visualisasi Data Asli')
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if year_cols:
            yearly = pd.DataFrame({'tahun':year_cols,'kasus':[gdf[c].fillna(0).sum() for c in year_cols]})
            st.plotly_chart(px.line(yearly, x='tahun', y='kasus', markers=True, title='Tren Kasus per Tahun'), use_container_width=True)
        if 'tahun' in df.columns:
            yearly = df.groupby('tahun').size().reset_index(name='kasus')
            st.plotly_chart(px.bar(yearly, x='tahun', y='kasus', title='Kasus per Tahun'), use_container_width=True)
        if len(num_cols) >= 2:
            target = num_cols[-1]
            feats = num_cols[:-1]
            X = df[feats].fillna(0)
            y = df[target].fillna(0)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X,y)
            pred = model.predict(X)
            df['prediksi'] = pred
            st.subheader('Prediksi Random Forest')
            st.dataframe(df[[target,'prediksi']].head(20), use_container_width=True)
            imp = pd.DataFrame({'Fitur':feats,'Importance':model.feature_importances_})
            st.plotly_chart(px.bar(imp, x='Fitur', y='Importance', title='Feature Importance'), use_container_width=True)
    st.subheader('Ringkasan Dashboard')
    c1,c2=st.columns(2)
    c1.metric('Jumlah Wilayah', len(gdf))
    c2.metric('Jumlah Kasus', len(df) if not df.empty else 0)
    if not df.empty and 'prediksi' in df.columns:
        st.metric('Rata-rata Prediksi', round(df['prediksi'].mean(),2))
    st.markdown('---')
    st.subheader('Kesimpulan Otomatis')
    if not df.empty:
        st.success('Dashboard berhasil menampilkan sebaran kasus per kelurahan Kota Tangerang berdasarkan data asli, tren tahunan, serta analisis prediktif untuk mendukung keputusan pemerintah daerah.')
    st.download_button('Download Data CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='hasil_dashboard.csv', mime='text/csv')
else:
    st.info('Upload GeoJSON untuk memulai.')
