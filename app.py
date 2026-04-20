import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title='Dashboard DBD Tangerang',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Header
st.title('🦟 Dashboard DBD Tangerang')
st.caption('Analisis spasial, visualisasi, prediksi, dan dashboard siap sidang')

with st.container():
    st.markdown('### Dashboard Monitoring Demam Berdarah Dengue Kota Tangerang')

# Sidebar controls
st.sidebar.header('⚙️ Kontrol Dashboard')
geo = st.sidebar.file_uploader(
    'Upload GeoJSON Kelurahan (opsional)',
    type=['geojson', 'json']
)
csv = st.sidebar.file_uploader(
    'Upload CSV/Excel Kasus (opsional)',
    type=['csv', 'xlsx']
)

kecamatan = st.sidebar.text_input('Filter Kecamatan (opsional)')
tahun = st.sidebar.selectbox('Pilih Tahun', ['Semua', 2022, 2023, 2024, 2025, 2026])

# Default file paths
DEFAULT_GEO = 'map.geojson'
DEFAULT_XLS = 'data.xlsx'


@st.cache_data
def load_geo(src) -> gpd.GeoDataFrame:
    """Load dan proses GeoJSON file."""
    try:
        if src is None:
            if not Path(DEFAULT_GEO).exists():
                st.error(f"File {DEFAULT_GEO} tidak ditemukan!")
                return gpd.GeoDataFrame()
            gdf = gpd.read_file(DEFAULT_GEO)
        elif hasattr(src, 'read'):
            # File upload dari Streamlit
            gdf = gpd.read_file(src)
        else:
            gdf = gpd.read_file(src)
        
        gdf = gdf.to_crs(epsg=4326)
        
        # Cari kolom nama yang sesuai
        name_candidates = ['NAME_4', 'NAME_3', 'KELURAHAN', 'kelurahan', 'NAMA', 'nama']
        name_col = None
        for col in name_candidates:
            if col in gdf.columns:
                name_col = col
                break
        
        if name_col is None:
            name_col = gdf.columns[0]
        
        gdf['KELURAHAN'] = gdf[name_col].astype(str).str.upper().str.strip()
        return gdf
    
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
        return gpd.GeoDataFrame()


@st.cache_data
def load_data(src) -> pd.DataFrame:
    """Load dan proses data kasus dari CSV/Excel."""
    try:
        if src is not None:
            # File dari upload
            if hasattr(src, 'name'):
                if src.name.endswith('.xlsx'):
                    df = pd.read_excel(src)
                else:
                    df = pd.read_csv(src)
            else:
                df = pd.read_csv(src)
        else:
            # File default
            if not Path(DEFAULT_XLS).exists():
                st.warning(f"File {DEFAULT_XLS} tidak ditemukan. Menggunakan data dummy.")
                return create_dummy_data()
            
            raw = pd.read_excel(DEFAULT_XLS, header=None)
            if raw.empty:
                return create_dummy_data()
            raw.columns = raw.iloc[0]
            df = raw[1:].copy().reset_index(drop=True)
        
        # Normalisasi kolom KELURAHAN
        kelurahan_candidates = ['KELURAHAN', 'kelurahan', 'Kelurahan', 'NAMA', 'nama']
        for col in kelurahan_candidates:
            if col in df.columns:
                df = df.rename(columns={col: 'KELURAHAN'})
                break
        
        # Jika tidak ada kolom KELURAHAN, gunakan kolom kedua
        if 'KELURAHAN' not in df.columns and len(df.columns) > 1:
            df = df.rename(columns={df.columns[1]: 'KELURAHAN'})
        
        if 'KELURAHAN' in df.columns:
            df['KELURAHAN'] = df['KELURAHAN'].astype(str).str.upper().str.strip()
        
        # Konversi kolom tahun ke numerik
        for col in df.columns:
            col_str = str(col)
            if col_str.isdigit():
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_dummy_data()


def create_dummy_data() -> pd.DataFrame:
    """Buat data dummy jika file tidak tersedia."""
    kelurahan_list = [
        'KARAWACI', 'CIMONE', 'CIPONDOH', 'TANAH TINGGI', 'SUKASARI',
        'SUKARASA', 'CIBODAS', 'CILEDUG', 'LARANGAN', 'KARANG TENGAH'
    ]
    
    np.random.seed(42)
    data = {
        'KELURAHAN': kelurahan_list,
        '2022': np.random.randint(10, 100, len(kelurahan_list)),
        '2023': np.random.randint(15, 120, len(kelurahan_list)),
        '2024': np.random.randint(20, 150, len(kelurahan_list)),
    }
    return pd.DataFrame(data)


def create_map(df: gpd.GeoDataFrame, year_cols: list) -> folium.Map:
    """Buat peta Folium dengan choropleth."""
    if df.empty or df.geometry.is_empty.all():
        # Return peta default Tangerang jika tidak ada data
        return folium.Map(location=[-6.2, 106.63], zoom_start=11, tiles='OpenStreetMap')
    
    # Hitung center
    centroids = df.geometry.centroid
    center_y = centroids.y.mean()
    center_x = centroids.x.mean()
    
    if pd.isna(center_y) or pd.isna(center_x):
        center = [-6.2, 106.63]  # Default Tangerang
    else:
        center = [center_y, center_x]
    
    m = folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap')
    
    if year_cols:
        value_col = sorted(year_cols)[-1]
        
        # Pastikan data valid untuk choropleth
        df_valid = df[df[value_col].notna()].copy()
        
        if not df_valid.empty:
            folium.Choropleth(
                geo_data=df_valid.to_json(),
                data=df_valid,
                columns=['KELURAHAN', value_col],
                key_on='feature.properties.KELURAHAN',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.5,
                legend_name=f'Kasus DBD {value_col}',
                nan_fill_color='white'
            ).add_to(m)
    
    # Tambahkan tooltip
    tooltip_fields = ['KELURAHAN'] + year_cols if year_cols else ['KELURAHAN']
    available_fields = [f for f in tooltip_fields if f in df.columns]
    
    if available_fields:
        folium.GeoJson(
            df,
            tooltip=folium.GeoJsonTooltip(
                fields=available_fields,
                aliases=[f.replace('_', ' ').title() for f in available_fields],
                style="font-size: 12px;"
            ),
            style_function=lambda x: {
                'fillOpacity': 0,
                'color': 'blue',
                'weight': 1
            }
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m


def run_prediction(df: pd.DataFrame, year_cols: list) -> pd.DataFrame:
    """Jalankan prediksi dengan Random Forest."""
    if len(year_cols) < 2:
        return df
    
    target = sorted(year_cols)[-1]
    features = sorted(year_cols)[:-1]
    
    X = df[features].fillna(0).values
    y = df[target].fillna(0).values
    
    if len(X) < 2:
        return df
    
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    df = df.copy()
    df['prediksi'] = model.predict(X).round(0).astype(int)
    df['error'] = abs(df[target] - df['prediksi'])
    
    return df


# Main execution
def main():
    # Load data
    with st.spinner('Memuat data...'):
        gdf = load_geo(geo if geo else None)
        df_raw = load_data(csv if csv else None)
    
    # Filter data
    if 'kecamatan' in df_raw.columns and kecamatan:
        df_raw = df_raw[
            df_raw['kecamatan'].astype(str).str.contains(kecamatan, case=False, na=False)
        ]
    
    if 'tahun' in df_raw.columns and tahun != 'Semua':
        df_raw = df_raw[df_raw['tahun'] == tahun]
    
    # Merge data
    if not gdf.empty and not df_raw.empty:
        df = gdf.merge(df_raw, on='KELURAHAN', how='left')
    elif not gdf.empty:
        df = gdf.copy()
    else:
        st.error("Tidak dapat memuat data geografis!")
        return
    
    # Identifikasi kolom tahun
    year_cols = [c for c in df.columns if str(c).isdigit()]
    
    # Tampilkan peta
    st.subheader('🗺️ Peta Sebaran Kasus DBD')
    m = create_map(df, year_cols)
    st_folium(m, width=1200, height=550, returned_objects=[])
    
    # Visualisasi data
    if not df.empty and year_cols:
        st.markdown('---')
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('📈 Tren Kasus Tahunan')
            yearly_data = pd.DataFrame({
                'Tahun': year_cols,
                'Total Kasus': [df[c].fillna(0).sum() for c in year_cols]
            })
            st.line_chart(
                yearly_data.set_index('Tahun')['Total Kasus'],
                use_container_width=True
            )
        
        with col2:
            st.subheader('📊 Distribusi per Wilayah')
            if len(year_cols) > 0:
                latest_year = sorted(year_cols)[-1]
                top_10 = df.nlargest(10, latest_year)[['KELURAHAN', latest_year]]
                st.bar_chart(
                    top_10.set_index('KELURAHAN')[latest_year],
                    use_container_width=True
                )
        
        # Prediksi
        if len(year_cols) > 1:
            st.markdown('---')
            st.subheader('🤖 Analisis Prediksi Random Forest')
            
            df = run_prediction(df, year_cols)
            
            if 'prediksi' in df.columns:
                target = sorted(year_cols)[-1]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    display_cols = ['KELURAHAN', target, 'prediksi', 'error']
                    available_cols = [c for c in display_cols if c in df.columns]
                    st.dataframe(
                        df[available_cols].sort_values('prediksi', ascending=False).head(15),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    mae = df['error'].mean() if 'error' in df.columns else 0
                    accuracy = 100 - (mae / df[target].mean() * 100) if df[target].mean() > 0 else 0
                    
                    st.metric('Mean Absolute Error', f'{mae:.2f}')
                    st.metric('Akurasi Estimasi', f'{accuracy:.1f}%')
    
    # Ringkasan
    st.markdown('---')
    st.subheader('📋 Ringkasan Dashboard')
    
    c1, c2, c3, c4 = st.columns(4)
    
    total_wilayah = len(df) if not df.empty else 0
    total_kasus = int(df[year_cols].sum().sum()) if year_cols else 0
    rata_rata = round(total_kasus / total_wilayah, 2) if total_wilayah > 0 else 0
    max_kasus = int(df[year_cols].max().max()) if year_cols else 0
    
    c1.metric('🏘️ Jumlah Wilayah', total_wilayah)
    c2.metric('🦠 Total Kasus', f'{total_kasus:,}')
    c3.metric('📊 Rata-rata Kasus', rata_rata)
    c4.metric('⚠️ Kasus Tertinggi', max_kasus)
    
    # Kesimpulan
    st.markdown('---')
    st.subheader('💡 Kesimpulan Otomatis')
    
    if year_cols and len(year_cols) >= 2:
        trend = df[year_cols[-1]].sum() - df[year_cols[-2]].sum()
        trend_text = "meningkat" if trend > 0 else "menurun" if trend < 0 else "stabil"
        
        st.success(f'''
        Dashboard menampilkan sebaran kasus DBD per kelurahan di Kota Tangerang.
        - Total {total_wilayah} wilayah kelurahan dipantau
        - Tren kasus {trend_text} sebesar {abs(int(trend))} kasus dari tahun sebelumnya
        - Analisis prediksi berbasis Random Forest telah dijalankan
        ''')
    else:
        st.success('Dashboard menampilkan sebaran kasus DBD per kelurahan Kota Tangerang.')
    
    # Download
    st.markdown('---')
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df.drop(columns=['geometry'], errors='ignore').to_csv(index=False).encode('utf-8')
        st.download_button(
            '📥 Download Data CSV',
            data=csv_data,
            file_name='hasil_dashboard_dbd.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col2:
        if st.button('🔄 Refresh Data', use_container_width=True):
            st.cache_data.clear()
            st.rerun()


if __name__ == '__main__':
    main()
