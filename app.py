import streamlit as st
import joblib
import numpy as np

# 1. Konfigurasi Halaman (Lebih Lebar)
st.set_page_config(page_title="Rekomendasi Tanaman", page_icon="🌾", layout="wide")

# 2. Injeksi CSS Khusus
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        padding-bottom: 20px;
    }
    .result-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header Web dengan CSS kustom
st.markdown('<div class="main-header">🌾 Sistem Rekomendasi Tanaman Cerdas</div>', unsafe_allow_html=True)

# 3. Menggunakan Tabs untuk Navigasi Halaman
tab1, tab2 = st.tabs(["📊 Panel Prediksi", "📖 Informasi Sistem"])

with tab1:
    model = joblib.load('model_tanaman.pkl')
    
    st.write("Sesuaikan *slider* di bawah ini dengan kondisi lahan aktual untuk mendapatkan rekomendasi komoditas terbaik.")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    # Input Menggunakan Slider agar lebih interaktif
    with col1:
        st.subheader("🧪 Hara Makro")
        n = st.slider("Nitrogen (N)", 0, 140, 50)
        p = st.slider("Fosfor (P)", 0, 145, 50)
        k = st.slider("Kalium (K)", 0, 205, 50)

    with col2:
        st.subheader("🌡️ Cuaca & Iklim")
        temp = st.slider("Suhu Udara (°C)", 10.0, 50.0, 25.0, step=0.5)
        humidity = st.slider("Kelembapan (%)", 10.0, 100.0, 70.0, step=0.5)
        rainfall = st.slider("Curah Hujan (mm)", 10.0, 300.0, 100.0)

    with col3:
        st.subheader("💧 Kondisi Tanah")
        ph = st.slider("pH Tanah", 3.0, 9.0, 6.5, step=0.1)
        
        # Expander untuk bantuan
        with st.expander("ℹ️ Panduan pH Tanah"):
            st.write("Sebagian besar tanaman ideal tumbuh pada pH netral (6.0 - 7.0). pH di bawah 6 berarti asam, di atas 7 berarti basa.")

    st.markdown("---")
    
    # Tombol Analisis
    if st.button("🚀 Proses Analisis Lahan", use_container_width=True):
        input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
        
        prediksi = model.predict(input_data)[0]
        probabilitas = np.max(model.predict_proba(input_data)) * 100
        
        # 4. Tampilan Hasil Kustom dengan HTML dan Komponen Metrik
        st.markdown(f"""
        <div class="result-box">
            <h3 style='margin:0; color:#1b5e20;'>Rekomendasi Utama: {prediksi.upper()}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Menampilkan probabilitas/tingkat keyakinan model
        st.metric(label="Tingkat Keyakinan AI", value=f"{probabilitas:.2f}%")
        st.balloons() # Efek animasi balon saat berhasil

with tab2:
    st.header("Tentang Sistem")
    st.write("Sistem ini dibangun menggunakan algoritma **Random Forest Classifier** yang dilatih dengan ribuan data historis pertanian.")
    st.write("Faktor penentu utama mencakup rasio NPK, suhu lingkungan, persentase kelembapan udara, tingkat keasaman (pH) tanah, serta intensitas curah hujan.")