import streamlit as st
import joblib
import numpy as np
import random

# Load model
model = joblib.load('delivery_time_model.pkl')

# Fungsi prediksi
def predict_delivery_time(features):
  prediction = model.predict([features])
  return prediction[0]

# ---- UI ----
st.set_page_config(page_title="Food Delivery Prediction", page_icon='random', layout="centered")

st.image("assets/food-delivery-service2.jpg", use_container_width=True)
st.title("🚀 Prediksi Waktu Pengiriman Makanan")
st.markdown("Masukkan detail pesanan untuk memperkirakan waktu pengiriman.")

st.markdown("---")

# Fungsi untuk mengisi input dengan nilai acak
def generate_random_inputs():
  return {
    "distance": round(random.uniform(0.1, 20.0), 1),
    "prep_time": random.randint(5, 30),
    "courier_exp": random.randint(0, 10),
    "traffic_level": random.choice(["Low", "Medium", "High"]),
    "time_of_day": random.choice(["Afternoon", "Evening", "Morning", "Night"]),
    "vehicle_type": random.choice(["Bike", "Car", "Scooter"]),
    "weather": random.choice(["Clear", "Foggy", "Rainy", "Snowy", "Windy"]),
  }

if "random_inputs" not in st.session_state:
  st.session_state.random_inputs = generate_random_inputs()

if st.button("🎲 Gunakan Nilai Acak"):
  st.session_state.random_inputs = generate_random_inputs()

random_inputs = st.session_state.random_inputs

with st.container():
  st.subheader("📌 Informasi Pengiriman")

  distance = st.number_input("📍 Jarak ke Pelanggan (km)", min_value=0.1, max_value=50.0, step=0.1, value=random_inputs["distance"])
  prep_time = st.number_input("⏳ Waktu Persiapan (menit)", min_value=1, max_value=120, step=1, value=random_inputs["prep_time"])
  courier_exp = st.number_input("🚴‍♂️ Pengalaman Kurir (tahun)", min_value=0, max_value=20, step=1, value=random_inputs["courier_exp"])

  traffic_level = st.selectbox("🚦 Tingkat Kemacetan", ["Low", "Medium", "High"], index=["Low", "Medium", "High"].index(random_inputs["traffic_level"]))
  time_of_day = st.selectbox("⏰ Waktu Pemesanan", ["Afternoon", "Evening", "Morning", "Night"], index=["Afternoon", "Evening", "Morning", "Night"].index(random_inputs["time_of_day"]))
  vehicle_type = st.selectbox("🚗 Jenis Kendaraan Kurir", ["Bike", "Car", "Scooter"], index=["Bike", "Car", "Scooter"].index(random_inputs["vehicle_type"]))
  weather = st.selectbox("🌦️ Kondisi Cuaca", ["Clear", "Foggy", "Rainy", "Snowy", "Windy"], index=["Clear", "Foggy", "Rainy", "Snowy", "Windy"].index(random_inputs["weather"]))

# Encoding fitur kategorikal
traffic_mapping = {"Low": 0, "Medium": 1, "High": 2}
traffic_encoded = traffic_mapping[traffic_level]

time_mapping = {"Evening": [1, 0, 0], "Morning": [0, 1, 0], "Night": [0, 0, 1], "Afternoon": [0, 0, 0]}
time_encoded = time_mapping[time_of_day]

vehicle_mapping = {"Car": [1, 0], "Scooter": [0, 1], "Bike": [0, 0]}
vehicle_encoded = vehicle_mapping[vehicle_type]

weather_mapping = {"Foggy": [1, 0, 0, 0], "Rainy": [0, 1, 0, 0], "Snowy": [0, 0, 1, 0], "Windy": [0, 0, 0, 1], "Clear": [0, 0, 0, 0]}
weather_encoded = weather_mapping[weather]

# Gunakan fitur numerik tanpa scaling
features = [distance, prep_time, courier_exp] + [traffic_encoded] + time_encoded + vehicle_encoded + weather_encoded

st.markdown("---")

if st.button("🚀 Prediksi Waktu Pengiriman"):
  result = predict_delivery_time(features)
  st.success(f"📦 Perkiraan waktu pengiriman: **{result:.2f} menit**")

st.markdown("---")
st.markdown("🔍 *Aplikasi ini dikembangkan untuk membantu restoran & kurir memperkirakan waktu pengiriman dengan lebih akurat.*")