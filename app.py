import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Membaca dataset dari CSV
df = pd.read_csv('data_park.csv')

# Mengubah format datetime pada kolom date_time
df['date_time'] = pd.to_datetime(df['date_time'])

# Mendapatkan fitur-fitur yang diinginkan
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['day_of_week'] = df['date_time'].dt.dayofweek
df['week_of_year'] = df['date_time'].dt.isocalendar().week
df['hour'] = df['date_time'].dt.hour
df['minute'] = df['date_time'].dt.minute
df['hour_min'] = df['hour'] + (df['minute'] / 60)

# Memilih fitur yang akan digunakan
features = ['year', 'month', 'day', 'day_of_week', 'week_of_year', 'hour', 'minute', 'hour_min']
target = 'parking_zone'

# Memisahkan fitur dan target
X = df[features]
y = df[target]

# Membuat model regresi logistik
model = LogisticRegression()
model.fit(X, y)

# Fungsi untuk memprediksi zona parkir berdasarkan input tanggal dan jam
def predict_parking_zone(date, hour, minute):
    # Membuat fitur-fitur dari input pengguna
    input_features = pd.DataFrame({
        'year': [date.year],
        'month': [date.month],
        'day': [date.day],
        'day_of_week': [date.weekday()],
        'week_of_year': [date.isocalendar()[1]],
        'hour': [hour],
        'minute': [minute],
        'hour_min': [hour + (minute / 60)]
    })

    # Melakukan prediksi zona parkir
    prediction = model.predict(input_features)
    return prediction[0]

# Tampilan antarmuka pengguna dengan Streamlit
st.title('Prediksi Zona Parkir')
st.write('Masukkan tanggal dan jam untuk memprediksi zona parkir.')

# Input tanggal
date = st.date_input('Tanggal')

# Input jam
hour = st.number_input('Jam', min_value=0, max_value=23, step=1, value=0)
minute = st.number_input('Menit', min_value=0, max_value=59, step=1, value=0)

# Button untuk memprediksi zona parkir
if st.button('Prediksi'):
    # Memanggil fungsi prediksi
    prediction = predict_parking_zone(date, hour, minute)
    st.success('Prediksi Zona Parkir: {}'.format(prediction))
