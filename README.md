# IMDB Movie Review Sentiment & Regression Dashboard

Aplikasi dashboard berbasis Streamlit untuk analisis sentimen (klasifikasi) dan prediksi skor (regresi) pada review film IMDB.

## Fitur

- Prediksi sentimen (positif/negatif) menggunakan Decision Tree dan Neural Network
- Prediksi skor review (regresi) menggunakan Neural Network
- Batch prediction untuk file CSV
- Visualisasi hasil prediksi

## Cara Menjalankan

1. Pastikan Python 3.8+ sudah terinstall.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run dashboard_imdb.py
   ```

## Struktur File Penting

- `dashboard_imdb.py` : File utama dashboard
- `Model/` : Folder model terlatih
- `tfidf_vectorizer_decisiontree.joblib` : Vectorizer untuk Decision Tree
- `tokenizer_sentiment_klasifikasi.pkl` : Tokenizer untuk model klasifikasi
- `tokenizer_sentiment_regresi.pkl` : Tokenizer untuk model regresi

## Catatan

- Pastikan file model dan tokenizer sudah tersedia sesuai struktur di atas.
- Untuk batch prediction, file CSV harus memiliki kolom `review`.

## Contoh Penggunaan

Lihat pada aplikasi setelah dijalankan.

RUN APP:

C:\Users\ilham\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run dashboard_imdb.py
