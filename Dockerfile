# 1. Gunakan basis image Python 3.9 yang ringan dan stabil
FROM python:3.9-slim

# 2. Atur folder kerja di dalam container
WORKDIR /app

# 3. Salin file daftar dependensi terlebih dahulu untuk optimasi cache
COPY requirements.txt requirements.txt

# 4. Install semua dependensi yang dibutuhkan dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Salin semua sisa file proyek Anda (main.py, model .h5, dan semua .pkl)
COPY . .

# 6. Perintah untuk menjalankan aplikasi Flask Anda dengan Gunicorn
#    Ini akan mencari objek bernama 'app' di dalam file 'main.py'
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "4", "main:app"]
