# 1. Gunakan basis image Python 3.9 yang ringan
FROM python:3.9-slim

# 2. Atur folder kerja di dalam container
WORKDIR /app

# 3. Salin file daftar dependensi terlebih dahulu
COPY requirements.txt requirements.txt

# 4. Install semua dependensi yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# 5. Salin semua file proyek (main.py, model .h5, dan semua .pkl)
COPY . .

# 6. Jalankan server aplikasi menggunakan Gunicorn saat container dimulai
#    -b 0.0.0.0:80 : Jalankan di semua interface network pada port 80
#    -w 4 : Gunakan 4 worker process untuk menangani request
#    -k uvicorn.workers.UvicornWorker : Gunakan worker Uvicorn yang cocok untuk FastAPI
#    main:app : Jalankan objek 'app' dari file 'main.py'
CMD ["gunicorn", "-b", "0.0.0.0:80", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app"]