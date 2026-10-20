# Studi Kasus: Fusi Satelit (Sentinel-1 & Sentinel-2) dan Atribusi AI Penyebab Deforestasi Tropis

Berdasarkan kuliah praktisi **"Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI"** (19 Oktober 2026) oleh **Bart Slagter** (Researcher, Laboratory of Geo-Information Science & Remote Sensing, **Wageningen University & Research / WUR**).

Pesan utama materi: pemantauan hutan tropis modern menggabungkan citra radar **Sentinel-1** (mampu menembus awan) dan citra optik **Sentinel-2** dengan model *Convolutional Neural Network* (CNN) pada resolusi 10 meter untuk mengatribusikan **penyebab langsung (*direct drivers*)** kehilangan tutupan hutan secara otomatis guna mendukung transparansi publik (*Global Forest Watch / Global Nature Watch*) dan penegakan kebijakan yang tepat sasaran.

## Hubungan dengan Materi

| Poin Materi Kuliah Praktisi | Implementasi pada Proyek Ini |
| :--- | :--- |
| **Fusi Multi-Sensor Sentinel-1 (Radar) & Sentinel-2 (Optik)** | Menyesuaikan bobot fusi radar SAR dan optik secara dinamis berdasarkan persentase tutupan awan aktual dari API Open-Meteo di 9 koridor hutan Sumatera (termasuk Bengkulu) dan Kalimantan |
| **Klasifikasi *Direct Drivers* dengan CNN (Resolusi 10m)** | Mengekstraksi fitur spasial (ukuran bukaan lahan ha, tekstur batas konvolusi 2D, dan kedekatan sungai) untuk membedakan *Pertanian Skala Besar*, *Pertanian Skala Kecil*, *Penebangan (Selective Logging)*, *Pertambangan*, serta *Kebakaran Hutan & Gangguan Alami* |
| **Pola Regional Asia Tenggara & *Driver Co-occurrence*** | Memetakan rantai keterkaitan antarpenyebab (*co-occurrence*), seperti pembangunan jalan hutan yang diikuti penebangan selektif lalu kebakaran, atau pertambangan yang diikuti pertanian skala kecil |
| **Evaluasi Keterbatasan Model (Kebakaran vs Banjir)** | Menyertakan deteksi **Ambiguitas Dataran Banjir (*Floodplain Confusion*)** untuk menandai peringatan di tepi sungai yang berisiko tertukar antara bekas kebakaran dan gangguan banjir alami |
| **Otomasi Cloud & Transparansi Publik** | GitHub Actions memperbarui hasil klasifikasi CSV dan peta interaktif (`docs/index.html`) secara terjadwal setiap 6 jam |

## Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
python pipeline_deforestasi_ai.py
