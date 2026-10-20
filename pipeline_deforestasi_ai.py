"""
Studi Kasus: Pipeline Fusi Satelit (Sentinel-1/2) & Atribusi AI Penyebab Deforestasi Tropis
Berdasarkan kuliah praktisi "Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI"
oleh Bart Slagter (Laboratory of Geo-Information Science & Remote Sensing, Wageningen University & Research).

Alur (sesuai pesan materi: fusi radar-optik, klasifikasi CNN resolusi 10m, co-occurrence, & evaluasi limitasi):
  1. Ambil kondisi awan, hujan, & kekeringan koridor hutan dari API -> Konteks tropis (tutupan awan & musim)
  2. Fusi fitur Sentinel-1 SAR (tembus awan) + Sentinel-2 Optik     -> Komplementaritas sensor radar & optik
  3. Ekstraksi fitur konvolusi 2D (CNN) & ukuran bukaan lahan (ha)  -> Klasifikasi 5 Direct Drivers (10m)
  4. Identifikasi Driver Co-occurrence & Ambiguitas Kebakaran/Banjir-> Pola sekuensial & keterbatasan model
  5. Publikasi peta ala Global Nature Watch (docs/index.html)       -> Transparansi pengawasan & otomasi cloud
Jika API tidak dapat diakses, dipakai DATA CONTOH sintetis (ditandai jelas di dashboard).
"""
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# 9 Koridor Pantau Hutan Tropis Asia Tenggara / Indonesia (Sumatera, Bengkulu, & Kalimantan)
KAWASAN = {
    "TN Kerinci Seblat (Bengkulu)": (-3.15, 101.85, "Sumatera (Bengkulu)"),
    "Bukit Barisan Selatan (Bengkulu–Lampung)": (-4.65, 103.90, "Sumatera (Bengkulu)"),
    "Ekosistem Leuser (Aceh–Sumut)": (3.78, 97.25, "Sumatera"),
    "Tesso Nilo & Kampar (Riau)": (-0.18, 101.62, "Sumatera"),
    "Berbak & Sembilang (Jambi–Sumsel)": (-1.72, 104.15, "Sumatera"),
    "Kapuas Hulu (Kalimantan Barat)": (0.82, 112.65, "Kalimantan"),
    "Sebangau & Katingan (Kalimantan Tengah)": (-2.45, 113.55, "Kalimantan"),
    "Mahakam Ulu (Kalimantan Timur)": (0.55, 115.40, "Kalimantan"),
    "Meratus & Sungai Barito (Kalimantan Selatan)": (-2.85, 115.25, "Kalimantan"),
}
WIB = timezone(timedelta(hours=7))
DOCS = Path("docs")


def konvolusi_2d(patch, kernel):
    """Operasi konvolusi spasial 2D (3x3) dengan NumPy untuk mengekstrak tekstur tepi/bentuk bukaan lahan."""
    h, w = patch.shape
    out = np.zeros((h - 2, w - 2), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out += patch[i:h - 2 + i, j:w - 2 + j] * kernel[i, j]
    return float(np.mean(np.abs(out)))


def ambil_api():
    """Ambil kondisi tutupan awan, curah hujan, suhu, dan kelembapan koridor hutan dari Open-Meteo API."""
    lats = ",".join(str(v[0]) for v in KAWASAN.values())
    lons = ",".join(str(v[1]) for v in KAWASAN.values())
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        "&daily=cloud_cover_mean,precipitation_sum,temperature_2m_max,shortwave_radiation_sum"
        "&past_days=14&forecast_days=1&timezone=GMT"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "studi-kasus-deforestasi-wur-unib"})
    with urllib.request.urlopen(req, timeout=25) as r:
        payload = json.load(r)

    items = payload if isinstance(payload, list) else [payload]
    rows = []
    nama_kawasan = list(KAWASAN.keys())
    for idx, item in enumerate(items):
        if idx >= len(nama_kawasan):
            break
        kawasan = nama_kawasan[idx]
        lat, lon, pulau = KAWASAN[kawasan]
        d = item.get("daily", {})
        times = d.get("time", [])
        clouds = d.get("cloud_cover_mean", [])
        rains = d.get("precipitation_sum", [])
        temps = d.get("temperature_2m_max", [])
        rads = d.get("shortwave_radiation_sum", [])
        for t_s, cl, ra, te, rd in zip(times, clouds, rains, temps, rads):
            if cl is None and te is None:
                continue
            rows.append({
                "tanggal": t_s,
                "kawasan": kawasan,
                "pulau": pulau,
                "lat": lat,
                "lon": lon,
                "awan_pct": float(cl if cl is not None else 68.0),
                "hujan_mm": float(ra if ra is not None else 8.0),
                "suhu_maks": float(te if te is not None else 31.5),
                "radiasi": float(rd if rd is not None else 18.0),
            })
    if not rows:
        raise ValueError("API tidak mengembalikan data harian koridor hutan")
    return pd.DataFrame(rows)


def data_contoh(hari=15):
    """Data sintetis kondisi atmosfer & penginderaan jauh koridor hutan (cadangan uji offline)."""
    rng = np.random.default_rng(42)
    today = datetime.now(timezone.utc).date()
    rows = []
    for d in range(hari):
        tgl = (today - timedelta(days=d)).strftime("%Y-%m-%d")
        for kawasan, (lat, lon, pulau) in KAWASAN.items():
            awan = np.clip(rng.normal(66.0, 18.0), 15.0, 98.0)
            hujan = max(0.0, round(rng.exponential(9.5) - 2.0, 1)) if awan > 50 else 0.0
            suhu = round(31.8 - 0.04 * hujan + rng.normal(0, 1.1), 1)
            rad = round(21.0 - 0.09 * awan + rng.normal(0, 1.5), 1)
            rows.append({
                "tanggal": tgl, "kawasan": kawasan, "pulau": pulau, "lat": lat, "lon": lon,
                "awan_pct": round(float(awan), 0), "hujan_mm": float(hujan),
                "suhu_maks": float(suhu), "radiasi": float(rad),
            })
    return pd.DataFrame(rows)


def klasifikasi_cnn_alerts(df):
    """
    Simulasi Fusi Multi-Sensor (Sentinel-1 SAR + Sentinel-2 Optik) & Klasifikasi CNN Resolusi 10m:
      - Menentukan bobot sensor: ketika awan > 60%, Sentinel-1 (radar) menjadi tumpuan utama
      - Mengekstraksi fitur spasial (ukuran bukaan lahan ha, tekstur tepi CNN, bentuk koridor, jarak sungai)
      - Mengatribusikan 5 Direct Drivers sesuai taksonomi Bart Slagter (WUR)
      - Melacak Driver Co-occurrence & menandai potensi kebingungan klasifikasi Kebakaran vs Banjir
    """
    rng = np.random.default_rng(2026)
    df = df.sort_values("tanggal", ascending=False).reset_index(drop=True)

    # Kernel konvolusi 3x3 pendeteksi batas geometris lurus vs tidak beraturan
    k_edge = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float64)
    k_spot = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)

    drivers, luas_ha, sensor_utama, conf_scores = [], [], [], []
    co_occur, ambiguitas_list, lat_j, lon_j = [], [], [], []

    for idx, row in df.iterrows():
        awan = row["awan_pct"]
        hujan = row["hujan_mm"]
        suhu = row["suhu_maks"]
        kawasan = row["kawasan"]

        # Jitter koordinat agar setiap peringatan (alert) merepresentasikan poligon patch 10m di sekitar kawasan
        lat_j.append(round(row["lat"] + float(rng.normal(0, 0.16)), 3))
        lon_j.append(round(row["lon"] + float(rng.normal(0, 0.16)), 3))

        # 1. Fusi Sensor Sentinel-1 (Radar SAR) & Sentinel-2 (Optik)
        bobot_s1 = round(min(0.92, max(0.25, awan / 100.0)), 2)
        if awan >= 65:
            sensor = f"Dominan Sentinel-1 SAR ({int(bobot_s1*100)}% radar, awan {int(awan)}%)"
        else:
            sensor = f"Fusi Sentinel-2 Optik + S1 ({int((1-bobot_s1)*100)}% optik)"
        sensor_utama.append(sensor)

        # 2. Bangkitkan patch fitur multi-spektral 8x8 (resolusi 10m per piksel) & jalankan konvolusi 2D
        patch = rng.normal(0.5, 0.2, (8, 8))
        if "Kalimantan" in row["pulau"] and idx % 3 == 0:
            patch[3:5, :] += 0.9  # Pola koridor jalan tebang / kanal
        elif "Riau" in kawasan or "Jambi" in kawasan:
            patch[1:7, 1:7] += 0.8  # Blok geometris besar

        skor_linear = konvolusi_2d(patch, k_edge)
        skor_bercak = konvolusi_2d(patch, k_spot)

        # 3. Ukuran bukaan lahan (hektare) & konteks sungai/termal
        dekat_sungai = (hujan >= 12.0) or ("Sungai" in kawasan) or ("Kapuas" in kawasan) or ("Mahakam" in kawasan)
        anomali_termal = (suhu >= 32.2 and hujan < 3.0)
        ukuran = round(float(rng.lognormal(2.2, 0.85)), 1)

        # 4. Atribusi Penyebab Langsung (Direct Driver Classification)
        if anomali_termal and idx % 4 == 0:
            drv = "Kebakaran Hutan & Gangguan Alami"
            ukuran = round(ukuran * 1.6, 1)
            co = "Penebangan Selektif → Kekeringan → Kebakaran"
        elif dekat_sungai and ("Kalimantan" in row["pulau"] or "Kerinci" in kawasan) and idx % 4 == 1:
            drv = "Pertambangan"
            co = "Pembukaan Akses Sungai → Pertambangan → Pertanian Kecil"
        elif skor_linear > 0.62 and ("Kalimantan" in row["pulau"] or "Leuser" in kawasan or "Kerinci" in kawasan):
            drv = "Penebangan (Selective Logging)"
            co = "Pembangunan Jalan Hutan → Penebangan Selektif"
        elif ukuran >= 14.0 or "Riau" in kawasan or "Jambi" in kawasan:
            drv = "Pertanian Skala Besar"
            ukuran = max(15.5, round(ukuran * 1.8, 1))
            co = "Pembersihan Lahan → Monokultur Skala Besar"
        else:
            drv = "Pertanian Skala Kecil"
            ukuran = min(12.0, max(1.2, ukuran))
            co = "Bukaan Ladang Tepi Hutan → Pertanian Skala Kecil"

        # 5. Evaluasi Keterbatasan Model: Ambiguitas Kebakaran vs Banjir di Dataran Sungai
        if dekat_sungai and (drv == "Kebakaran Hutan & Gangguan Alami" or hujan >= 16.0):
            ambig = "Waspada Ambiguitas (Kebakaran vs Banjir Sungai)"
            conf = round(float(rng.uniform(64.0, 76.0)), 1)
        else:
            ambig = "Klasifikasi Yakin (Tinggi)"
            conf = round(float(rng.uniform(82.0, 96.5)), 1)

        drivers.append(drv)
        luas_ha.append(ukuran)
        conf_scores.append(conf)
        co_occur.append(co)
        ambiguitas_list.append(ambig)

    df["lat"] = lat_j
    df["lon"] = lon_j
    df["driver"] = drivers
    df["luas_ha"] = luas_ha
    df["sensor_fusi"] = sensor_utama
    df["confidence"] = conf_scores
    df["co_occurrence"] = co_occur
    df["status_model"] = ambiguitas_list

    ringkasan = {
        "total_alerts": len(df),
        "total_luas_ha": round(float(df["luas_ha"].sum()), 1),
        "rata_awan": round(float(df["awan_pct"].mean()), 1),
        "rata_conf": round(float(df["confidence"].mean()), 1),
        "jml_ambiguitas": int((df["status_model"] != "Klasifikasi Yakin (Tinggi)").sum()),
    }
    kolom = [
        "tanggal", "kawasan", "pulau", "lat", "lon", "driver", "luas_ha",
        "awan_pct", "hujan_mm", "sensor_fusi", "confidence", "co_occurrence", "status_model",
    ]
    return df[kolom], ringkasan


def main():
    try:
        raw, sumber = ambil_api(), "Open-Meteo + Simulasi Fusi Sentinel-1/2"
    except Exception as e:
        print(f"[!] API gagal ({e}); memakai DATA CONTOH sintetis.")
        raw, sumber = data_contoh(), "DATA CONTOH (sintetis)"

    df, ringkasan = klasifikasi_cnn_alerts(raw)
    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").touch()
    df.to_csv(DOCS / "deforestasi_olahan.csv", index=False)

    meta = {
        "sumber": sumber,
        "diperbarui": datetime.now(WIB).strftime("%d %b %Y %H:%M WIB"),
        "ringkasan": ringkasan,
    }
    html = (
        TEMPLATE.replace("__DATA__", df.to_json(orient="records"))
        .replace("__META__", json.dumps(meta))
    )
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print(
        f"Selesai: {len(df)} peringatan deforestasi ({sumber}) | "
        f"Total {ringkasan['total_luas_ha']} ha -> docs/index.html"
    )


TEMPLATE = r'''<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Atribusi AI Penyebab Deforestasi (Sentinel-1/2 & CNN)</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
:root{--ink:#111;--bg:#fffbee;--y:#ffd23f;--p:#ff9ecd;--g:#7ee8a2;--b:#7cc4ff;--r:#ff6b6b;--o:#ffa94d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Space Grotesk","Helvetica Neue",Arial,sans-serif;line-height:1.45}
.wrap{max-width:1280px;margin:auto;padding:28px 22px 40px}
h1{margin:0;font-size:clamp(1.8rem,4.8vw,3.1rem);font-weight:800;letter-spacing:-.03em;line-height:1.06}
h1 mark{background:var(--g);padding:0 .15em;border:3px solid var(--ink);box-shadow:5px 5px 0 var(--ink);display:inline-block;transform:rotate(-1deg)}
.sub{margin:14px 0 20px;font-weight:600;max-width:880px}
.bar{display:flex;flex-wrap:wrap;gap:9px;align-items:center;background:#fff;border:3px solid var(--ink);box-shadow:8px 8px 0 var(--ink);padding:12px 14px;margin-bottom:24px}
.bar b{font-size:.78rem;text-transform:uppercase}.bar button{font:inherit;font-weight:800;border:3px solid var(--ink);background:#fff;padding:7px 12px;cursor:pointer;box-shadow:3px 3px 0 var(--ink);transition:.1s;font-size:.81rem}
.bar button:hover{transform:translate(-2px,-2px)}.bar button.on{background:var(--y);transform:translate(2px,2px);box-shadow:1px 1px 0 var(--ink)}
.tag{margin-left:auto;border:3px solid var(--ink);padding:5px 12px;font-weight:700;font-size:.78rem;background:var(--g)}.tag.warn{background:var(--r)}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:22px}.s3{grid-column:span 3}.s4{grid-column:span 4}.s8{grid-column:span 8}.s6{grid-column:span 6}.s12{grid-column:span 12}
.card{background:#fff;border:3px solid var(--ink);box-shadow:7px 7px 0 var(--ink);padding:16px 18px}.card h2{margin:0 0 10px;font-size:1rem;text-transform:uppercase}
.kpi{border:3px solid var(--ink);box-shadow:6px 6px 0 var(--ink);padding:14px 16px}.kpi span{font-size:.74rem;font-weight:700;text-transform:uppercase;display:block}.kpi b{font-size:1.9rem;display:block;margin-top:2px}.kpi small{font-weight:700;font-size:.77rem}
#map{height:520px;border:3px solid var(--ink)}.list{max-height:520px;overflow-y:auto;border:3px solid var(--ink)}
.row{display:flex;gap:10px;align-items:center;padding:8px 10px;border-bottom:2px solid var(--ink);font-size:.81rem}.row:last-child{border:0}
.m{min-width:58px;text-align:center;font-weight:800;border:2px solid var(--ink);padding:4px 4px}.row small{color:#333;display:block;margin-top:2px}
.br{display:flex;align-items:center;gap:8px;margin:7px 0;font-size:.83rem;font-weight:600}.br i{display:block;height:20px;border:2px solid var(--ink);min-width:3px}.br span{min-width:190px}
.bkl{font-size:22px;line-height:22px;text-shadow:-1px -1px #fff,1px 1px #fff,1px -1px #fff,-1px 1px #fff;text-align:center}
.leg{display:flex;flex-wrap:wrap;gap:8px 14px;margin-bottom:12px;font-size:.79rem;font-weight:700;align-items:center}
.leg i{display:inline-block;width:15px;height:15px;border:2px solid var(--ink);border-radius:50%;vertical-align:-3px;margin-right:5px}
.how{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}.how div{border:3px solid var(--ink);padding:10px 12px;font-size:.87rem;font-weight:500}
.how b{display:block;text-transform:uppercase;font-size:.78rem;margin-bottom:3px}.note{margin:14px 0 0;border:3px solid var(--ink);background:var(--y);padding:8px 12px;font-weight:700;font-size:.84rem}
ol{margin:0;padding-left:20px;font-weight:500;font-size:.9rem}ol li{margin-bottom:5px}code{background:var(--y);padding:0 4px;border:1px solid var(--ink)}
@media(max-width:860px){.s3{grid-column:span 6}.s4,.s8,.s6{grid-column:span 12}#map,.list{height:380px}}
</style></head><body><div class="wrap">
<h1>Atribusi AI <mark>Penyebab Deforestasi</mark></h1>
<p class="sub">Fusi citra radar <b>Sentinel-1</b> (tembus awan) &amp; optik <b>Sentinel-2</b> (resolusi 10m) dengan <i>Convolutional Neural Network</i> (CNN) untuk mengklasifikasikan penyebab langsung (<i>direct drivers</i>) pembukaan hutan di Sumatera (Bengkulu) &amp; Kalimantan. Studi kasus kuliah praktisi <b>Bart Slagter</b> (Wageningen University &amp; Research).</p>

<div class="bar"><b>Filter Direct Driver (CNN)</b>
<button data-d="ALL" class="on">Semua Penyebab</button>
<button data-d="Pertanian Skala Besar">Pertanian Besar</button>
<button data-d="Pertanian Skala Kecil">Pertanian Kecil</button>
<button data-d="Penebangan (Selective Logging)">Logging</button>
<button data-d="Pertambangan">Pertambangan</button>
<button data-d="Kebakaran Hutan & Gangguan Alami">Kebakaran / Alami</button>
<button data-d="AMBIGU">⚠️ Ambiguitas Banjir/Api</button>
<span class="tag" id="tag"></span></div>

<div class="grid">
<div class="kpi s3" style="background:var(--g)"><span>Peringatan Terklasifikasi</span><b id="k1"></b><small id="k1s"></small></div>
<div class="kpi s3" style="background:var(--y)"><span>Estimasi Luas Bukaan</span><b id="k2"></b><small id="k2s"></small></div>
<div class="kpi s3" style="background:var(--b)"><span>Tutupan Awan &amp; Peran Radar</span><b id="k3"></b><small id="k3s"></small></div>
<div class="kpi s3" style="background:var(--r)"><span>Evaluasi Limitasi Model</span><b id="k4"></b><small id="k4s"></small></div>

<div class="card s8"><h2>Peta Peringatan Deforestasi &amp; Atribusi Penyebab Langsung (Skala 10m)</h2>
<div class="leg">
<span><i style="background:#ffd23f"></i>Pertanian Besar</span>
<span><i style="background:#7ee8a2"></i>Pertanian Kecil</span>
<span><i style="background:#7cc4ff"></i>Penebangan (Logging)</span>
<span><i style="background:#ff9ecd"></i>Pertambangan</span>
<span><i style="background:#ff6b6b"></i>Kebakaran / Alami</span>
<span>★ Koridor Bengkulu (Kerinci Seblat)</span>
</div>
<div id="map"></div></div>

<div class="card s4"><h2>Log Peringatan &amp; Co-occurrence Terbaru</h2><div class="list" id="list"></div></div>

<div class="card s6"><h2>Proporsi Penyebab Langsung (Jumlah Peringatan)</h2><div id="b1"></div></div>
<div class="card s6"><h2>Total Luas Bukaan Lahan per Kawasan Hutan (ha)</h2><div id="b2"></div></div>

<div class="card s12"><h2>Keterkaitan dengan Materi Kuliah Praktisi (Bart Slagter, Wageningen University &amp; Research)</h2><div class="how">
<div style="background:var(--b)"><b>1 · Fusi Sentinel-1 &amp; Sentinel-2</b>Citra optik Sentinel-2 mendeteksi perubahan tajuk tetapi terhalang awan tropis; radar Sentinel-1 menembus awan setiap 5–12 hari sehingga keduanya digabungkan pada resolusi 10m.</div>
<div style="background:var(--g)"><b>2 · Klasifikasi CNN &amp; Konteks Spasial</b>CNN membedakan pertanian skala besar vs kecil, <i>selective logging</i>, pertambangan, dan kebakaran berdasarkan ukuran bukaan, bentuk geometris, serta konteks spasial sekitar.</div>
<div style="background:var(--p)"><b>3 · Driver Co-occurrence</b>Melacak urutan keterkaitan antarpenyebab di lapangan, seperti pembangunan jalan hutan yang diikuti penebangan selektif lalu kebakaran atau ekspansi pertanian di sekitar area tambang.</div>
<div style="background:var(--y)"><b>4 · Limitasi Kebakaran vs Banjir</b>Mendeteksi area dataran banjir/tepi sungai yang rawan salah klasifikasi antara bekas kebakaran hutan dan gangguan alami banjir, sebagaimana dipublikasikan di <i>Global Nature Watch</i>.</div></div>
<p class="note" id="bNote"></p></div>

<div class="card s12"><h2>Alur Otomatis Pipeline Pemantauan Deforestasi</h2><ol>
<li><code>pipeline_deforestasi_ai.py</code> mengambil parameter tutupan awan, curah hujan, dan radiasi harian pada 9 kawasan hutan Sumatera–Kalimantan dari API Open-Meteo.</li>
<li>Fusi bobot Sentinel-1 (radar) &amp; Sentinel-2 (optik) serta filter konvolusi 2D (CNN) mengatribusikan penyebab langsung, rantai <i>co-occurrence</i>, dan peringatan ambiguitas dataran banjir.</li>
<li>GitHub Actions memperbarui data CSV dan peta pengawasan publik setiap 6 jam secara otomatis.</li>
</ol></div>
</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
const D=__DATA__,M=__META__;
const COL={
 'Pertanian Skala Besar':'#ffd23f',
 'Pertanian Skala Kecil':'#7ee8a2',
 'Penebangan (Selective Logging)':'#7cc4ff',
 'Pertambangan':'#ff9ecd',
 'Kebakaran Hutan & Gangguan Alami':'#ff6b6b'
};
const $=i=>document.getElementById(i);
const map=L.map('map').setView([-1.2,107.5],5);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:12,attribution:'© OpenStreetMap contributors'}).addTo(map);
L.marker([-3.15,101.85],{icon:L.divIcon({html:'★',className:'bkl',iconSize:[24,24]})}).addTo(map).bindPopup('<b>TN Kerinci Seblat (Bengkulu)</b><br>Kawasan konservasi utama Sumatera');
const layer=L.layerGroup().addTo(map);let filterD='ALL';
$('tag').textContent=M.sumber+' · diperbarui '+M.diperbarui;
$('tag').className='tag'+(M.sumber.startsWith('Open-Meteo')?'':' warn');
const rk=M.ringkasan;
$('bNote').textContent=`Ringkasan Evaluasi Model CNN (Resolusi 10m): Dari ${rk.total_alerts} peringatan dengan rata-rata tutupan awan tropis ${rk.rata_awan}%, model mencapai keyakinan rata-rata ${rk.rata_conf}%. Terdapat ${rk.jml_ambiguitas} peringatan di sepanjang sungai/dataran banjir yang diberi tanda peninjauan ulang (mitigasi kebingungan klasifikasi kebakaran vs banjir).`;

function bars(el,obj,colf,suf=''){const mx=Math.max(1,...Object.values(obj));$(el).innerHTML=Object.entries(obj).map(([k,v])=>
`<div class="br"><span>${k}</span><i style="width:${v/mx*50}%;background:${colf(k)}"></i>${v}${suf}</div>`).join('')||'-'}

function render(){
 const f=D.filter(d=>filterD==='ALL'?true:filterD==='AMBIGU'?d.status_model!=='Klasifikasi Yakin (Tinggi)':d.driver===filterD);
 layer.clearLayers();
 f.forEach(d=>{
  const isAmbig=d.status_model!=='Klasifikasi Yakin (Tinggi)';
  L.circleMarker([d.lat,d.lon],{radius:5+Math.min(d.luas_ha/3.5,11),color:isAmbig?'#d00000':'#111',weight:isAmbig?3:1.8,fillColor:COL[d.driver]||'#ccc',fillOpacity:.86}).addTo(layer)
   .bindPopup(`<b>${d.driver}</b> (${d.luas_ha} ha)<br><b>Kawasan:</b> ${d.kawasan}<br><b>Sensor:</b> ${d.sensor_fusi}<br><b>Rantai Co-occurrence:</b> ${d.co_occurrence}<br><b>Keyakinan CNN:</b> ${d.confidence}% · <i>${d.status_model}</i><br><small>Tanggal: ${d.tanggal} · Hujan ${d.hujan_mm} mm</small>`);
 });

 const totHa=f.reduce((a,b)=>a+b.luas_ha,0).toFixed(1);
 $('k1').textContent=f.length+' Alerts';$('k1s').textContent=`Keyakinan CNN rata-rata: ${rk.rata_conf}%`;
 $('k2').textContent=totHa+' ha';$('k2s').textContent=`Resolusi spasial piksel 10 meter`;
 $('k3').textContent=rk.rata_awan+'\% Awan';$('k3s').textContent=`Diimbangi fusi radar Sentinel-1 SAR`;
 const jmlAmb=f.filter(d=>d.status_model!=='Klasifikasi Yakin (Tinggi)').length;
 $('k4').textContent=jmlAmb+' Kasus';$('k4s').textContent=`Ambiguitas api vs banjir tepi sungai`;

 $('list').innerHTML=f.slice(0,30).map(d=>`<div class="row"><div class="m" style="background:${COL[d.driver]}">${d.luas_ha} ha</div><div><b>${d.driver}</b> · ${d.kawasan}<small>${d.tanggal} · ${d.co_occurrence} (${d.confidence}%)</small></div></div>`).join('')||'<div class="row">Tidak ada peringatan</div>';

 const dc={};Object.keys(COL).forEach(k=>dc[k]=f.filter(d=>d.driver===k).length);
 bars('b1',dc,k=>COL[k],' alert');

 const kc={};f.forEach(d=>kc[d.kawasan]=+( (kc[d.kawasan]||0)+d.luas_ha ).toFixed(1));
 const kSort=Object.fromEntries(Object.entries(kc).sort((a,b)=>b[1]-a[1]));
 bars('b2',kSort,()=>'#7ee8a2',' ha');
}
document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{filterD=b.dataset.d;document.querySelectorAll('.bar button').forEach(x=>x.classList.toggle('on',x===b));render()});
render();
</script></body></html>'''

if __name__ == "__main__":
    main()
