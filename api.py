"""
SERVER REST API SISTEM REKOMENDASI LABORATORIUM KKSI
Departemen Sistem Informasi - Universitas Andalas

Endpoint:
- GET  /        : Informasi status layanan & metadata model
- GET  /health  : Health check untuk pemantauan server Railway / Render
- POST /predict : Evaluasi dua lapis (Gatekeeper Administratif & Mesin Rekomendasi)
"""

import os
import sys
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Pastikan path modul lokal dikenali
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gatekeeper import RuleGatekeeper
from recommender import compute_recommendations, FEATURE_COLUMNS

# Inisialisasi Aplikasi Flask & CORS
app = Flask(__name__)
CORS(app)

# Konfigurasi Model
MODEL_FILENAME = 'lab_model.pkl'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)

model_bundle = None
gatekeeper = RuleGatekeeper(min_ipk=3.00)


def load_model_bundle():
    """Memuat artefak model Machine Learning ke memori RAM saat server start."""
    global model_bundle
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Berkas model '{MODEL_PATH}' belum ada. Menjalankan pipeline ekspor...")
        from export_model import export_bundle
        export_bundle()

    try:
        model_bundle = joblib.load(MODEL_PATH)
        print(f"[SUCCESS] Bundel model KKSI berhasil dimuat dari: {MODEL_PATH}")
        return model_bundle
    except Exception as e:
        print(f"[FATAL] Gagal memuat bundel model: {e}")
        model_bundle = None
        return None


# Muat model saat server pertama kali diinisialisasi
load_model_bundle()


@app.route('/', methods=['GET'])
def index():
    """Endpoint informasi status layanan API."""
    return jsonify({
        "status": "online",
        "service": "Sistem Rekomendasi Asisten Laboratorium KKSI SI UNAND",
        "version": "1.0.0",
        "architecture": "Two-Tier Hybrid Architecture",
        "model_loaded": model_bundle is not None,
        "model_file": MODEL_FILENAME,
        "supported_laboratories": [
            "Rekayasa Data dan Business Intelligence (RDBI)",
            "System Development (SD)",
            "System Enterprise (SE)",
            "Tata Kelola dan Infrastruktur Teknologi Informasi (TKITI)"
        ]
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Endpoint pemeriksaan kesehatan untuk monitoring Railway / Render."""
    is_ready = model_bundle is not None
    return jsonify({
        "healthy": is_ready,
        "model_status": "ready" if is_ready else "not_loaded"
    }), 200 if is_ready else 503


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint utama inferensi dua lapis:
    1. Lapisan 1: Rule-Based Gatekeeper (Penyaring kelayakan administratif)
    2. Lapisan 2: Hybrid Recommender (Cosine Similarity sentroid + XGBoost probabilities)
    """
    global model_bundle
    if model_bundle is None:
        model_bundle = load_model_bundle()
        if model_bundle is None:
            return jsonify({
                "success": False,
                "error": "Bundel model Machine Learning belum siap dimuat di server."
            }), 500

    try:
        data = request.get_json(force=True)
        if not data or not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Data payload JSON pendaftaran tidak valid atau kosong."
            }), 400

        # ==============================================================
        # LAPISAN 1: RULE-BASED GATEKEEPER (Evaluasi Regulasi Administratif)
        # ==============================================================
        gk_eval = gatekeeper.evaluate(data)

        if not gk_eval['is_eligible']:
            # Pendaftar tidak memenuhi syarat administratif resmi
            return jsonify({
                "success": True,
                "is_eligible": False,
                "tier_evaluated": 1,
                "status": "TIDAK LOLOS",
                "message": "Pendaftar belum memenuhi persyaratan administratif awal rekrutmen asisten laboratorium KKSI.",
                "violations": gk_eval['violations'],
                "reasons": gk_eval['reasons'],
                "details": {
                    "ipk_evaluated": gk_eval['ipk_evaluated'],
                    "semester_evaluated": gk_eval['semester_evaluated']
                }
            }), 200

        # ==============================================================
        # LAPISAN 2: HYBRID RECOMMENDATION ENGINE (Kalkulasi Afinitas & ML)
        # ==============================================================
        rec_result = compute_recommendations(model_bundle, data, alpha=0.5)

        return jsonify({
            "success": True,
            "is_eligible": True,
            "tier_evaluated": 2,
            "status": "LOLOS",
            "message": "Pendaftar memenuhi persyaratan administratif dan profil berhasil dipetakan.",
            "primary_recommendation": rec_result['primary_recommendation'],
            "backup_recommendation": rec_result['backup_recommendation'],
            "rankings": rec_result['all_rankings'],
            "competency_profile": rec_result['competency_profile'],
            "separation_margin": rec_result['separation_margin']
        }), 200

    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": f"Format data angka tidak valid: {str(ve)}"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Terjadi kesalahan saat memproses rekomendasi: {str(e)}"
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Server Flask KKSI berjalan di http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)
