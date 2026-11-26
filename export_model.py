"""
SKRIP EKSPOR DAN SERIALISASI BUNDEL MODEL HYBRID
Sistem Rekomendasi Laboratorium KKSI - Departemen Sistem Informasi UNAND

Fungsi:
Melatih model XGBoost dan menghitung sentroid Cosine 4 lab KKSI pada 70 data
asisten riil, lalu mengemasnya menjadi berkas tunggal 'lab_model.pkl'.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# Tambahkan direktori saat ini ke sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gatekeeper import CANONICAL_MAP, grade_to_numeric
from recommender import FEATURE_COLUMNS, LAB_CLASSES, LAB_SHORT_NAME, DomainFixedScaler, compute_recommendations


def export_bundle():
    print("=" * 80)
    print("MEMULAI PROSES PELATIHAN & EKSPOR BUNDEL MODEL REKOMENDASI KKSI")
    print("=" * 80)

    # 1. Lokasi Berkas Dataset 70 Data Asli Riil (Dinamis & Portabel)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    dataset_candidates = [
        os.path.join(base_dir, 'Dataset_70_Asisten_Survei.xlsx'),
        os.path.join(base_dir, 'Dataset_70_Asisten_Survei.csv'),
        os.path.join(parent_dir, 'ini', 'Dataset_70_Asisten_Survei.xlsx'),
        os.path.join(parent_dir, 'Dataset_70_Asisten_Survei.xlsx'),
        os.path.join(parent_dir, '..', 'ini', 'Dataset_70_Asisten_Survei.xlsx')
    ]

    dataset_path = None
    for p in dataset_candidates:
        if os.path.exists(p):
            dataset_path = p
            break

    if dataset_path is None:
        raise FileNotFoundError("Berkas Dataset_70_Asisten_Survei.xlsx tidak ditemukan!")

    print(f"• Memuat data latih primer riil dari: {dataset_path}")
    df_raw = pd.read_excel(dataset_path)
    print(f"• Jumlah sampel: {len(df_raw)} asisten riil")

    # 2. Harmonisasi Kolom & Target
    rename_cols = {}
    for col in df_raw.columns:
        c_clean = str(col).strip().lower()
        if c_clean in CANONICAL_MAP:
            rename_cols[col] = CANONICAL_MAP[c_clean]
    df_norm = df_raw.rename(columns=rename_cols).copy()

    # Ekstraksi Target Laboratorium
    target_col = None
    for c in ['Laboratorium Diterima', 'Laboratorium', 'Target']:
        if c in df_norm.columns:
            target_col = c
            break

    if target_col is None:
        raise KeyError("Kolom target laboratorium tidak ditemukan!")

    # Standardisasi Nama Laboratorium
    def map_to_official_lab(val):
        v = str(val).strip()
        for official in LAB_CLASSES:
            short = LAB_SHORT_NAME[official]
            if short in v or official in v:
                return official
        return val

    df_norm['Lab_Official'] = df_norm[target_col].apply(map_to_official_lab)
    y_labels = df_norm['Lab_Official'].values

    # Buat ID numerik untuk label target [0, 1, 2, 3]
    lab_to_id = {lab: idx for idx, lab in enumerate(LAB_CLASSES)}
    y_ids = np.array([lab_to_id[l] for l in y_labels])

    # 3. Ekstraksi Matriks Fitur 23 Dimensi
    X_raw_list = []
    for _, row in df_norm.iterrows():
        row_vec = []
        for col in FEATURE_COLUMNS:
            val = row.get(col, 0.0)
            if col.startswith('Nilai '):
                row_vec.append(grade_to_numeric(val))
            else:
                try:
                    f_val = float(str(val).replace(',', '.'))
                    row_vec.append(max(1.0, min(5.0, f_val)))
                except:
                    row_vec.append(1.0)
        X_raw_list.append(row_vec)

    X_raw = np.array(X_raw_list, dtype=float)

    # 4. Penskalaan Domain-Aware [0, 1]
    scaler = DomainFixedScaler(num_courses=11)
    X_scaled = scaler.fit_transform(X_raw)

    # 5. Perhitungan Vektor Sentroid untuk 4 Laboratorium KKSI
    centroids = np.zeros((len(LAB_CLASSES), 23), dtype=float)
    print("• Menghitung sentroid kompetensi 4 laboratorium KKSI:")
    for idx, lab in enumerate(LAB_CLASSES):
        mask = (y_ids == idx)
        centroids[idx] = X_scaled[mask].mean(axis=0)
        print(f"  - {LAB_SHORT_NAME[lab]:6s} : {mask.sum()} responden asisten riil")

    # 6. Pelatihan Model XGBoost (Hyperparameter Sesuai Laporan: n_est=40, depth=3, lr=0.1)
    print("• Melatih XGBClassifier multi-class (multi:softprob)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=40,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_scaled, y_ids)

    # 7. Pengemasan Bundel Model
    bundle = {
        'scaler': scaler,
        'centroids': centroids,
        'xgb_model': xgb_model,
        'lab_classes': LAB_CLASSES,
        'lab_short': LAB_SHORT_NAME,
        'feature_cols': FEATURE_COLUMNS,
        'metadata': {
            'dataset_source': '70 Data Riil Asisten Survei UNAND',
            'num_features': 23,
            'num_classes': 4,
            'composite_weight_alpha': 0.5
        }
    }

    output_path = os.path.join(os.path.dirname(__file__), 'lab_model.pkl')
    joblib.dump(bundle, output_path)
    print(f"• BERHASIL: Bundel model tersimpan di '{output_path}' ({os.path.getsize(output_path)} bytes)")

    # 8. Uji Coba Verifikasi Mandiri
    print("-" * 80)
    print("UJI VALIDASI BUNDEL MODEL TEREKSPOR:")
    sample_input = {col: 3.5 if col.startswith('Nilai ') else 4 for col in FEATURE_COLUMNS}
    test_result = compute_recommendations(bundle, sample_input, alpha=0.5)
    print(f"• Rekomendasi Utama   : {test_result['primary_recommendation']['lab_name']} ({test_result['primary_recommendation']['composite_score']}%)")
    print(f"• Rekomendasi Cadangan: {test_result['backup_recommendation']['lab_name']} ({test_result['backup_recommendation']['composite_score']}%)")
    print("• Status Uji          : VALID & SIAP DIGUNAKAN DI API FLASK")
    print("=" * 80)


if __name__ == '__main__':
    export_bundle()
