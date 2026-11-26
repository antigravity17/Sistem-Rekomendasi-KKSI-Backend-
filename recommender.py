"""
MODUL MESIN REKOMENDASI HYBRID (LAPISAN 2)
Sistem Rekomendasi Laboratorium KKSI - Departemen Sistem Informasi UNAND

Tujuan:
1. Melakukan normalisasi DomainFixedScaler [0, 1] pada 23 fitur kompetensi.
2. Menghitung Cosine Similarity terhadap sentroid 4 lab KKSI dan mengestimasi probabilitas XGBoost.
3. Memadukan skor komposit True Hybrid (0.5 * Cosine + 0.5 * XGBoost).
4. Menerapkan Eliminasi Spesifik Per Laboratorium (Targeted Lab Disqualification) sehingga
   laboratorium yang prasyarat mata kuliahnya tidak terpenuhi otomatis dicoret dari opsi
   Rekomendasi Utama (#1) maupun Pilihan Cadangan (#2) dan digeser ke urutan bawah.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gatekeeper import normalize_input_keys, grade_to_numeric, RuleGatekeeper

# 23 Fitur Murni Kompetensi Sistem KKSI
FEATURE_COLUMNS = [
    # 11 Mata Kuliah Keilmuan Fondasi
    'Nilai DDP',
    'Nilai SDA',
    'Nilai PBO',
    'Nilai Pemrograman Web',
    'Nilai Manajemen Proses Bisnis',
    'Nilai Infrastruktur TI',
    'Nilai Sistem Operasi',
    'Nilai Keamanan Informasi',
    'Nilai Basis Data',
    'Nilai Perancangan Basis Data',
    'Nilai Data Mining',
    # 8 Kemahiran Tools Praktis
    'Tools Database & SQL',
    'Tools Visualisasi Data',
    'Tools Python Analitik',
    'Tools Pemrograman Web/App',
    'Tools Desain UI/UX',
    'Tools Pemodelan BPMN',
    'Tools ERP & Bisnis',
    'Tools Jaringan & OS',
    # 4 Minat Bidang Laboratorium
    'Minat Data & Analitik',
    'Minat Proses Bisnis & ERP',
    'Minat Coding & Pemrograman',
    'Minat Server & Jaringan'
]

# 4 Laboratorium Kelompok Keilmuan Sistem Informasi (KKSI) UNAND
LAB_CLASSES = [
    'Laboratorium Rekayasa Data dan Business Intelligence (RDBI)',
    'Laboratorium System Development (SD)',
    'Laboratorium System Enterprise (SE)',
    'Laboratorium Tata Kelola dan Infrastruktur Sistem Informasi (TKITI)'
]

LAB_SHORT_NAME = {
    'Laboratorium Rekayasa Data dan Business Intelligence (RDBI)': 'RDBI',
    'Laboratorium System Development (SD)': 'SD',
    'Laboratorium System Enterprise (SE)': 'SE',
    'Laboratorium Tata Kelola dan Infrastruktur Sistem Informasi (TKITI)': 'TKITI'
}

LAB_FULL_TITLE = {
    'RDBI': 'Rekayasa Data dan Business Intelligence',
    'SD': 'System Development',
    'SE': 'System Enterprise',
    'TKITI': 'Tata Kelola dan Infrastruktur Teknologi Informasi'
}


class DomainFixedScaler:
    """
    Penskalaan Fitur Berbasis Batas Teoretis Domain (Domain-Aware Fixed Scaling):
    - 11 Nilai Mata Kuliah (0.0 s.d. 4.0) -> dinormalisasi dengan nilai / 4.0
    - 8 Tools & 4 Minat (Skala Likert 1 s.d. 5) -> dinormalisasi dengan (skor - 1.0) / 4.0
    Menjamin seluruh 23 dimensi berada pada rentang teoretis [0.0, 1.0] tanpa data leakage.
    """

    def __init__(self, num_courses: int = 11):
        self.num_courses = num_courses

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.array(X, dtype=float, copy=True)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # 11 Mata Kuliah: Nilai / 4.0
        X_arr[:, :self.num_courses] = np.clip(X_arr[:, :self.num_courses] / 4.0, 0.0, 1.0)

        # 12 Tools dan Minat: (Skor - 1.0) / 4.0
        X_arr[:, self.num_courses:] = np.clip((X_arr[:, self.num_courses:] - 1.0) / 4.0, 0.0, 1.0)

        return X_arr

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


def extract_raw_features(raw_data: Dict[str, Any]) -> np.ndarray:
    """Mengekstrak 23 nilai numerik dari payload masukan pengguna."""
    data = normalize_input_keys(raw_data)
    vec = []
    for col in FEATURE_COLUMNS:
        raw_val = data.get(col, 0.0)
        if col.startswith('Nilai '):
            vec.append(grade_to_numeric(raw_val))
        else:
            try:
                f_val = float(str(raw_val).replace(',', '.'))
                vec.append(max(1.0, min(5.0, f_val)))
            except (ValueError, TypeError):
                vec.append(1.0)
    return np.array(vec, dtype=float).reshape(1, -1)


def compute_recommendations(
    bundle: Dict[str, Any],
    raw_input: Dict[str, Any],
    alpha: float = 0.5,
    disqualified_labs: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Menghitung inferensi rekomendasi dua lapis secara end-to-end:
    - Cosine Similarity terhadap sentroid profil 4 lab
    - Probabilitas prediksi XGBoost
    - Skor Komposit: alpha * Cosine + (1 - alpha) * XGBoost
    - Pemeringkatan Terfilter: Laboratorium yang memenuhi prasyarat mata kuliah
      ditempatkan di peringkat teratas (#1, #2), sedangkan laboratorium yang
      tereliminasi (disqualified_labs) digeser ke urutan bawah.
    """
    scaler: DomainFixedScaler = bundle['scaler']
    centroids: np.ndarray = bundle['centroids']
    xgb_model = bundle['xgb_model']
    lab_classes: List[str] = bundle['lab_classes']

    # Evaluasi otomatis disqualified_labs apabila fungsi dipanggil langsung
    if disqualified_labs is None:
        gk_eval = RuleGatekeeper().evaluate(raw_input)
        disqualified_labs = gk_eval.get('disqualified_labs', {})

    # 1. Ekstraksi dan Normalisasi Vektor 23 Dimensi
    X_raw = extract_raw_features(raw_input)
    X_scaled = scaler.transform(X_raw)

    # 2. Perhitungan Cosine Similarity terhadap Sentroid (Skala 0 - 100%)
    sims = cosine_similarity(X_scaled, centroids)[0] * 100.0

    # 3. Perhitungan Probabilitas XGBoost (Skala 0 - 100%)
    probs = xgb_model.predict_proba(X_scaled)[0] * 100.0

    # 4. Penyatuan Skor Komposit True Hybrid (alpha = 0.5)
    composite_scores = (alpha * sims) + ((1.0 - alpha) * probs)

    # 5. Pemisahan Laboratorium Memenuhi Syarat vs Tereliminasi Prasyarat Mata Kuliah
    raw_sorted_indices = np.argsort(-composite_scores)
    qualified_indices = []
    disqualified_indices = []

    for idx in raw_sorted_indices:
        full_lab_name = lab_classes[idx]
        short_code = LAB_SHORT_NAME.get(full_lab_name, full_lab_name)
        if short_code in disqualified_labs:
            disqualified_indices.append(idx)
        else:
            qualified_indices.append(idx)

    ordered_indices = qualified_indices + disqualified_indices

    rankings = []
    for rank_pos, idx in enumerate(ordered_indices, start=1):
        full_lab_name = lab_classes[idx]
        short_code = LAB_SHORT_NAME.get(full_lab_name, full_lab_name)
        clean_title = LAB_FULL_TITLE.get(short_code, full_lab_name)
        is_disq = short_code in disqualified_labs
        disq_reason = disqualified_labs.get(short_code) if is_disq else None

        if is_disq:
            match_label = 'Tereliminasi (Prasyarat)'
        elif rank_pos == 1:
            match_label = 'Sangat Cocok (Utama)'
        elif rank_pos == 2:
            match_label = 'Pilihan Cadangan Ideal'
        else:
            match_label = 'Cukup'

        rankings.append({
            'rank': rank_pos,
            'lab_code': short_code,
            'lab_name': f"Laboratorium {clean_title}",
            'composite_score': round(float(composite_scores[idx]), 2),
            'cosine_similarity': round(float(sims[idx]), 2),
            'xgb_probability': round(float(probs[idx]), 2),
            'match_label': match_label,
            'is_disqualified': is_disq,
            'disqualification_reason': disq_reason
        })

    # 6. Ringkasan Skor Indeks Kompetensi (Hanya merata-ratakan mata kuliah yang sudah ditempuh)
    academic_scores = X_raw[0, :11]
    taken_academic = [s for s in academic_scores if s > 0.0]
    mean_acad = np.mean(taken_academic) if len(taken_academic) > 0 else 0.0

    tool_scores = X_raw[0, 11:19]
    interest_scores = X_raw[0, 19:]

    competency_profile = {
        'academic_competency_score': round(float(mean_acad / 4.0 * 100.0), 1),
        'technical_skills_score': round(float((np.mean(tool_scores) - 1.0) / 4.0 * 100.0), 1),
        'domain_interest_score': round(float((np.mean(interest_scores) - 1.0) / 4.0 * 100.0), 1)
    }

    # 7. Perhitungan Margin Keunggulan antara Peringkat #1 dan Peringkat #2 yang Memenuhi Syarat
    if len(rankings) > 1 and not rankings[1]['is_disqualified']:
        sep_margin = round(float(rankings[0]['composite_score'] - rankings[1]['composite_score']), 2)
    else:
        # Saat 3 lab lain terdiskualifikasi (hanya 1 lab lolos), tidak ada margin kompetisi
        sep_margin = 0.0

    return {
        'primary_recommendation': rankings[0],
        'backup_recommendation': rankings[1],
        'all_rankings': rankings,
        'disqualified_labs': disqualified_labs,
        'competency_profile': competency_profile,
        'separation_margin': max(0.0, sep_margin)
    }
