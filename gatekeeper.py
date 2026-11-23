"""
MODUL GATEKEEPER ADMINISTRATIF (LAPISAN 1)
Sistem Rekomendasi Laboratorium KKSI - Departemen Sistem Informasi UNAND

Tujuan:
Menyaring pendaftar asisten laboratorium berdasarkan regulasi akademik resmi
secara deterministik sebelum masuk ke Mesin Rekomendasi Machine Learning (Lapisan 2).
"""

from typing import Dict, Any, List
import pandas as pd

# Konversi nilai huruf resmi Universitas Andalas ke bobot numerik
GRADE_WEIGHT_MAP = {
    'A': 4.00,
    'A-': 3.75,
    'B+': 3.50,
    'B': 3.00,
    'B-': 2.75,
    'C+': 2.50,
    'C': 2.00,
    'D': 1.00,
    'E': 0.00,
    'Belum Mengambil': 0.00,
    'Belum Mengambil / Semester 3': 0.00
}

# 11 Mata Kuliah Fondasi Keilmuan KKSI
COURSE_COLUMNS = [
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
    'Nilai Data Mining'
]

# Pemetaan alias kolom untuk fleksibilitas pembacaan
CANONICAL_MAP = {
    'semester pendaftaran': 'Semester Pendaftaran',
    'semester': 'Semester Pendaftaran',
    'ipk kumulatif': 'IPK Kumulatif',
    'ipk': 'IPK Kumulatif',
    'nilai ddp': 'Nilai DDP',
    'ddp': 'Nilai DDP',
    'nilai sda': 'Nilai SDA',
    'sda': 'Nilai SDA',
    'nilai pbo': 'Nilai PBO',
    'pbo': 'Nilai PBO',
    'nilai pemrograman web': 'Nilai Pemrograman Web',
    'pemrograman web': 'Nilai Pemrograman Web',
    'nilai manajemen proses bisnis': 'Nilai Manajemen Proses Bisnis',
    'manajemen proses bisnis': 'Nilai Manajemen Proses Bisnis',
    'mpb': 'Nilai Manajemen Proses Bisnis',
    'nilai infrastruktur ti': 'Nilai Infrastruktur TI',
    'infrastruktur ti': 'Nilai Infrastruktur TI',
    'nilai sistem operasi': 'Nilai Sistem Operasi',
    'sistem operasi': 'Nilai Sistem Operasi',
    'so': 'Nilai Sistem Operasi',
    'nilai keamanan informasi': 'Nilai Keamanan Informasi',
    'keamanan informasi': 'Nilai Keamanan Informasi',
    'nilai basis data': 'Nilai Basis Data',
    'basis data': 'Nilai Basis Data',
    'nilai perancangan basis data': 'Nilai Perancangan Basis Data',
    'perancangan basis data': 'Nilai Perancangan Basis Data',
    'pbd': 'Nilai Perancangan Basis Data',
    'nilai data mining': 'Nilai Data Mining',
    'data mining': 'Nilai Data Mining',
    'tools database & sql': 'Tools Database & SQL',
    'tools visualisasi data': 'Tools Visualisasi Data',
    'tools python analitik': 'Tools Python Analitik',
    'tools pemrograman web/app': 'Tools Pemrograman Web/App',
    'tools desain ui/ux': 'Tools Desain UI/UX',
    'tools pemodelan bpmn': 'Tools Pemodelan BPMN',
    'tools erp & bisnis': 'Tools ERP & Bisnis',
    'tools jaringan & os': 'Tools Jaringan & OS',
    'minat data & analitik': 'Minat Data & Analitik',
    'minat proses bisnis & erp': 'Minat Proses Bisnis & ERP',
    'minat coding & pemrograman': 'Minat Coding & Pemrograman',
    'minat server & jaringan': 'Minat Server & Jaringan'
}


def normalize_input_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Menyelaraskan kunci masukan agar sesuai nama atribut resmi."""
    cleaned = {}
    for k, v in data.items():
        k_clean = str(k).strip().lower()
        canonical_k = CANONICAL_MAP.get(k_clean, k)
        cleaned[canonical_k] = v
    return cleaned


def grade_to_numeric(val: Any) -> float:
    """Mengonversi nilai huruf atau teks status ke bobot numerik float [0.0 - 4.0]."""
    if val is None or pd.isna(val):
        return 0.0
    v_str = str(val).strip()
    if v_str in GRADE_WEIGHT_MAP:
        return float(GRADE_WEIGHT_MAP[v_str])
    try:
        f_val = float(v_str.replace(',', '.'))
        return max(0.0, min(4.0, f_val))
    except ValueError:
        return 0.0


def is_grade_e(val: Any) -> bool:
    """Mendeteksi apakah nilai adalah nilai mutu E (Gagal Mutlak)."""
    if val is None or pd.isna(val):
        return False
    v_str = str(val).strip().upper()
    return v_str == 'E' or v_str == '0'


def is_unregistered(val: Any) -> bool:
    """Mendeteksi apakah status mata kuliah adalah Belum Mengambil."""
    if val is None or pd.isna(val):
        return True
    v_str = str(val).strip().lower()
    return 'belum' in v_str or 'semester 3' in v_str or v_str in ['-', '']


class RuleGatekeeper:
    """
    Evaluator Logika Administratif Lapisan 1:
    - Aturan 1: Mahasiswa aktif Semester 3 atau Semester 5
    - Aturan 2: IPK Kumulatif minimal 3.00
    - Aturan 3: Bebas nilai mutu E pada 11 mata kuliah keilmuan fondasi
    - Aturan 4: Mata kuliah fondasi prasyarat (DDP, Basis Data) lulus minimal C
    """

    def __init__(self, min_ipk: float = 3.00):
        self.min_ipk = min_ipk

    def evaluate(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        data = normalize_input_keys(raw_data)
        violations = []
        reasons = []

        # 1. Validasi Semester
        sem_val = str(data.get('Semester Pendaftaran', '')).strip().lower()
        if not ('3' in sem_val or '5' in sem_val):
            violations.append("Semester pendaftaran harus Semester 3 atau Semester 5 aktif")
            reasons.append("Semester di luar ketentuan rekrutmen")

        # 2. Validasi IPK Kumulatif
        ipk_raw = data.get('IPK Kumulatif', 0.0)
        try:
            ipk = float(str(ipk_raw).replace(',', '.'))
        except (ValueError, TypeError):
            ipk = 0.0

        if ipk < self.min_ipk:
            violations.append(f"IPK Kumulatif ({ipk:.2f}) berada di bawah batas minimum {self.min_ipk:.2f}")
            reasons.append(f"IPK < {self.min_ipk:.2f}")

        # 3. Validasi Bebas Nilai E
        failed_courses_e = []
        for course in COURSE_COLUMNS:
            grade_val = data.get(course)
            if is_grade_e(grade_val):
                clean_name = course.replace('Nilai ', '')
                failed_courses_e.append(clean_name)

        if failed_courses_e:
            violations.append(f"Terdapat nilai mutu E pada mata kuliah: {', '.join(failed_courses_e)}")
            reasons.append(f"Nilai E pada: {', '.join(failed_courses_e)}")

        # 4. Validasi Kelulusan Mata Kuliah Fondasi (DDP dan Basis Data wajib >= C)
        core_requirements = [
            ('Nilai DDP', 'Dasar-dasar Pemrograman'),
            ('Nilai Basis Data', 'Basis Data')
        ]
        failed_core = []
        for col_name, label in core_requirements:
            val = data.get(col_name)
            if not is_unregistered(val):
                num_score = grade_to_numeric(val)
                if num_score < 2.00:  # Nilai C = 2.00
                    failed_core.append(f"{label} ({str(val).strip()})")

        if failed_core:
            violations.append(f"Mata kuliah fondasi utama wajib lulus minimal C: {', '.join(failed_core)}")
            reasons.append(f"Fondasi belum lulus: {', '.join(failed_core)}")

        is_eligible = len(violations) == 0

        return {
            'is_eligible': is_eligible,
            'violations': violations,
            'reasons': reasons,
            'status': 'LOLOS' if is_eligible else 'TIDAK LOLOS',
            'ipk_evaluated': ipk,
            'semester_evaluated': data.get('Semester Pendaftaran', '')
        }
