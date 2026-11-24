"""
MODUL GATEKEEPER ADMINISTRATIF & PRASYARAT LABORATORIUM (LAPISAN 1)
Sistem Rekomendasi Laboratorium KKSI - Departemen Sistem Informasi UNAND

Tujuan:
1. Menyaring pendaftar asisten laboratorium berdasarkan syarat administratif mutlak
   (Semester 3 atau 5 aktif, IPK Kumulatif >= 3.00).
2. Mengevaluasi kelulusan mata kuliah prasyarat spesifik per laboratorium
   berdasarkan kurikulum Semester 3 vs Semester 5 (Targeted Lab Disqualification).
3. Menggugurkan pendaftar secara total apabila seluruh 4 laboratorium tereliminasi.
"""

import re
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
    'Belum Mengambil / Semester 3': 0.00,
    'Belum Mengambil / Sedang Ditempuh (Belum Ada KHS)': 0.00
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

COURSE_SHORT_LABELS = {
    'Nilai DDP': 'DDP',
    'Nilai SDA': 'SDA',
    'Nilai PBO': 'PBO',
    'Nilai Pemrograman Web': 'Pemrograman Web',
    'Nilai Manajemen Proses Bisnis': 'Manajemen Proses Bisnis (MPB)',
    'Nilai Infrastruktur TI': 'Infrastruktur TI',
    'Nilai Sistem Operasi': 'Sistem Operasi',
    'Nilai Keamanan Informasi': 'Keamanan Informasi',
    'Nilai Basis Data': 'Basis Data',
    'Nilai Perancangan Basis Data': 'Perancangan Basis Data (PBD)',
    'Nilai Data Mining': 'Data Mining'
}

# Matriks Prasyarat Mata Kuliah Spesifik Per Laboratorium (Semester 3 vs Semester 5)
# - sem3_required: Wajib sudah diambil dan lulus minimal C (2.00) bagi pendaftar Semester 3 maupun Semester 5.
# - sem5_required:
#   * Bagi pendaftar Semester 3: Boleh berstatus "Belum Mengambil / Sedang Ditempuh",
#     namun apabila sudah memiliki nilai KHS maka wajib minimal C (tidak boleh D atau E).
#   * Bagi pendaftar Semester 5: Wajib sudah diambil dan lulus minimal C (tidak boleh D, E, atau Belum Mengambil).
LAB_PREREQUISITES = {
    'SD': {
        'lab_name': 'Laboratorium System Development (SD)',
        'sem3_required': ['Nilai DDP', 'Nilai SDA'],
        'sem5_required': ['Nilai PBO', 'Nilai Pemrograman Web']
    },
    'RDBI': {
        'lab_name': 'Laboratorium Rekayasa Data dan Business Intelligence (RDBI)',
        'sem3_required': ['Nilai Basis Data'],
        'sem5_required': ['Nilai Perancangan Basis Data', 'Nilai Data Mining']
    },
    'SE': {
        'lab_name': 'Laboratorium System Enterprise (SE)',
        'sem3_required': [],
        'sem5_required': ['Nilai Manajemen Proses Bisnis']
    },
    'TKITI': {
        'lab_name': 'Laboratorium Tata Kelola dan Infrastruktur Teknologi Informasi (TKITI)',
        'sem3_required': ['Nilai Infrastruktur TI'],
        'sem5_required': ['Nilai Sistem Operasi', 'Nilai Keamanan Informasi']
    }
}

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


def is_grade_d(val: Any) -> bool:
    """Mendeteksi apakah nilai adalah nilai mutu D (di bawah standar minimal C)."""
    if val is None or pd.isna(val):
        return False
    v_str = str(val).strip().upper()
    return v_str in ['D', 'D+', '1', '1.0', '1.00']


def is_unregistered(val: Any) -> bool:
    """Mendeteksi apakah status mata kuliah adalah Belum Mengambil / Sedang Ditempuh."""
    if val is None or pd.isna(val):
        return True
    v_str = str(val).strip().lower()
    return 'belum' in v_str or 'sedang ditempuh' in v_str or 'semester 3' in v_str or v_str in ['-', '', 'none', 'nan', 'null']


def parse_semester_number(sem_raw: Any) -> int:
    """Mengekstrak angka semester (misal 'Semester 3' -> 3, 'Semester 5' -> 5)."""
    if sem_raw is None:
        return 0
    if isinstance(sem_raw, (int, float)):
        try:
            return int(sem_raw)
        except (ValueError, TypeError):
            return 0
    digits = re.findall(r'\d+', str(sem_raw))
    return int(digits[0]) if digits else 0


class RuleGatekeeper:
    """
    Evaluator Logika Administratif & Prasyarat Spesifik Laboratorium:
    - Syarat Mutlak 1: Mahasiswa aktif Semester 3 atau Semester 5
    - Syarat Mutlak 2: IPK Kumulatif minimal 3.00
    - Evaluasi Prasyarat Spesifik per Laboratorium (Targeted Disqualification):
      Mengecek kelulusan minimal C (2.00) dan kewajiban pengambilan mata kuliah
      pada setiap laboratorium sesuai jenjang Semester 3 vs Semester 5.
      Apabila suatu lab tidak memenuhi syarat matkulnya, HANYA lab tersebut yang
      dieliminasi (disqualified_labs). Pendaftar hanya gugur total apabila ke-4
      laboratorium tereliminasi seluruhnya.
    """

    def __init__(self, min_ipk: float = 3.00):
        self.min_ipk = min_ipk

    def evaluate_lab_prerequisites(self, data: Dict[str, Any], sem_num: int) -> Dict[str, str]:
        """
        Mengevaluasi pemenuhan prasyarat mata kuliah untuk masing-masing dari 4 laboratorium.
        Mengembalikan dict {lab_code: alasan_eliminasi} untuk laboratorium yang tereliminasi.
        """
        disqualified_labs: Dict[str, str] = {}
        effective_sem = sem_num if sem_num in [3, 5] else 5

        for lab_code, rule in LAB_PREREQUISITES.items():
            lab_issues: List[str] = []

            # 1. Cek mata kuliah wajib bagi Semester 3 maupun Semester 5
            for col in rule['sem3_required']:
                val = data.get(col)
                label = COURSE_SHORT_LABELS.get(col, col)
                if is_unregistered(val):
                    lab_issues.append(f"{label} (Belum Mengambil)")
                else:
                    score = grade_to_numeric(val)
                    if is_grade_e(val) or score < 2.00:
                        grade_disp = str(val).strip()
                        lab_issues.append(f"{label} (Nilai {grade_disp} < C)")

            # 2. Cek mata kuliah lanjutan (Wajib sudah diambil di Semester 5; boleh Belum Mengambil di Semester 3)
            for col in rule['sem5_required']:
                val = data.get(col)
                label = COURSE_SHORT_LABELS.get(col, col)
                if is_unregistered(val):
                    if effective_sem == 5:
                        lab_issues.append(f"{label} (Wajib sudah diambil di Semester 5)")
                else:
                    score = grade_to_numeric(val)
                    if is_grade_e(val) or score < 2.00:
                        grade_disp = str(val).strip()
                        lab_issues.append(f"{label} (Nilai {grade_disp} < C)")

            if lab_issues:
                disqualified_labs[lab_code] = f"Prasyarat tidak terpenuhi: {', '.join(lab_issues)}"

        return disqualified_labs

    def evaluate(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        data = normalize_input_keys(raw_data)
        violations: List[str] = []
        reasons: List[str] = []

        # 1. Validasi Semester Pendaftaran (Wajib Semester 3 atau Semester 5)
        sem_raw = data.get('Semester Pendaftaran', '')
        sem_num = parse_semester_number(sem_raw)
        if sem_num not in [3, 5]:
            violations.append("Semester pendaftaran harus mahasiswa aktif Semester 3 atau Semester 5")
            reasons.append("Semester di luar ketentuan rekrutmen (wajib Semester 3 atau 5)")

        # 2. Validasi IPK Kumulatif (Wajib >= 3.00)
        ipk_raw = data.get('IPK Kumulatif', 0.0)
        try:
            ipk = float(str(ipk_raw).replace(',', '.'))
        except (ValueError, TypeError):
            ipk = 0.0

        if ipk < self.min_ipk:
            violations.append(f"IPK Kumulatif ({ipk:.2f}) berada di bawah batas minimum kelayakan {self.min_ipk:.2f}")
            reasons.append(f"IPK < {self.min_ipk:.2f}")

        # 3. Evaluasi Prasyarat Spesifik per Laboratorium (Targeted Disqualification)
        disqualified_labs = self.evaluate_lab_prerequisites(data, sem_num)
        qualified_labs = [code for code in ['RDBI', 'SD', 'SE', 'TKITI'] if code not in disqualified_labs]

        # 4. Apabila ke-4 laboratorium tereliminasi semuanya, pendaftar gugur total
        if len(disqualified_labs) == len(LAB_PREREQUISITES):
            for lab_code, reason_text in disqualified_labs.items():
                violations.append(f"Lab {lab_code}: {reason_text}")
            reasons.append("Tidak memenuhi prasyarat mata kuliah minimal C pada seluruh 4 laboratorium KKSI")

        is_eligible = len(violations) == 0

        return {
            'is_eligible': is_eligible,
            'violations': violations,
            'reasons': reasons,
            'disqualified_labs': disqualified_labs,
            'qualified_labs': qualified_labs,
            'status': 'LOLOS' if is_eligible else 'TIDAK LOLOS',
            'ipk_evaluated': ipk,
            'semester_evaluated': data.get('Semester Pendaftaran', f"Semester {sem_num}" if sem_num else '')
        }
