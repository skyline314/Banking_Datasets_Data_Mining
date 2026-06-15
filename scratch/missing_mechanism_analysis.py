"""
==========================================================================
ANALISIS MEKANISME MISSING DATA
==========================================================================
Tujuan: Menentukan apakah missing data bersifat:
  - MCAR (Missing Completely at Random)
  - MAR  (Missing at Random)
  - MNAR (Missing Not at Random)

Ini penting karena:
  - MCAR → aman di-drop, tidak ada bias
  - MAR  → bisa di-impute (median, KNN, dll) ATAU drop jika jumlah kecil
  - MNAR → harus hati-hati, drop/impute keduanya bisa bias

Metode yang digunakan:
  1. Pola missing (visualisasi co-occurrence)
  2. T-test / Chi-square — apakah baris missing berbeda dari non-missing
  3. Little's MCAR approximation — korelasi antar indikator missing
  4. Logistic regression — prediksi missingness dari variabel lain
==========================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ── Load raw data ─────────────────────────────────────────────────────────
RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "bank_transactions.csv"
print(f"Loading data from: {RAW_PATH}")
df = pd.read_csv(RAW_PATH)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# Standardize column names untuk kemudahan
import re
def to_snake(col):
    col = re.sub(r"([a-z])([A-Z])", r"\1_\2", col)
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^\w]", "", col)
    return col.lower()

df.columns = [to_snake(c) for c in df.columns]

# ══════════════════════════════════════════════════════════════════════════
# BAGIAN 1: IDENTIFIKASI MISSING
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("BAGIAN 1: PROFIL MISSING DATA")
print("=" * 70)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(4)
missing_df = pd.DataFrame({
    "null_count": missing,
    "null_pct": missing_pct,
}).sort_values("null_count", ascending=False)

print(missing_df[missing_df["null_count"] > 0].to_string())
print(f"\nTotal baris dengan ≥1 null: {df.isnull().any(axis=1).sum():,}")
print(f"Total baris lengkap: {df.dropna().shape[0]:,}")

# Kolom yang punya missing
cols_with_missing = missing[missing > 0].index.tolist()
print(f"\nKolom dengan missing: {cols_with_missing}")

# ══════════════════════════════════════════════════════════════════════════
# BAGIAN 2: POLA CO-OCCURRENCE MISSING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 2: POLA CO-OCCURRENCE MISSING")
print("=" * 70)
print("Apakah missing di satu kolom berkorelasi dengan missing di kolom lain?")
print("Jika ya → kemungkinan MAR atau MNAR (ada pola sistematis)")
print("Jika tidak → mendukung MCAR\n")

# Buat indikator missing (1 = missing, 0 = ada)
missing_indicators = df[cols_with_missing].isnull().astype(int)
missing_indicators.columns = [f"miss_{c}" for c in cols_with_missing]

# Korelasi antar indikator missing
print("Korelasi antar indikator missing:")
corr = missing_indicators.corr()
print(corr.round(4).to_string())

# Interpretasi
print("\nInterpretasi:")
for i, c1 in enumerate(corr.columns):
    for j, c2 in enumerate(corr.columns):
        if i < j:
            r = corr.loc[c1, c2]
            strength = "KUAT" if abs(r) > 0.3 else "SEDANG" if abs(r) > 0.1 else "LEMAH"
            print(f"  {c1} ↔ {c2}: r = {r:.4f} ({strength})")

# ══════════════════════════════════════════════════════════════════════════
# BAGIAN 3: T-TEST — Apakah baris missing BERBEDA dari non-missing?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 3: T-TEST (Numeric) & CHI-SQUARE (Categorical)")
print("=" * 70)
print("Logika: Untuk setiap kolom yang punya missing, kita bandingkan")
print("baris yang MISSING vs NON-MISSING pada kolom-kolom LAIN.")
print("Jika ada perbedaan signifikan → MAR (missingness tergantung variabel lain)")
print("Jika tidak ada perbedaan → mendukung MCAR\n")

# Konversi numerik
df["cust_account_balance"] = pd.to_numeric(df["cust_account_balance"], errors="coerce")
df["transaction_amount_inr"] = pd.to_numeric(df["transaction_amount_inr"], errors="coerce")

numeric_test_cols = ["cust_account_balance", "transaction_amount_inr"]

for target_col in cols_with_missing:
    print(f"\n--- Menguji: '{target_col}' (null={df[target_col].isnull().sum():,}) ---")
    
    is_missing = df[target_col].isnull()
    group_missing = df[is_missing]
    group_present = df[~is_missing]
    
    # T-test pada kolom numerik lain
    for num_col in numeric_test_cols:
        if num_col == target_col:
            continue
        
        vals_missing = group_missing[num_col].dropna()
        vals_present = group_present[num_col].dropna()
        
        if len(vals_missing) < 2:
            print(f"  [{num_col}] Terlalu sedikit data untuk t-test")
            continue
        
        t_stat, p_value = stats.ttest_ind(vals_missing, vals_present, equal_var=False)
        sig = "*** SIGNIFIKAN" if p_value < 0.05 else "tidak signifikan"
        
        mean_miss = vals_missing.mean()
        mean_pres = vals_present.mean()
        
        print(f"  [{num_col}] t={t_stat:.3f}, p={p_value:.6f} {sig}")
        print(f"    Mean (missing group): {mean_miss:,.2f}")
        print(f"    Mean (present group): {mean_pres:,.2f}")
        print(f"    Selisih: {abs(mean_miss - mean_pres):,.2f}")
    
    # Chi-square pada kolom kategorikal
    for cat_col in ["cust_gender"]:
        if cat_col == target_col:
            continue
        
        # Buat contingency table
        ct = pd.crosstab(is_missing, df[cat_col].fillna("UNKNOWN"))
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        
        chi2, p_value, dof, expected = stats.chi2_contingency(ct)
        sig = "*** SIGNIFIKAN" if p_value < 0.05 else "tidak signifikan"
        print(f"  [{cat_col}] χ²={chi2:.3f}, p={p_value:.6f} {sig}")

# ══════════════════════════════════════════════════════════════════════════
# BAGIAN 4: DISTRIBUSI — Apakah missing terpusat di subset tertentu?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 4: DISTRIBUSI MISSING PER KATEGORI")
print("=" * 70)
print("Jika missing rate TIDAK SAMA di semua kategori → MAR")
print("Jika missing rate MERATA di semua kategori → mendukung MCAR\n")

# Missing rate per gender
if "cust_gender" in df.columns:
    print("Missing rate per Gender:")
    for col in cols_with_missing:
        if col == "cust_gender":
            continue
        gender_miss = df.groupby("cust_gender")[col].apply(
            lambda x: x.isnull().mean() * 100
        ).round(4)
        print(f"  {col}:")
        for gender, rate in gender_miss.items():
            print(f"    {gender}: {rate:.4f}%")

# Missing rate per location (top 10)
if "cust_location" in df.columns:
    print("\nMissing rate per Location (Top 10 cities):")
    top_locs = df["cust_location"].value_counts().head(10).index
    for col in ["cust_account_balance", "customer_dob"]:
        if col not in cols_with_missing:
            continue
        loc_miss = df[df["cust_location"].isin(top_locs)].groupby("cust_location")[col].apply(
            lambda x: x.isnull().mean() * 100
        ).round(4).sort_values(ascending=False)
        print(f"  {col}:")
        for loc, rate in loc_miss.items():
            print(f"    {loc}: {rate:.4f}%")

# ══════════════════════════════════════════════════════════════════════════
# BAGIAN 5: LOGISTIC REGRESSION — Prediksi missingness
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 5: LOGISTIC REGRESSION (Prediksi Missingness)")
print("=" * 70)
print("Jika variabel lain BISA memprediksi missingness → MAR")
print("Jika TIDAK bisa memprediksi → mendukung MCAR")
print("(Pseudo R² mendekati 0 = MCAR, jauh dari 0 = MAR)\n")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Siapkan fitur prediktor (kolom yang TIDAK missing)
# Gunakan transaction_amount dan balance (konversi ke numerik)
predictors = df[["transaction_amount_inr"]].copy()
predictors = predictors.dropna()

for target_col in ["customer_dob", "cust_gender", "cust_location", "cust_account_balance"]:
    if target_col not in cols_with_missing:
        continue
    
    # Target: apakah kolom ini missing?
    y = df.loc[predictors.index, target_col].isnull().astype(int)
    X = predictors.values
    
    if y.sum() < 10:
        print(f"  [{target_col}] Terlalu sedikit missing untuk model")
        continue
    
    try:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # McFadden's pseudo R²
        from sklearn.metrics import log_loss
        ll_model = -log_loss(y, model.predict_proba(X), normalize=False)
        ll_null = -log_loss(y, np.full((len(y), 2), [1 - y.mean(), y.mean()]), normalize=False)
        pseudo_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0
        
        accuracy = model.score(X, y)
        
        print(f"  [{target_col}]")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Pseudo R²: {pseudo_r2:.6f}")
        
        if pseudo_r2 < 0.02:
            print(f"    → R² sangat rendah → Missingness TIDAK bisa diprediksi → mendukung MCAR")
        elif pseudo_r2 < 0.10:
            print(f"    → R² rendah → Missingness sedikit terprediksi → kemungkinan MAR lemah")
        else:
            print(f"    → R² signifikan → Missingness BISA diprediksi → kemungkinan MAR")
    except Exception as e:
        print(f"  [{target_col}] Error: {e}")

# ══════════════════════════════════════════════════════════════════════════
# BAGIAN 6: LITTLE'S MCAR TEST (Approximation)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 6: LITTLE'S MCAR TEST (Approximasi)")
print("=" * 70)
print("Little's MCAR test menguji H0: data MCAR")
print("Jika p > 0.05 → GAGAL tolak H0 → data MCAR")
print("Jika p < 0.05 → TOLAK H0 → data BUKAN MCAR (kemungkinan MAR atau MNAR)\n")

# Simplified approach: bandingkan mean variabel numerik antara 
# berbagai pola missing (complete vs incomplete cases)
numeric_for_test = ["cust_account_balance", "transaction_amount_inr"]

# Buat pola missing sebagai string key
df["_miss_pattern"] = ""
for col in cols_with_missing:
    df["_miss_pattern"] += df[col].isnull().astype(str)

patterns = df["_miss_pattern"].value_counts()
print(f"Jumlah pola missing unik: {len(patterns)}")
print(f"Distribusi pola:")
for pattern, count in patterns.head(10).items():
    pct = count / len(df) * 100
    readable = []
    for i, col in enumerate(cols_with_missing):
        if pattern[i*5:(i+1)*5].startswith("T"):
            readable.append(col)
    label = ", ".join(readable) if readable else "COMPLETE"
    print(f"  {label}: {count:,} ({pct:.2f}%)")

# Test: apakah mean berbeda antar pola
complete_mask = ~df[cols_with_missing].isnull().any(axis=1)
for num_col in numeric_for_test:
    vals_complete = df.loc[complete_mask, num_col].dropna()
    vals_incomplete = df.loc[~complete_mask, num_col].dropna()
    
    if len(vals_incomplete) < 2:
        continue
    
    t_stat, p_value = stats.ttest_ind(vals_complete, vals_incomplete, equal_var=False)
    sig = "*** SIGNIFIKAN (bukan MCAR)" if p_value < 0.05 else "tidak signifikan (mendukung MCAR)"
    
    print(f"\n  [{num_col}] Complete vs Incomplete cases:")
    print(f"    Mean complete: {vals_complete.mean():,.2f}")
    print(f"    Mean incomplete: {vals_incomplete.mean():,.2f}")
    print(f"    t={t_stat:.3f}, p={p_value:.6f} → {sig}")

# Cleanup
df = df.drop(columns=["_miss_pattern"])

# ══════════════════════════════════════════════════════════════════════════
# KESIMPULAN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("KESIMPULAN & REKOMENDASI")
print("=" * 70)
print("""
CARA MEMBACA HASIL:

┌─────────────────────────────────────────────────────────────────────┐
│ MCAR (Missing Completely at Random)                                │
│ • Missingness TIDAK tergantung variabel apapun (observed maupun    │
│   unobserved)                                                      │
│ • T-test: TIDAK signifikan                                         │
│ • Logistic R²: ≈ 0                                                 │
│ • Missing rate: MERATA di semua kategori                           │
│ • Aman di-drop ATAU di-impute                                      │
├─────────────────────────────────────────────────────────────────────┤
│ MAR (Missing at Random)                                            │
│ • Missingness tergantung variabel LAIN yang terobservasi           │
│ • T-test: SIGNIFIKAN pada beberapa variabel                        │
│ • Logistic R²: > 0 tapi tidak besar                                │
│ • Missing rate: TIDAK MERATA antar kategori                        │
│ • Bisa di-impute (multiple imputation lebih baik dari median)      │
│ • Drop juga masih aman jika jumlah kecil (<5%)                     │
├─────────────────────────────────────────────────────────────────────┤
│ MNAR (Missing Not at Random)                                       │
│ • Missingness tergantung NILAI variabel itu sendiri                │
│ • Contoh: orang dengan balance SANGAT tinggi sengaja tidak mengisi │
│ • TIDAK bisa dideteksi secara statistik dari data yang ada         │
│ • Butuh domain knowledge untuk mengidentifikasi                    │
│ • Drop maupun impute keduanya berpotensi bias                      │
└─────────────────────────────────────────────────────────────────────┘

PENTING: MNAR tidak bisa dibuktikan secara statistik — hanya bisa 
diduga berdasarkan domain knowledge. Yang bisa kita uji secara 
statistik hanyalah: MCAR vs non-MCAR (yaitu MAR/MNAR).

UNTUK KONTEKS BANKING:
  • customer_dob & cust_gender: KYC fields → JANGAN di-impute apapun
    hasilnya (regulatory constraint, bukan statistik)
  • cust_account_balance: Jika MCAR → drop aman. Jika MAR → impute 
    dengan median. Dalam kedua kasus, 0.23% missing rate sangat kecil
    sehingga drop tidak akan memperkenalkan bias signifikan.
  • cust_location: 0.01% — terlalu kecil untuk dianalisis, drop aman.
""")
