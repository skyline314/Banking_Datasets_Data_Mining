# Preprocessing & Pipeline Decision Log
## Bank Transactions Data Mining Project

> This document records every decision made during EDA, data cleaning,
> feature engineering, and ETL pipeline construction — including the
> reasoning, alternatives considered, and domain context for each choice.

---

## Table of Contents
1. [Data Loading & Customer Stats](#1-data-loading--customer-stats)
2. [Deduplication](#2-deduplication)
3. [Column Standardization](#3-column-standardization)
4. [Missing Value Treatment](#4-missing-value-treatment)
5. [Type Conversion](#5-type-conversion)
6. [DOB 2-Digit Year Fix](#6-dob-2-digit-year-fix)
7. [TransactionTime Parsing](#7-transactiontime-parsing)
8. [KYC Consistency — Drop vs Impute](#8-kyc-consistency--drop-vs-impute)
9. [Gender Filtering](#9-gender-filtering)
10. [Location Alias Merge](#10-location-alias-merge)
11. [Age Calculation Method](#11-age-calculation-method)
12. [Temporal Feature Extraction](#12-temporal-feature-extraction)
13. [Outlier Treatment](#13-outlier-treatment)
14. [Age Encoding — Ordinal](#14-age-encoding--ordinal)
15. [Gender Encoding — Binary](#15-gender-encoding--binary)
16. [Location Grouping — Top N + OTHER](#16-location-grouping--top-n--other)
17. [One-Hot Encoding with drop_first](#17-one-hot-encoding-with-drop_first)
18. [OHE dtype — bool to int](#18-ohe-dtype--bool-to-int)
19. [Normalization — Yeo-Johnson](#19-normalization--yeo-johnson)
20. [Normalization Scope — All Numerics](#20-normalization-scope--all-numerics)
21. [Normalization Timing — After Feature Assembly](#21-normalization-timing--after-feature-assembly)
22. [Feature Selection — Entropy & Variance](#22-feature-selection--entropy--variance)
23. [Feature Selection — Keep All Dimensions](#23-feature-selection--keep-all-dimensions)
24. [Correlation — Drop Redundant Features](#24-correlation--drop-redundant-features)
25. [customer_id Retention](#25-customer_id-retention)
26. [transaction_id Drop](#26-transaction_id-drop)
27. [Export — index=False](#27-export--indexfalse)
28. [Pipeline Architecture — ETL Separation](#28-pipeline-architecture--etl-separation)
29. [Config Centralization](#29-config-centralization)
30. [Load Validation Assertions](#30-load-validation-assertions)
31. [Dual Output — clean.csv + clean_categorical.csv](#31-dual-output--cleancsv--clean_categoricalcsv)

---

## 1. Data Loading & Customer Stats

**Decision:** Compute `customer_freq`, `avg_txn_amount`, and `total_txn_amount`
from the raw dataset immediately after loading, before any row drops.

**Justification:**
The cleaning pipeline removes ~29% of rows during the KYC consistency drop
(Step 8). Most removed rows belong to customers who appear multiple times
in the dataset — the very customers who would have `customer_freq > 1`.

If customer stats are computed after the drop, the remaining dataset is
dominated by single-transaction customers, making `customer_freq`
near-constant (99.75% = 1). This completely destroys its value as a
transaction rate feature for segmentation and mining.

Computing on raw data gives the true picture:
- 884,265 unique customers
- `customer_freq` range: 1–6, mean: 1.19
- This reflects real multi-transaction behavior, not a filtering artifact

**Alternative considered:** Compute after cleaning.
**Why rejected:** Produces near-constant feature — useless for transaction
rate mining, which is a core analytical angle of this project.

---

## 2. Deduplication

**Decision:** Drop exact duplicate rows using `drop_duplicates()` with no
subset restriction (all columns must match).

**Justification:**
Exact row-level duplicates in a transaction dataset indicate a data
ingestion or system replication error. A legitimate bank transaction
is uniquely identified by `TransactionID` — two rows that are identical
in every column cannot represent two real transactions.

Result: 0 duplicates found in this dataset, confirming clean ingestion.

**Alternative considered:** Dedup on `TransactionID` only.
**Why not primary:** Deduping on all columns is safer — it catches cases
where the ID was reused but data differs, which would require a separate
investigation rather than silent removal.

---

## 3. Column Standardization

**Decision:** Convert all column names to `snake_case` using regex-based
camelCase splitting and whitespace replacement.

**Justification:**
The raw dataset uses inconsistent naming conventions (`TransactionID`,
`CustGender`, `TransactionAmount (INR)`). Standardizing to `snake_case`
ensures consistent programmatic access throughout the pipeline and avoids
errors from spaces or mixed casing in column references.

The transformation is deterministic and fully logged — every rename is
printed so the mapping is auditable.

---

## 4. Missing Value Treatment

**Decision:** Drop all rows with any null value using `dropna()`.

**Justification:**
Missing values are present in 4 columns:
- `customer_dob`: 3,397 nulls (0.32%)
- `cust_gender`: 1,100 nulls (0.10%)
- `cust_location`: 151 nulls (0.01%)
- `cust_account_balance`: 2,369 nulls (0.23%)

The missing rate is below 0.5% for all columns. At this scale
(1,048,567 rows), dropping ~6,953 rows is a negligible loss (<0.7%)
that avoids the risk of imputing demographically sensitive fields
incorrectly.

Additionally, `cust_gender` and `customer_dob` are KYC fields (see Step 8)
— imputation is not permitted for these in a banking context regardless
of the missing rate.

**Alternative considered:** Column-specific imputation (mode for gender,
median for balance).
**Why rejected:** Gender and DOB cannot be imputed (KYC constraint).
Balance imputation would introduce artificial values into a financial
feature. Given the negligible row loss, dropping is more defensible.

---

## 5. Type Conversion

**Decision:** Convert date strings to `datetime`, HHMMSS integer to
`time` object, ID columns to `str`, and numeric columns to `float64`.

**Justification:**
The raw data stores all fields as mixed types on load:
- `TransactionDate` and `CustomerDOB` arrive as strings — datetime
  conversion is required for any date arithmetic (age calculation,
  temporal feature extraction)
- `TransactionTime` arrives as an integer (e.g. `143207` = 14:32:07)
  — must be parsed to extract the hour component for time-of-day analysis
- ID columns are kept as strings to prevent any accidental numeric
  operations on identifier fields

`dayfirst=True` is specified for date parsing because the dataset uses
DD/MM/YYYY format — without this, January 3 (01/03) would be parsed as
March 1 (03/01).

---

## 6. DOB 2-Digit Year Fix

**Decision:** Subtract 100 years from any `CustomerDOB` that parses as a
future date.

**Justification:**
Python's `pd.to_datetime` with a 2-digit year component defaults to the
21st century (e.g. `01/01/96` → `2096-01-01`). This creates 117,080
customers with birth years in 2024–2099, which is clearly impossible.

These are actually customers born in 1924–1999. The fix subtracts 100
years from all future-dated DOBs, correctly restoring the intended century.

**Why not `strptime` with `%y`:** Python's `%y` directive uses a cutoff
year of 68 — years 00–68 map to 2000–2068, years 69–99 map to 1969–1999.
This would still misclassify customers born before 1969, so an explicit
post-hoc correction is more reliable.

---

## 7. TransactionTime Parsing

**Decision:** Extract `hour` directly from the HHMMSS integer before
converting to a `time` object. Store the clean `time` object without the
`1900-01-01` date prefix.

**Justification:**
When converting a time-only string to `datetime`, pandas has no date
to attach — it defaults to `1900-01-01` as a placeholder. This is
cosmetically confusing but not a bug in the data.

The `1900-01-01` prefix is eliminated by using `.time()` to extract
only the time component, giving clean values like `14:32:07`.

Hour is extracted first (while the column is still an integer or datetime)
because `.dt.hour` is not available on `time` objects — extraction order
matters.

`transaction_time` is dropped before export since `transaction_time_category`
(Morning, Evening, etc.) encodes all necessary information in a form
suitable for OHE.

---

## 8. KYC Consistency — Drop vs Impute

**Decision:** Drop all transactions belonging to any `CustomerID` that
has more than one unique `CustomerDOB` or `CustGender` across their rows.
No imputation is applied.

**Justification:**
`CustomerDOB` and `CustGender` are KYC (Know Your Customer) fields.
In banking, these values are collected at account opening from legal
identity documents (passport, national ID, etc.) and are verified by
the institution. They are system-of-record data.

If a customer appears with two different DOBs or genders across
transactions, this indicates one of:
- Identity corruption (two people merged under one ID)
- System migration error
- Potential fraud or synthetic identity

In any of these cases, choosing which value to believe using a
statistical method (mode, median) is not appropriate — the bank cannot
overwrite a legally verified field with a frequency estimate.

**Why not mode imputation:**
Picking the most frequent DOB does not mean it is the correct DOB.
For a customer with 2 transactions, one with each DOB, mode is arbitrary.
This could result in misidentifying a real person's age, affecting
segmentation results and potentially regulatory reporting.

**Data loss documented:**
- 140,711 customers with inconsistent records
- 301,715 rows removed (29.0% of data)
- Mining conclusions reflect customers with clean, verified identity only

**Alternative considered:** Mode imputation per customer.
**Why rejected:** KYC fields are not eligible for statistical imputation
in banking. Regulatory and compliance standards (KYC/AML) require that
demographic data match original documentation.

---

## 9. Gender Filtering

**Decision:** Retain only rows where `CustGender` is `'M'` or `'F'`.
Remove 1 row with value `'T'`.

**Justification:**
The single `'T'` entry is negligible at 0.0001% of data. It likely
represents a data entry error rather than an intentional third-gender
classification — the dataset has no other evidence of non-binary
gender recording. Retaining it would require a third encoding category
for 1 row, adding noise without analytical value.

---

## 10. Location Alias Merge

**Decision:** Replace `'NEW DELHI'` with `'DELHI'` before any location
grouping or OHE. This step runs in Step 8 of the pipeline — before
`top_locations` is computed and before OHE is applied.

**Justification:**
`'DELHI'` and `'NEW DELHI'` refer to the same metropolitan area. They
appear as separate values due to inconsistent data entry. Combined,
they represent ~155K transactions in the raw data — making Delhi the
largest single city in the dataset.

If not merged before OHE, they produce two separate binary columns
(`loc_DELHI`, `loc_NEW DELHI`) that split this signal. Any model or
clustering algorithm would treat them as distinct geographic locations,
which is factually incorrect and dilutes the Delhi signal across two
dimensions.

**Order matters:** The merge must happen before `top_locations` is computed.
If done after, both names independently rank in the top 15 and receive
separate OHE columns regardless.

---

## 11. Age Calculation Method

**Decision:** Calculate age as `(transaction_date - customer_dob).dt.days // 365`
(elapsed days, integer division).

**Justification:**
The simpler method — `transaction_date.dt.year - customer_dob.dt.year` —
subtracts calendar year numbers only. It does not check whether the
birthday has occurred yet in the transaction year.

**Example of the error:**
A customer born on December 31, 1990 transacting on January 1, 2016
would be counted as age 26 by year subtraction, but is actually 25 years
old — the birthday has not occurred yet.

For broad age buckets (18–24, 25–34, etc.) this causes misclassification
at boundary ages. A 24.9-year-old classified as 25 would be placed in a
different segment, distorting age-based mining rules.

Elapsed days avoids this by computing actual time elapsed regardless of
the calendar year boundary.

**Note on leap years:** `// 365` slightly overestimates age for people
born before a cluster of leap years. For the purposes of ordinal age
bucketing (not precise actuarial calculation), this is an acceptable
approximation.

---

## 12. Temporal Feature Extraction

**Decision:** Extract `transaction_day_of_week`, `transaction_month`,
`is_weekend`, and `transaction_time_category` before dropping
`transaction_date`.

**Justification:**
`transaction_date` is dropped before export because raw date values
are not directly usable in distance-based mining algorithms. However,
the temporal patterns within dates are essential for the mining angle:

- `is_weekend`: Captures behavioral difference between weekday and
  weekend transactions — a key segmentation dimension
- `transaction_month`: Captures seasonal spending patterns
- `transaction_time_category`: Captures time-of-day behavior
  (Morning, Evening, Night, etc.)
- `transaction_day_of_week`: Extracted but later found correlated
  with `is_weekend` (r=0.78) — dropped in Step 24

If extraction happened after the drop, these features would be
permanently lost. Temporal features must be computed while the source
column still exists.

---

## 13. Outlier Treatment

**Decision:** Retain all outliers. Apply Yeo-Johnson normalization to
compress extreme values.

**Justification:**
IQR analysis identified:
- 13.4% of account balances above upper fence (₹137K)
- 10.7% of transaction amounts above upper fence (₹2,759)
- Maximum balance: ₹115M (vs median ₹16.8K)

**Why outliers are retained:**
1. This is unsupervised data mining, not regression — there is no
   target variable to distort. Outliers do not inflate prediction error.
2. High-balance customers (HNI/corporate accounts) are a meaningful
   and analytically interesting segment, not noise. Removing them
   would make it impossible to identify and characterize this group.
3. Yeo-Johnson normalization compresses the scale of extreme values
   adequately for distance-based algorithms without removing the
   underlying information.
4. In banking, extreme account balances are expected — they represent
   corporate or high-net-worth customers and are a legitimate segment
   for targeted analysis.

**Alternative considered:** Winsorize or cap at 99th percentile.
**Why rejected:** Capping arbitrarily merges all values above the cap
into a single point, losing the distinction between a ₹500K balance
and a ₹115M balance. For clustering, this collapses the HNI segment.

---

## 14. Age Encoding — Ordinal

**Decision:** Encode age as an ordinal integer (`<18`=0, `18-24`=1, ...,
`65+`=6) rather than one-hot encoding.

**Justification:**
Age is an inherently ordered variable — a 45-year-old is meaningfully
"between" a 35-year-old and a 55-year-old in terms of life stage,
spending behavior, and financial needs. Ordinal encoding preserves
this natural ordering.

One-hot encoding age would produce 7 binary columns and destroy the
ordering information, treating all age groups as equally distinct
categorical neighbors. This would harm clustering quality for any
age-based segmentation.

**Limitation acknowledged:** Ordinal encoding assumes equal intervals
between buckets, but age ranges are not equal width (`18-24` = 6 years,
`25-34` = 9 years, etc.). For distance-based clustering, this is a
known approximation. Yeo-Johnson normalization is applied to `age_ordinal`
to further compress the scale.

---

## 15. Gender Encoding — Binary

**Decision:** Encode `CustGender` as a binary integer (`M`=0, `F`=1).

**Justification:**
After filtering to `M` and `F` only (Step 9), gender is a binary
variable. Binary encoding with a single 0/1 column is equivalent to
OHE with `drop_first=True` for a 2-category variable — it preserves
all information without creating any redundant column.

---

## 16. Location Grouping — Top N + OTHER

**Decision:** Keep the top 15 cities as individual OHE columns.
All remaining cities are grouped into a single `'OTHER'` category.

**Justification:**
The dataset has a large number of unique city names (hundreds), with
a highly skewed distribution — a few cities dominate transaction volume
while hundreds of cities appear only a handful of times.

OHE-ing every city would produce hundreds of near-zero columns, most
of which have near-zero variance and near-zero entropy. This adds
dimensionality without signal and would dominate distance calculations
in clustering.

The `'OTHER'` bucket is critical — it does not discard the rare-city
transactions, it groups them into a meaningful "non-metro" segment.
The `loc_OTHER` column captures the behavioral pattern of customers
transacting outside the top 15 cities, which is itself a useful
segmentation dimension.

**Note:** The Delhi/New Delhi merge (Step 10) must happen before this
step so that Delhi is correctly counted as a single city in the top 15,
not split across two entries.

---

## 17. One-Hot Encoding with drop_first

**Decision:** Apply `pd.get_dummies(..., drop_first=True)` to both
`transaction_time_category` and `location_grouped`.

**Justification:**
Without `drop_first=True`, OHE produces one column per category. For
K groups, this creates a perfect multicollinearity — the K columns
always sum to 1, meaning any one column is a linear combination of
the others. This is the dummy variable trap.

For linear models (logistic regression, linear SVM, LDA), this makes
the coefficient matrix singular and the model unidentifiable.
For distance-based clustering (K-Means), it artificially inflates the
weight of the time and location groups in Euclidean distance — a
transaction's location is represented by 16 correlated dimensions
instead of 15 independent ones.

`drop_first=True` removes one reference category per group, eliminating
the collinearity while retaining all information (the dropped category
is implicitly encoded when all remaining columns are 0).

---

## 18. OHE dtype — bool to int

**Decision:** Cast all OHE columns from `bool` to `int` after
`pd.get_dummies()`.

**Justification:**
`pd.get_dummies()` returns boolean columns by default in recent pandas
versions. Several downstream libraries do not handle boolean inputs
gracefully:
- scikit-learn's SVM requires numeric input
- Some XGBoost versions throw dtype errors on bool arrays
- `np.histogram` (used in entropy calculation) raises a
  `DTypePromotionError` when encountering bool alongside float columns

Explicit casting to `int` (0/1) removes this ambiguity universally
and is a zero-information-loss operation.

---

## 19. Normalization — Yeo-Johnson

**Decision:** Use `PowerTransformer(method='yeo-johnson')` for all
numeric feature normalization.

**Justification:**
The numeric features (`cust_account_balance`, `transaction_amount_inr`,
etc.) are heavily right-skewed with extreme outliers. Standard scaling
(z-score) does not address skewness — it shifts and scales but preserves
the distributional shape. Log transformation cannot handle zero or
negative values.

Yeo-Johnson is chosen over log1p because:
1. It handles zero values natively (log1p also handles zeros, but
   Yeo-Johnson handles negative values too)
2. It finds the optimal transformation parameter (lambda) per feature,
   adapting to each column's specific distribution shape
3. It is designed for the full range of skewness patterns, not just
   right-skewed data

For K-Means and other distance-based algorithms, normalizing to
comparable scales is essential — without it, `cust_account_balance`
(range: 0–115M) would dominate Euclidean distances over `age_ordinal`
(range: 0–6).

---

## 20. Normalization Scope — All Numerics

**Decision:** Normalize all continuous numeric features including
`customer_freq`, `age_ordinal`, `transaction_month`, not just
balance and amount.

**Justification:**
The original EDA notebook only normalized `cust_account_balance` and
`transaction_amount_inr`. The remaining numeric features were left
on their raw scales:
- `customer_freq`: integer (1–6)
- `age_ordinal`: integer (0–6)
- `transaction_month`: integer (1–12)

In any distance-based algorithm (K-Means, DBSCAN, hierarchical
clustering), all features contribute to Euclidean distance proportionally
to their numeric range. A feature with range 0–12 (`transaction_month`)
will naturally contribute more to distance than a feature with range 0–6
(`age_ordinal`) even if both are equally important conceptually.

Normalizing all numerics ensures each feature contributes proportionally
to its information content, not its raw scale.

**OHE columns (0/1) and `is_weekend` (0/1) are excluded** — they are
already on unit scale and normalization would distort their binary meaning.
`gender_encoded` (0/1) is also excluded for the same reason.

---

## 21. Normalization Timing — After Feature Assembly

**Decision:** Run normalization in Step 15 of the pipeline — after
all features are assembled, encoded, and redundant columns are removed.

**Justification:**
The original EDA notebook normalized `cust_account_balance` and
`transaction_amount_inr` early, before feature engineering was complete.
This is incorrect for two reasons:

1. **Wasted computation:** Features like `avg_txn_amount` and
   `total_txn_amount` were normalized and then later found to be
   perfectly correlated and dropped. Normalizing them was wasted work.

2. **Incorrect feature selection input:** Entropy-based feature selection
   was run on the normalized versions of only 4 columns, while OHE and
   other features were never evaluated. Feature selection should run on
   the complete, assembled feature set.

Normalizing last ensures only the final feature set is transformed,
and the transformer is fitted on the correct data.

---

## 22. Feature Selection — Entropy & Variance

**Decision:** Evaluate all features using Shannon entropy and variance
as selection criteria.

**Justification:**
- **Shannon entropy** measures information content — a feature with high
  entropy has more varied values and is more likely to contribute
  discriminating power to clustering or rule mining
- **Variance** measures spread — a near-zero variance feature is nearly
  constant and provides almost no basis for segment differentiation

Both metrics are computed on the full assembled feature set including
OHE columns — the original notebook only evaluated 4 numeric columns
and ignored the 20+ OHE columns entirely, leaving their information
content unassessed.

---

## 23. Feature Selection — Keep All Dimensions

**Decision:** Retain all features despite some showing low entropy
and low variance. No features dropped based on entropy/variance alone.

**Justification:**
In supervised ML, dropping weak features reduces overfitting.
In unsupervised data mining (clustering, Apriori, segmentation),
dropping a feature removes an entire analytical dimension — location,
time-of-day, transaction rate — from all downstream analysis.

The mining angle of this project explicitly includes:
- Transaction rate analysis → requires `customer_freq_normalized`
- Location-based patterns → requires all `loc_` columns
- Time-of-day behavior → requires all `time_` columns

Dropping any of these groups would make the corresponding analysis
impossible, not just weaker. The cost of removing an analytical
dimension outweighs the benefit of a slightly leaner feature set.

`loc_OTHER` already handles all cities not in the top 15 — dropping
any top-15 city column would silently remove that city from all
analysis, defeating the purpose of the grouping.

**Entropy/variance analysis conclusion:** Used for interpretive context
(understanding which features will drive strong vs weak segment
differentiation), not as a drop threshold.

---

## 24. Correlation — Drop Redundant Features

**Decision:** Drop `avg_txn_amount`, `total_txn_amount`, and
`transaction_day_of_week` based on correlation heatmap analysis.

**Findings:**
- `transaction_amount_inr` vs `avg_txn_amount`: r = 1.00
- `transaction_amount_inr` vs `total_txn_amount`: r = 1.00
- `transaction_day_of_week` vs `is_weekend`: r = 0.78

**Justification:**

**Perfect collinearity (r=1.00):** After the KYC consistency drop,
most remaining customers have exactly 1 transaction. For a customer
with 1 transaction, their average amount equals their total amount
equals their single transaction amount. All three columns encode
identical information. Including all three in clustering triple-counts
the transaction amount dimension, pulling centroids disproportionately
toward amount differences.

`transaction_amount_inr` is kept as the canonical representative —
it represents the actual transaction value without the statistical
aggregation framing that becomes meaningless at freq=1.

**High correlation (r=0.78):** `is_weekend` is derived directly from
`transaction_day_of_week` (days 5 and 6 → is_weekend=1). They are
not perfectly collinear because `transaction_day_of_week` also
encodes Monday–Friday distinctions, but 0.78 is sufficient to cause
meaningful double-counting of the weekend signal.

`is_weekend` is kept because it is more interpretable (binary
behavioral split) and more directly aligned with the mining angle
(weekend vs weekday transaction behavior). `transaction_day_of_week`
is dropped because its additional Mon–Fri granularity is not a
stated mining objective and adds correlated noise.

---

## 25. customer_id Retention

**Decision:** Keep `customer_id` in `clean.csv` as a non-feature
joinable key. It is excluded from all normalization and mining
algorithm inputs.

**Justification:**
Without `customer_id`, the clean dataset is a flat table of features
with no way to trace a row back to a specific customer, join with
external data, or reconstruct customer-level summaries after mining.

For data mining workflows:
- After clustering, segments need to be profiled — this requires
  joining cluster assignments back to the original customer record
- For Apriori/association rule mining, `customer_id` is needed to
  group transactions per customer (market basket)
- For anomaly detection, flagged rows must be traceable to a customer

`customer_id` is an identifier, not a feature — it is excluded from
`X` (the feature matrix) at the moment mining algorithms are called,
not during preprocessing.

---

## 26. transaction_id Drop

**Decision:** Drop `transaction_id` before export.

**Justification:**
`transaction_id` is a unique row identifier with no predictive or
descriptive value. In clustering, a unique ID creates 699,261
distinct categories — it contributes no grouping signal and would
consume a dimension of the feature space. In Apriori, it cannot
form meaningful association rules because each value appears exactly
once.

Unlike `customer_id`, `transaction_id` does not serve as a useful
join key for customer-level analysis — `customer_id` already serves
that purpose.

---

## 27. Export — index=False

**Decision:** Call `df.to_csv(path, index=False)` — never `to_csv(path)`.

**Justification:**
`pd.DataFrame.to_csv()` writes the DataFrame's integer index as a
column by default. When the file is read back, this produces an
`Unnamed: 0` column containing values 0, 1, 2, 3... — pure noise
that is numerically structured enough to correlate spuriously with
other features in some algorithms.

`index=False` suppresses this. A validation assertion in the load
step explicitly checks for `Unnamed: 0` and fails the pipeline if
found, preventing silent export of corrupt files.

---

## 28. Pipeline Architecture — ETL Separation

**Decision:** Separate the pipeline into distinct `extract.py`,
`transform.py`, and `load.py` modules orchestrated by `pipeline.py`.

**Justification:**
Monolithic notebooks (like the original `EDA.ipynb`) mix data loading,
cleaning, feature engineering, and export in a single sequential file.
This creates several problems:
- No reusability — impossible to re-run just the transform step
- No testability — individual functions cannot be unit tested
- No auditability — no structured logging of what happened and why
- Fragile ordering — step 3 accidentally runs before step 2 with no
  enforcement

ETL separation enforces a clean execution model:
- **Extract:** reads data, computes customer stats on raw — no side effects
- **Transform:** pure data transformation — input df, output clean df
- **Load:** validates then writes — no transformation logic

Each module has a single responsibility and can be developed,
tested, and modified independently.

---

## 29. Config Centralization

**Decision:** Store all constants, paths, feature lists, and
parameters in a single `config/config.py` file.

**Justification:**
Without centralization, values like `TOP_N_LOCATIONS = 15` or
`AGE_BINS = [0, 18, 25, 35, 45, 55, 65, 100]` are scattered across
multiple notebook cells and pipeline modules. Changing one value
requires finding and updating every occurrence — a common source of
inconsistency bugs.

With `config.py`:
- `TOP_N_LOCATIONS` is defined once — `transform.py` and any future
  analysis notebook import it from the same source
- All paths (`RAW_DATA_PATH`, `PROCESSED_DATA_PATH`) are absolute and
  computed relative to `ROOT_DIR` — no hardcoded paths that break
  when the project is moved
- The full list of decisions (what to normalize, what to drop, what
  aliases to apply) is visible in one place for review and audit

---

## 30. Load Validation Assertions

**Decision:** Run hard assertions before writing `clean.csv`. Fail
loudly if any assertion fails — do not write a partial or corrupt file.

**Assertions checked:**
1. No `Unnamed: 0` ghost column
2. `customer_id` is present
3. No bool dtype OHE columns
4. No null values in any column
5. No non-numeric columns (except `customer_id`)

**Justification:**
Silent failures are the most dangerous kind in a data pipeline.
If a corrupt file is written without error, downstream mining
algorithms silently train on bad data — producing results that
appear valid but are based on flawed inputs.

Hard assertions at the load step ensure:
- The pipeline fails immediately with a clear error message
- No file is written until all checks pass
- The error log points to exactly which check failed
- The developer is forced to fix the root cause before proceeding

This is a "fail fast" design pattern — the cost of a pipeline
failure is much lower than the cost of discovering a data quality
issue after running a 2-hour clustering job on corrupt data.

---

## 31. Dual Output — clean.csv + clean_categorical.csv

**Decision:** Produce two CSV files from Stage 1:
- `clean.csv` — normalised and encoded features (for clustering / distance-based ML)
- `clean_categorical.csv` — cleaned data with original-scale values (for ARM / EDA)

**Justification:**
`clean.csv` contains Yeo-Johnson normalised floats and OHE-encoded columns.
This format is required for distance-based algorithms (K-Means, DBSCAN) but is
**fundamentally incompatible** with Association Rule Mining (ARM), which requires
domain-meaningful categorical items like `balance_group=High (100K-500K)`.

Discretizing normalised values (e.g. `-0.787`) into rupee-based bins is
meaningless because Yeo-Johnson is a non-linear power transform — the mapping
between normalised values and original rupee values is not recoverable without
the fitted transformer object.

Previously, Stage 3 worked around this by re-reading the raw CSV and re-running
all cleaning steps. While functionally correct (the cleaning functions were
imported from Stage 1 with no code duplication), this approach:
- Wasted compute by re-processing ~1M rows of cleaning
- Created a hidden dependency on the raw data file for Stage 3
- Violated single-source-of-truth — the cleaned dataset should be the
  canonical output of Stage 1, not re-derived by each consumer

**Implementation:**
The `transform()` function was refactored into:
- `transform_base()` — cleaning steps 1–11 (dedup through customer stats merge)
- `transform()` — calls `transform_base()` + encoding + normalization

The pipeline runs `transform_base()` once, exports `clean_categorical.csv`
from the result, then continues with encoding/normalization to produce
`clean.csv`. Both files contain the same cleaned rows — the difference is
only in post-cleaning processing.

**Columns in `clean_categorical.csv`:**
`customer_id`, `cust_gender`, `cust_location`, `cust_account_balance`,
`transaction_amount_inr`, `age`, `transaction_month`, `is_weekend`,
`transaction_time_category`, `customer_freq`, `avg_txn_amount`,
`total_txn_amount`

**Alternative considered:** Keep the raw-data re-processing approach.
**Why rejected:** Violates ETL single-source-of-truth principle. Two consumers
reading the same raw file and independently cleaning it could theoretically
diverge if cleaning logic changes in one place but not the other.

---

---

## 32. Missing Value Mechanism — MCAR Confirmed

**Decision:** Apply `dropna()` (listwise deletion) across all columns.
No per-column imputation.

**Formal analysis performed** (`scratch/missing_mechanism_analysis.py`):

| Test | Result |
|------|--------|
| Logistic regression pseudo-R² (predict missingness from `transaction_amount_inr`) | ≈ 0.000 for all columns |
| T-test: balance mean in missing vs non-missing groups | Not significant (p > 0.05) |
| Missing rate per gender (M / F) | Uniform across categories |
| Missing rate per top-10 locations | Uniform across cities |
| Co-occurrence correlation between missing indicators | Weak (r < 0.1) |

All tests are consistent with **MCAR (Missing Completely at Random)**: the
probability of a value being missing is independent of any observed or unobserved
variable. Under MCAR, listwise deletion is statistically unbiased.

**Why not impute `cust_account_balance` (0.23% missing)?**
- Under MCAR, drop and impute both produce unbiased estimates — the choice is
  indifferent. Dropping is simpler and avoids introducing artificial values into
  a financial feature.
- Total row loss < 0.7% (< 7,000 rows from 1,048,567).

**KYC constraint** (independent of mechanism): `customer_dob` and `cust_gender`
cannot be imputed regardless of missing mechanism — banking regulatory requirement.

**Alternative considered:** Per-column treatment (impute balance with median, drop KYC nulls).
**Why rejected:** MCAR makes the approaches statistically equivalent. Blanket drop is simpler and the row loss is negligible.

---

## 33. Outlier Assessment — Reproducible Code in Pipeline

**Decision:** Implement `assess_outliers()` as a pipeline step that logs
IQR statistics to the run log. No rows are removed.

**Why added:** The original DECISIONS.MD (#13) claimed IQR statistics
("13.4% balance outliers, 10.7% amount outliers") without any pipeline code
to reproduce those numbers. This was a documentation-only claim.

`assess_outliers()` computes and logs IQR fences, counts below/above fence,
percentage, min/max, mean, and skewness for `cust_account_balance` and
`transaction_amount_inr` on every pipeline run. The decision to **retain**
outliers (DECISIONS.MD #13) is unchanged — only the reproducibility gap
is closed.

**When called:** After `convert_types()` (data is numeric) and before
`drop_inconsistent_kyc()` (full row count still present for accurate
percentage calculation).

---

## 34. Cyclical Encoding for `transaction_month`

**Decision:** Replace Yeo-Johnson normalization of `transaction_month` (1–12)
with sin/cos cyclical encoding: `month_sin = sin(2π·month/12)`,
`month_cos = cos(2π·month/12)`.

**Justification:**
`transaction_month` is a **cyclical variable** — December (month 12) and
January (month 1) are adjacent months, not maximally distant ones. Applying
Yeo-Johnson (a monotonic power transform on the range 1–12) places December
as far as possible from January in Euclidean space. This distorts K-Means
cluster boundaries along the temporal dimension.

Sin/cos encoding guarantees:
- `distance(December, January) == distance(January, February)` for all months
- Both components lie in [−1, 1] — compatible with Yeo-Johnson-normalized
  features without additional scaling
- Month identity is fully recoverable from the (sin, cos) pair

`transaction_month` is added to `COLS_TO_DROP` and removed from
`COLS_TO_NORMALIZE` since it is replaced by the two cyclical columns before
the drop step.

**Stage 3 (ARM) is unaffected:** `clean_categorical.csv` is produced from
`transform_base()` which preserves the original `transaction_month` integer
for season mapping in Stage 3 discretization.

**Alternative considered:** Map month → season (Winter/Spring/Summer/Autumn)
then OHE (same as Stage 3 treatment).
**Why rejected:** Season mapping collapses 3 months into one bucket, losing
within-season granularity. Sin/cos preserves all monthly information with
correct cyclical topology.

---

## 35. `age_ordinal` — Yeo-Johnson on 7-Level Ordinal

**Decision:** Retain Yeo-Johnson normalization for `age_ordinal` (7 discrete
levels: 0–6). No change to this step.

**Concern addressed:** Some sources suggest power transforms are unusual on
ordinal variables with few unique values.

**Why it is acceptable here:**
1. **Monotonicity preserved:** Yeo-Johnson is a monotonic transform — ordinal
   ordering (0 < 1 < 2 < ... < 6) is fully preserved after transformation.
   No two levels swap rank.
2. **Lambda estimation is stable:** The dataset has ~700K rows distributed
   across all 7 levels. Lambda is estimated from the empirical distribution
   of 700K values, not from 7 values — the estimate is stable.
3. **Purpose is scale equalization, not distributional normality:** The goal
   is to put `age_ordinal` (range 0–6) on a comparable scale with
   `cust_account_balance` (range 0–115M). Any monotonic scaling achieves
   this. Yeo-Johnson is consistent with the transform already applied to
   all other numeric features.

**Alternative considered:** Exclude `age_ordinal` from `COLS_TO_NORMALIZE`
and let Stage 2 StandardScaler handle it.
**Why rejected:** Stage 2 StandardScaler is applied on top of already
Yeo-Johnson-transformed features — mixing un-Yeo-Johnson'd ordinal with
power-transformed continuous features would create an inconsistent input
for the StandardScaler (different distributional shapes per feature).

---

## 36. Feature Information Assessment — Pipeline Code

**Decision:** Implement `assess_feature_information()` as a pipeline step
that logs Shannon entropy and variance for all features after encoding,
before normalization. No features are dropped.

**Why added:** DECISIONS.MD #22 discussed Shannon entropy analysis but the
pipeline contained no code implementing it — the analysis existed only in
notebooks and was therefore not reproducible from the pipeline alone.

`assess_feature_information()` computes:
- **Variance** per column
- **Shannon entropy** (bits) — continuous features binned into 20 bins;
  discrete features use value counts directly

The decision to **retain all features** (DECISIONS.MD #23) is unchanged —
this step closes the reproducibility gap and provides logged evidence for
that decision on every pipeline run.

**When called:** After `apply_ohe_and_drop()` (full encoded feature set is
assembled) and before `normalize_features()` (entropy computed on pre-transform
values for interpretability).

---

## 37. Descriptive Statistics Export

**Decision:** Export `descriptive_statistics.csv` and `categorical_summary.csv`
to `data/stage1_ETL/` as part of the pipeline run.

**Justification:**
The project specification requires "statistik deskriptif" as a Data
Understanding deliverable. While EDA notebooks contain this analysis, they
are not pipeline artifacts — they must be run manually and their outputs are
not version-controlled as part of the ETL output.

`export_descriptive_stats()` is called on `base_df` (cleaned, original-scale
data) immediately after `transform_base()`. This ensures:
- Statistics reflect post-cleaning data (deduped, KYC-dropped, etc.)
- Values are in interpretable original units (₹, years, counts) — not
  Yeo-Johnson space
- Output is produced automatically on every pipeline run

**`descriptive_statistics.csv`** columns: count, mean, std, min, 25%, 50%,
75%, max, skewness, kurtosis, null_count, null_pct — one row per numeric column.

**`categorical_summary.csv`** columns: column, unique_values, top_value,
top_freq, null_count — one row per string/categorical column.

---

*Document version: 1.2 — added decisions #32–37 (missing mechanism MCAR,
outlier code, cyclical month, age_ordinal justification, entropy code,
descriptive stats export).*