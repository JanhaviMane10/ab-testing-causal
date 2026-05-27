# ============================================================
# config.py — Central settings for the A/B Testing project
# ============================================================

import os

# --- Paths ---
DATA_DIR    = "data"
OUTPUT_DIR  = "outputs"
DB_PATH     = "outputs/hillstrom.db"

for folder in [DATA_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# --- Data File ---
DATA_FILE = os.path.join(DATA_DIR, "hillstrom.csv")

# --- Experiment Settings ---
# The Hillstrom dataset has 3 groups:
# 0 = No email (control)
# 1 = Men's merchandise email (treatment 1)
# 2 = Women's merchandise email (treatment 2)
CONTROL_GROUP   = "No E-Mail"
TREATMENT_GROUP = "Mens E-Mail"   # We compare Men's email vs No email

# --- Statistical Settings ---
ALPHA           = 0.05    # Significance level (95% confidence)
RANDOM_STATE    = 42

# --- Business Settings ---
EMAIL_COST      = 0.10    # Cost per email sent ($)
AVG_ORDER_VALUE = 50.0    # Average order value ($)
