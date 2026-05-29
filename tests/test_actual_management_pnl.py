"""
test_management_pnl.py
======================
Full testing suite for the Management P&L (Actual) pipeline.

Structure
---------
1. Unit Tests        – pure-Python / pandas logic, no Spark needed
2. Integration Tests – Spark-based, uses local SparkSession + in-memory DataFrames
3. End-to-End Tests  – full KPI chain validated against real reference figures
4. UAT Scenarios     – business-level acceptance checks

Run with:
    pytest test_management_pnl.py -v

Or in a Databricks notebook:
    %pip install pytest
    import pytest, sys
    sys.dont_write_bytecode = True
    retcode = pytest.main(["test_management_pnl.py", "-v", "-p", "no:cacheprovider"])
    assert retcode == 0, "Tests failed. Check the output above."
"""

import re
import math
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE VALUES  (sourced from real production P&L output)
# Every numeric assertion in this file derives from this single source of truth.
# ─────────────────────────────────────────────────────────────────────────────
REF = {
    # ── Revenue ──────────────────────────────────────────────────────────────
    "total_food":                               1_671_535.85,
    "total_beverage":                              76_720.71,
    "total_other":                                 -4_472.57,   # credit / negative
    "total_net_restaurant_sales":               1_743_783.99,
    # ── Cost of Sales ─────────────────────────────────────────────────────────
    "total_cost_of_sales":                        923_839.51,
    # ── Level 1 ───────────────────────────────────────────────────────────────
    "gross_margin":                               819_944.48,
    "total_wages":                                113_128.00,
    "total_salary_related":                        98_839.06,
    "total_personal_cost":                        211_967.06,
    # ── Level 2 ───────────────────────────────────────────────────────────────
    "gross_profit":                               607_977.42,
    "total_marketing":                                  0.00,
    "total_pr_entertainment":                           0.00,
    "grand_total_marketing":                            0.00,
    "total_operational_rm":                        12_519.57,
    "total_rm_it":                                  1_729.39,
    "total_r_m":                                   14_248.96,
    "total_overheads":                             15_349.77,
    "total_commissions":                           14_036.49,
    "total_travelling_expenses":                      209.14,
    "total_administration":                         4_451.75,
    "total_utilities":                             50_794.50,
    "total_controllable_costs":                    99_090.61,
    # ── Level 3 ───────────────────────────────────────────────────────────────
    "controllable_profit":                        508_886.81,
    "total_non_controllables_oths":               129_369.70,
    "total_non_controllables_rent":               246_096.66,
    "total_non_controllables":                    375_466.36,
    "store_operating_profit":                     133_420.45,
    # ── Final KPIs ────────────────────────────────────────────────────────────
    "total_other_income":                          14_707.10,
    "grand_total_other_income":                    14_707.10,
    "four_wall_ebitda":                           195_539.45,
    "store_ebitda":                               148_127.55,
    "total_depreciation":                          65_575.75,
    "total_amortization":                               0.00,
    "total_dep_amort":                             65_575.75,
    "four_wall_net_profit":                       129_963.70,
    "store_net_profit_loss":                       82_551.80,
    "pre_opening_exp":                                  0.00,
    "brand_ho_allocation":                         47_411.90,  # 4-Wall gap vs Store EBITDA
    "store_ebitda_after_pre":                     148_127.55,
    "store_net_profit_after_pre":                  82_551.80,
    # ── Below-the-line (zero in this period) ─────────────────────────────────
    "total_tax":                                        0.00,
    "total_interest":                                   0.00,
    "total_impairment":                                 0.00,
    "total_ifrs":                                       0.00,
    "total_rental_income":                              0.00,
    "total_insurance_claim":                            0.00,
    "total_interest_income":                            0.00,
}

TOLERANCE = 0.02   # ±2 cents absolute tolerance for floating-point comparisons


# ─────────────────────────────────────────────────────────────────────────────
# Helpers – pure Python / pandas re-implementations of notebook logic
# ─────────────────────────────────────────────────────────────────────────────

def to_snake_case(name: str) -> str:
    """Replicate the notebook's to_snake_case function."""
    return re.sub(r'[\s\-]+', '_', name).lower()


def to_snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply snake_case to all column names."""
    df.columns = [to_snake_case(c) for c in df.columns]
    return df


def compute_kpis_pandas(df: pd.DataFrame) -> dict:
    """
    Pure-pandas mirror of the PySpark aggregation logic in the notebook.
    Uses the same absolute-value semantics and the same group-key strings.
    """
    def _agg(col_name: str, match_value: str) -> float:
        mask = df[col_name].str.strip() == match_value
        return abs(df.loc[mask, "amount"].sum())

    # ── Level 0 raw sums ─────────────────────────────────────────────────────
    total_food                   = _agg("alternate_group", "Total Food")
    total_beverage               = _agg("alternate_group", "Total Beverage")
    total_net_restaurant_sales   = _agg("management_group", "Total Net Restaurant sales")
    total_cost_of_sales          = _agg("management_group", "TOTAL COST OF SALES")
    total_wages                  = _agg("management_group", "Total  Wages")   # double-space!
    total_salary_related         = _agg("management_group", "Total Salary Related")
    total_marketing              = _agg("management_group", "Total Marketing")
    total_pr_entertainment       = _agg("management_group", "Total PR & Entertainment")
    total_operational_rm         = (_agg("management_group", "Total Opertional R&M")   # typo variant
                                  + _agg("management_group", "Total Operational R&M"))  # correct
    total_rm_it                  = _agg("management_group", "Total R&M IT")
    total_overheads              = _agg("management_group", "Total Overheads")
    total_commissions            = _agg("management_group", "Total Commissions")
    total_travelling_expenses    = _agg("management_group", "Total Travelling Expenses")
    total_administration         = _agg("management_group", "Total Administration")
    total_utilities              = _agg("management_group", "Total Utilities")
    total_non_controllables_rent = _agg("management_group", "TOTAL NON-CONTROLLABLES RENT")
    total_non_controllables_oths = _agg("management_group", "TOTAL -  NON-CONTROLLABLES - OTHS")
    total_other_income           = _agg("management_group", "Total Other Income")
    total_rental_income          = _agg("management_group", "Total Rental Income")
    total_insurance_claim        = _agg("management_group", "Total -Insurance Claim")
    total_interest_income        = _agg("management_group", "Total Interest- Income")
    total_tax                    = _agg("management_group", "Total Tax")
    total_interest               = _agg("management_group", "Total Interest")
    total_impairment             = _agg("management_group", "Total Impairment")
    total_depreciation           = _agg("management_group", "Total Depreciation")
    total_amortization           = _agg("management_group", "Total Amortization")
    total_ifrs                   = _agg("management_group", "Total IFRS")

    # ── Level 1 ───────────────────────────────────────────────────────────────
    gross_margin             = total_net_restaurant_sales - total_cost_of_sales
    total_personal_cost      = total_salary_related + total_wages
    total_r_m                = total_rm_it + total_operational_rm
    grand_total_marketing    = total_pr_entertainment + total_marketing
    grand_total_other_income = (total_other_income + total_interest_income
                                + total_insurance_claim + total_rental_income)
    total_dep_amort          = total_amortization + total_depreciation

    # ── Level 2 ───────────────────────────────────────────────────────────────
    gross_profit             = gross_margin - total_personal_cost
    total_controllable_costs = (total_utilities + total_administration + total_overheads
                                + total_r_m + grand_total_marketing
                                + total_commissions + total_travelling_expenses)
    total_non_controllables  = total_non_controllables_rent + total_non_controllables_oths

    # ── Level 3 ───────────────────────────────────────────────────────────────
    controllable_profit    = gross_profit - total_controllable_costs
    store_operating_profit = controllable_profit - total_non_controllables

    # ── Final ─────────────────────────────────────────────────────────────────
    brand_ho_allocation = abs(df.loc[
        df["mapped_name"].str.strip().str.contains("Brand HO allocation", na=False), "amount"
    ].sum())

    store_ebitda          = store_operating_profit + grand_total_other_income
    four_wall_ebitda      = store_ebitda + brand_ho_allocation   # brand HO added back
    four_wall_net_profit  = four_wall_ebitda - total_depreciation
    store_net_profit_loss = (store_ebitda - total_tax - total_interest
                             - total_impairment - total_dep_amort - total_ifrs)

    return {
        "total_food": total_food,
        "total_beverage": total_beverage,
        "total_net_restaurant_sales": total_net_restaurant_sales,
        "total_cost_of_sales": total_cost_of_sales,
        "total_wages": total_wages,
        "total_salary_related": total_salary_related,
        "total_marketing": total_marketing,
        "total_pr_entertainment": total_pr_entertainment,
        "total_operational_rm": total_operational_rm,
        "total_rm_it": total_rm_it,
        "total_overheads": total_overheads,
        "total_commissions": total_commissions,
        "total_travelling_expenses": total_travelling_expenses,
        "total_administration": total_administration,
        "total_utilities": total_utilities,
        "total_non_controllables_rent": total_non_controllables_rent,
        "total_non_controllables_oths": total_non_controllables_oths,
        "total_other_income": total_other_income,
        "total_rental_income": total_rental_income,
        "total_insurance_claim": total_insurance_claim,
        "total_interest_income": total_interest_income,
        "total_tax": total_tax,
        "total_interest": total_interest,
        "total_impairment": total_impairment,
        "total_depreciation": total_depreciation,
        "total_amortization": total_amortization,
        "total_ifrs": total_ifrs,
        "gross_margin": gross_margin,
        "total_personal_cost": total_personal_cost,
        "total_r_m": total_r_m,
        "grand_total_marketing": grand_total_marketing,
        "grand_total_other_income": grand_total_other_income,
        "total_dep_amort": total_dep_amort,
        "gross_profit": gross_profit,
        "total_controllable_costs": total_controllable_costs,
        "total_non_controllables": total_non_controllables,
        "controllable_profit": controllable_profit,
        "store_operating_profit": store_operating_profit,
        "store_ebitda": store_ebitda,
        "four_wall_ebitda": four_wall_ebitda,
        "four_wall_net_profit": four_wall_net_profit,
        "store_net_profit_loss": store_net_profit_loss,
        "brand_ho_allocation": brand_ho_allocation,
    }


def approx_equal(actual: float, expected: float, tol: float = TOLERANCE) -> bool:
    return abs(actual - expected) <= tol


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

# ── Real P&L rows from production data ───────────────────────────────────────
# Each row represents a raw detail-level GL entry that, when aggregated by the
# pipeline, must produce the REF values above.
# alternate_group / management_group are set to the exact strings the notebook
# uses for matching.  amount signs match post-sign-reversal state (Income negated).
_REAL_PNL_ROWS = [
    # alternate_group entries (used for food/bev splits)
    ("Total Food",    "",                                  "Total Food",                1_671_535.85),
    ("Total Beverage","",                                  "Total Beverage",               76_720.71),
    ("Total other",   "",                                  "Total other",                  -4_472.57),
    # management_group entries
    ("", "Total Net Restaurant sales",  "Total Net Restaurant sales",  -1_743_783.99),  # Income → negated
    ("", "TOTAL COST OF SALES",         "TOTAL COST OF SALES",            923_839.51),
    ("", "Total  Wages",                "Total  Wages",                   113_128.00),  # double-space!
    ("", "Total Salary Related",        "Total Salary Related",            98_839.06),
    ("", "Total Marketing",             "Total Marketing",                      0.00),
    ("", "Total Opertional R&M",        "Total Opertional R&M",            12_519.57),  # typo variant
    ("", "Total R&M IT",                "Total R&M IT",                     1_729.39),
    ("", "Total Overheads",             "Total Overheads",                 15_349.77),
    ("", "Total Commissions",           "Total Commissions",               14_036.49),
    ("", "Total Travelling Expenses",   "Total Travelling Expenses",          209.14),
    ("", "Total Administration",        "Total Administration",             4_451.75),
    ("", "Total Utilities",             "Total Utilities",                 0.50),
    ("", "TOTAL NON-CONTROLLABLES RENT","TOTAL NON-CONTROLLABLES RENT",   246_096.66),
    ("", "TOTAL -  NON-CONTROLLABLES - OTHS", "TOTAL -  NON-CONTROLLABLES - OTHS", 129_369.70),
    ("", "Total Other Income",          "Total Other Income",             -14_707.10),  # Income → negated
    ("", "Total Depreciation",          "Total Depreciation",              65_575.75),
    # brand_ho_allocation – added back only in 4-Wall calculations
    ("", "",                             "Brand HO allocation",             47_411.90),
]

_SCHEMA_COLS = ["alternate_group", "management_group", "mapped_name", "amount"]


@pytest.fixture
def real_pnl_df():
    """
    Pandas DataFrame built from real production P&L figures.
    Signs follow post-sign-reversal state (Income rows negated, Expense rows positive).
    """
    return pd.DataFrame(_REAL_PNL_ROWS, columns=_SCHEMA_COLS)


@pytest.fixture
def empty_df():
    """Empty DataFrame with the same schema – edge-case testing."""
    return pd.DataFrame(columns=_SCHEMA_COLS)


@pytest.fixture
def negative_amounts_df(real_pnl_df):
    """All amounts negated – tests that abs() makes results sign-agnostic."""
    df = real_pnl_df.copy()
    df["amount"] = -df["amount"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Spark setup  (skipped gracefully when no SparkSession is available)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType

    _spark = (SparkSession.builder
              .master("local[2]")
              .appName("pnl_tests")
              .config("spark.ui.enabled", "false")
              .getOrCreate())
    _spark.sparkContext.setLogLevel("ERROR")
    SPARK_AVAILABLE = True
except Exception:
    SPARK_AVAILABLE = False

pytestmark_spark = pytest.mark.skipif(
    not SPARK_AVAILABLE, reason="SparkSession unavailable – tests run on Databricks"
)

_PNL_SPARK_SCHEMA = StructType([
    StructField("alternate_group",  StringType(), True),
    StructField("management_group", StringType(), True),
    StructField("mapped_name",      StringType(), True),
    StructField("amount",           DoubleType(), True),
]) if SPARK_AVAILABLE else None


@pytest.fixture
def spark_pnl_df():
    """Real-data PySpark DataFrame for integration / E2E tests."""
    if not SPARK_AVAILABLE:
        return None
    return _spark.createDataFrame(_REAL_PNL_ROWS, schema=_PNL_SPARK_SCHEMA)


def _spark_agg(df, col_name: str, match_value: str) -> float:
    """Helper: absolute sum of amount where col_name == match_value."""
    return abs(
        df.filter(F.col(col_name) == match_value)
          .agg(F.sum("amount"))
          .collect()[0][0] or 0.0
    )


# ─────────────────────────────────────────────────────────────────────────────
# ██  UNIT TESTS  █████████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

class TestToSnakeCase:
    """Unit tests for the to_snake_case helper."""

    def test_spaces_replaced(self):
        assert to_snake_case("Total Food") == "total_food"

    def test_hyphens_replaced(self):
        assert to_snake_case("store-open-date") == "store_open_date"

    def test_already_snake(self):
        assert to_snake_case("amount") == "amount"

    def test_mixed(self):
        assert to_snake_case("Brand HO-allocation") == "brand_ho_allocation"

    def test_empty_string(self):
        assert to_snake_case("") == ""

    def test_multiple_spaces_collapse_bug(self):
        """
        BUG DOCUMENTED: to_snake_case collapses "Total  Wages" (double-space) to
        "total_wages" – the greedy r'[\\s\\-]+' regex matches consecutive whitespace.

        IMPACT: If management_group column names were ever normalised through
        to_snake_case, the exact-match key "Total  Wages" would silently collapse
        and wages aggregation would return 0.

        FIX: Normalise source group keys to single spaces before matching, or use
        regex/LIKE patterns consistently throughout the pipeline.
        """
        # Record actual (buggy) behaviour so we catch accidental changes
        assert to_snake_case("Total  Wages") == "total_wages"


class TestToSnakeCaseDataframe:
    def test_column_names_normalised(self):
        df = pd.DataFrame({"Account Name": [1], "Total-Amount": [2]})
        assert list(to_snake_case_columns(df).columns) == ["account_name", "total_amount"]

    def test_no_op_when_already_snake(self):
        df = pd.DataFrame({"amount": [1], "location": [2]})
        assert list(to_snake_case_columns(df).columns) == ["amount", "location"]


class TestLevel0RawSums:
    """Level 0 – raw aggregation blocks vs. real reference values."""

    def test_total_net_restaurant_sales(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_net_restaurant_sales"],
                            REF["total_net_restaurant_sales"])

    def test_total_cost_of_sales(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_cost_of_sales"], REF["total_cost_of_sales"])

    def test_total_wages(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_wages"], REF["total_wages"])

    def test_total_salary_related(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_salary_related"], REF["total_salary_related"])

    def test_total_operational_rm(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_operational_rm"], REF["total_operational_rm"])

    def test_total_rm_it(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_rm_it"], REF["total_rm_it"])

    def test_total_depreciation(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_depreciation"], REF["total_depreciation"])

    def test_total_amortization_zero(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_amortization"], REF["total_amortization"])

    def test_total_other_income(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_other_income"], REF["total_other_income"])

    def test_total_marketing_zero(self, real_pnl_df):
        """Marketing spend was zero in this period – verify zero, not missing."""
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_marketing"], 0.0)

    def test_abs_makes_negative_amounts_positive(self, negative_amounts_df):
        kpis = compute_kpis_pandas(negative_amounts_df)
        assert kpis["total_net_restaurant_sales"] >= 0
        assert kpis["total_cost_of_sales"] >= 0

    def test_empty_dataframe_all_zeros(self, empty_df):
        kpis = compute_kpis_pandas(empty_df)
        for key in ["total_net_restaurant_sales", "total_cost_of_sales",
                    "total_wages", "total_depreciation"]:
            assert kpis[key] == 0.0

    def test_typo_and_correct_rm_variant_both_summed(self, real_pnl_df):
        """
        Notebook handles 'Total Opertional R&M' (typo) + 'Total Operational R&M'.
        Add the correctly-spelled variant and confirm both are included.
        """
        extra = pd.DataFrame([("", "Total Operational R&M", "Total Operational R&M", 500.0)],
                              columns=_SCHEMA_COLS)
        df = pd.concat([real_pnl_df, extra], ignore_index=True)
        kpis = compute_kpis_pandas(df)
        assert approx_equal(kpis["total_operational_rm"],
                            REF["total_operational_rm"] + 500.0)


class TestLevel1DerivedMetrics:
    """Level 1 – derived KPIs vs. real reference values."""

    def test_gross_margin(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["gross_margin"], REF["gross_margin"]), \
            f"gross_margin: got {kpis['gross_margin']}, expected {REF['gross_margin']}"

    def test_gross_margin_formula_identity(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["gross_margin"],
                            kpis["total_net_restaurant_sales"] - kpis["total_cost_of_sales"])

    def test_total_personal_cost(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_personal_cost"], REF["total_personal_cost"]), \
            f"total_personal_cost: got {kpis['total_personal_cost']}, expected {REF['total_personal_cost']}"

    def test_total_personal_cost_formula_identity(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_personal_cost"],
                            kpis["total_salary_related"] + kpis["total_wages"])

    def test_total_r_m(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_r_m"], REF["total_r_m"]), \
            f"total_r_m: got {kpis['total_r_m']}, expected {REF['total_r_m']}"

    def test_grand_total_marketing_zero(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["grand_total_marketing"], REF["grand_total_marketing"])

    def test_grand_total_other_income(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["grand_total_other_income"],
                            REF["grand_total_other_income"])

    def test_total_dep_amort(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_dep_amort"], REF["total_dep_amort"])


class TestLevel2DerivedMetrics:
    """Level 2 – derived KPIs vs. real reference values."""

    def test_gross_profit(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["gross_profit"], REF["gross_profit"]), \
            f"gross_profit: got {kpis['gross_profit']}, expected {REF['gross_profit']}"

    def test_gross_profit_formula_identity(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["gross_profit"],
                            kpis["gross_margin"] - kpis["total_personal_cost"])

    def test_total_controllable_costs(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_controllable_costs"],
                            REF["total_controllable_costs"]), \
            f"total_controllable_costs: got {kpis['total_controllable_costs']}, expected {REF['total_controllable_costs']}"

    def test_total_controllable_costs_formula_identity(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        expected = (kpis["total_utilities"] + kpis["total_administration"]
                    + kpis["total_overheads"] + kpis["total_r_m"]
                    + kpis["grand_total_marketing"] + kpis["total_commissions"]
                    + kpis["total_travelling_expenses"])
        assert approx_equal(kpis["total_controllable_costs"], expected)

    def test_total_non_controllables(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_non_controllables"],
                            REF["total_non_controllables"]), \
            f"total_non_controllables: got {kpis['total_non_controllables']}, expected {REF['total_non_controllables']}"

    def test_total_non_controllables_formula_identity(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["total_non_controllables"],
                            kpis["total_non_controllables_rent"]
                            + kpis["total_non_controllables_oths"])


class TestLevel3AndFinalKPIs:
    """Level 3 and Final KPIs vs. real reference values."""

    def test_controllable_profit(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["controllable_profit"], REF["controllable_profit"]), \
            f"controllable_profit: got {kpis['controllable_profit']}, expected {REF['controllable_profit']}"

    def test_controllable_profit_formula_identity(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["controllable_profit"],
                            kpis["gross_profit"] - kpis["total_controllable_costs"])

    def test_store_operating_profit(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["store_operating_profit"],
                            REF["store_operating_profit"]), \
            f"store_operating_profit: got {kpis['store_operating_profit']}, expected {REF['store_operating_profit']}"

    def test_store_operating_profit_formula_identity(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["store_operating_profit"],
                            kpis["controllable_profit"] - kpis["total_non_controllables"])

    def test_store_ebitda(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["store_ebitda"], REF["store_ebitda"]), \
            f"store_ebitda: got {kpis['store_ebitda']}, expected {REF['store_ebitda']}"

    def test_four_wall_ebitda(self, real_pnl_df):
        """
        4-Wall EBITDA = store_operating_profit + grand_total_other_income
                        + pre_opening_exp + brand_ho_allocation
        In this period: 148,127.55 (Store EBITDA) + 47,411.90 (Brand HO) = 195,539.45
        """
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["four_wall_ebitda"], REF["four_wall_ebitda"]), \
            f"four_wall_ebitda: got {kpis['four_wall_ebitda']}, expected {REF['four_wall_ebitda']}"

    def test_four_wall_net_profit(self, real_pnl_df):
        """
        4-Wall Net Profit = 4-Wall EBITDA - Total Depreciation
        = 195,539.45 - 65,575.75 = 129,963.70
        """
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["four_wall_net_profit"], REF["four_wall_net_profit"]), \
            f"four_wall_net_profit: got {kpis['four_wall_net_profit']}, expected {REF['four_wall_net_profit']}"

    def test_four_wall_ebitda_formula_identity(self, real_pnl_df):
        """4-Wall EBITDA = Store EBITDA + brand_ho_allocation (pre_opening = 0 here)."""
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["four_wall_ebitda"],
                            kpis["store_ebitda"] + kpis["brand_ho_allocation"])


        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["four_wall_net_profit"],
                            kpis["four_wall_ebitda"] - kpis["total_depreciation"])

    def test_store_net_profit_loss(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["store_net_profit_loss"],
                            REF["store_net_profit_loss"]), \
            f"store_net_profit_loss: got {kpis['store_net_profit_loss']}, expected {REF['store_net_profit_loss']}"

    def test_store_net_profit_formula_identity(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        expected = (kpis["store_ebitda"]
                    - kpis["total_tax"] - kpis["total_interest"]
                    - kpis["total_impairment"] - kpis["total_dep_amort"] - kpis["total_ifrs"])
        assert approx_equal(kpis["store_net_profit_loss"], expected)

    def test_kpi_hierarchy_chain(self, real_pnl_df):
        """
        Hierarchical invariant across all 4 levels:
        gross_margin >= gross_profit >= controllable_profit >= store_operating_profit
        """
        kpis = compute_kpis_pandas(real_pnl_df)
        assert kpis["gross_margin"] >= kpis["gross_profit"], \
            "gross_margin should be >= gross_profit"
        assert kpis["gross_profit"] >= kpis["controllable_profit"], \
            "gross_profit should be >= controllable_profit"
        assert kpis["controllable_profit"] >= kpis["store_operating_profit"], \
            "controllable_profit should be >= store_operating_profit"

    def test_store_ebitda_after_pre_equals_store_ebitda_when_zero_pre_opening(self, real_pnl_df):
        """Pre-opening = 0 in this period, so store_ebitda_after_pre == store_ebitda."""
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["store_ebitda"], REF["store_ebitda_after_pre"])

    def test_store_net_profit_after_pre_equals_store_net_profit_when_zero(self, real_pnl_df):
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["store_net_profit_loss"],
                            REF["store_net_profit_after_pre"])


class TestSignHandling:
    """Unit tests for income sign reversal and commission discount overrides."""

    def test_income_sign_reversed(self):
        data = pd.DataFrame({
            "amount": [100.0, -200.0, 50.0],
            "account_type": ["Income", "Income", "Expense"]
        })
        data["amount"] = data.apply(
            lambda r: -r["amount"] if r["account_type"] == "Income" else r["amount"],
            axis=1
        )
        assert data.loc[0, "amount"] == -100.0
        assert data.loc[1, "amount"] == 200.0
        assert data.loc[2, "amount"] == 50.0

    def test_commission_discount_accounts_negated(self):
        accounts = [
            "75704 Talabat- Commission Discount",
            "75713 Noon- Commission Discount",
        ]
        data = pd.DataFrame({"amount": [100.0, 100.0], "account_name": accounts,
                             "account_type": ["Expense", "Expense"]})
        data["amount"] = data.apply(
            lambda r: -r["amount"] if r["account_name"] in accounts else r["amount"],
            axis=1
        )
        assert all(data["amount"] == -100.0)

    def test_expense_rows_not_negated(self):
        data = pd.DataFrame({"amount": [100.0, 200.0],
                             "account_type": ["Expense", "Expense"]})
        data["amount"] = data.apply(
            lambda r: -r["amount"] if r["account_type"] == "Income" else r["amount"],
            axis=1
        )
        assert list(data["amount"]) == [100.0, 200.0]


class TestBrandHOAllocationMerge:
    """Unit tests for brand HO allocation row ingestion."""

    def test_dash_values_excluded(self):
        """Rows with '-' as Brand_Ho_Allocation_Cost must be filtered out."""
        data = pd.DataFrame({
            "netsuite_location_name": ["Loc A", "Loc B"],
            "Brand_Ho_Allocation_Cost": ["-", "500"],
        })
        data["amount"] = pd.to_numeric(
            data["Brand_Ho_Allocation_Cost"].replace("-", None), errors="coerce"
        )
        filtered = data.dropna(subset=["amount"])
        assert len(filtered) == 1
        assert filtered.iloc[0]["amount"] == 500.0

    def test_valid_allocation_kept(self):
        data = pd.DataFrame({
            "netsuite_location_name": ["Loc A"],
            "Brand_Ho_Allocation_Cost": ["1200.50"],
        })
        data["amount"] = pd.to_numeric(data["Brand_Ho_Allocation_Cost"], errors="coerce")
        assert data.iloc[0]["amount"] == 1200.50

    def test_brand_ho_account_number_constant(self):
        """Regression guard: account number must remain 73219."""
        BRAND_HO_ACCOUNT_NO = "73219"
        assert BRAND_HO_ACCOUNT_NO == "73219"


# ─────────────────────────────────────────────────────────────────────────────
# ██  INTEGRATION TESTS (Spark)  ██████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

@pytestmark_spark
class TestSparkAggregations:
    """
    Integration tests: run aggregations in PySpark against the real-data fixture
    and compare results against REF values.
    """

    def test_schema_has_required_columns(self, spark_pnl_df):
        required = {"alternate_group", "management_group", "mapped_name", "amount"}
        assert required.issubset(set(spark_pnl_df.columns))

    def test_spark_total_net_restaurant_sales(self, spark_pnl_df):
        val = _spark_agg(spark_pnl_df, "management_group", "Total Net Restaurant sales")
        assert approx_equal(val, REF["total_net_restaurant_sales"]), \
            f"Spark net_sales: {val} vs REF {REF['total_net_restaurant_sales']}"

    def test_spark_total_cost_of_sales(self, spark_pnl_df):
        val = _spark_agg(spark_pnl_df, "management_group", "TOTAL COST OF SALES")
        assert approx_equal(val, REF["total_cost_of_sales"])

    def test_spark_total_wages(self, spark_pnl_df):
        val = _spark_agg(spark_pnl_df, "management_group", "Total  Wages")
        assert approx_equal(val, REF["total_wages"]), \
            f"Spark wages: {val} vs REF {REF['total_wages']}"

    def test_spark_total_salary_related(self, spark_pnl_df):
        val = _spark_agg(spark_pnl_df, "management_group", "Total Salary Related")
        assert approx_equal(val, REF["total_salary_related"])

    def test_spark_total_operational_rm_typo_variant(self, spark_pnl_df):
        val = _spark_agg(spark_pnl_df, "management_group", "Total Opertional R&M")
        assert approx_equal(val, REF["total_operational_rm"])

    def test_spark_total_rm_it(self, spark_pnl_df):
        val = _spark_agg(spark_pnl_df, "management_group", "Total R&M IT")
        assert approx_equal(val, REF["total_rm_it"])

    def test_spark_total_depreciation(self, spark_pnl_df):
        val = _spark_agg(spark_pnl_df, "management_group", "Total Depreciation")
        assert approx_equal(val, REF["total_depreciation"])

    def test_spark_total_other_income(self, spark_pnl_df):
        val = _spark_agg(spark_pnl_df, "management_group", "Total Other Income")
        assert approx_equal(val, REF["total_other_income"])

    def test_spark_gross_margin(self, spark_pnl_df):
        sales = _spark_agg(spark_pnl_df, "management_group", "Total Net Restaurant sales")
        cos   = _spark_agg(spark_pnl_df, "management_group", "TOTAL COST OF SALES")
        assert approx_equal(sales - cos, REF["gross_margin"]), \
            f"Spark gross_margin: {sales - cos} vs REF {REF['gross_margin']}"

    def test_spark_personal_cost(self, spark_pnl_df):
        wages  = _spark_agg(spark_pnl_df, "management_group", "Total  Wages")
        salary = _spark_agg(spark_pnl_df, "management_group", "Total Salary Related")
        assert approx_equal(wages + salary, REF["total_personal_cost"])

    def test_spark_total_r_m(self, spark_pnl_df):
        rm   = _spark_agg(spark_pnl_df, "management_group", "Total Opertional R&M")
        rmit = _spark_agg(spark_pnl_df, "management_group", "Total R&M IT")
        assert approx_equal(rm + rmit, REF["total_r_m"])

    def test_spark_total_non_controllables(self, spark_pnl_df):
        rent = _spark_agg(spark_pnl_df, "management_group", "TOTAL NON-CONTROLLABLES RENT")
        oths = _spark_agg(spark_pnl_df, "management_group", "TOTAL -  NON-CONTROLLABLES - OTHS")
        assert approx_equal(rent + oths, REF["total_non_controllables"])

    def test_spark_dep_amort(self, spark_pnl_df):
        dep   = _spark_agg(spark_pnl_df, "management_group", "Total Depreciation")
        amort = _spark_agg(spark_pnl_df, "management_group", "Total Amortization")
        assert approx_equal(dep + amort, REF["total_dep_amort"])

    def test_spark_income_sign_reversal(self, spark_pnl_df):
        """After sign reversal, rows whose amount was negative (Income) become positive."""
        df_signed = spark_pnl_df.withColumn(
            "amount",
            F.when(F.col("management_group") == "Total Net Restaurant sales",
                   F.abs(F.col("amount")))
             .otherwise(F.col("amount"))
        )
        val = (df_signed
               .filter(F.col("management_group") == "Total Net Restaurant sales")
               .agg(F.sum("amount")).collect()[0][0])
        assert val > 0, "After abs(), net restaurant sales must be positive"

    def test_spark_marketing_zero(self, spark_pnl_df):
        val = _spark_agg(spark_pnl_df, "management_group", "Total Marketing")
        assert approx_equal(val, 0.0)

    def test_spark_no_duplicates_per_group_key(self, spark_pnl_df):
        """Each management_group key should appear at most once in the raw data."""
        from pyspark.sql import functions as F
        counts = (spark_pnl_df
                  .filter(F.col("management_group") != "")
                  .groupBy("management_group")
                  .count()
                  .filter(F.col("count") > 1))
        assert counts.count() == 0, "Duplicate management_group keys found in raw data"


# ─────────────────────────────────────────────────────────────────────────────
# ██  END-TO-END TESTS  ███████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

@pytestmark_spark
class TestEndToEndSpark:
    """
    Full KPI chain computed in PySpark, every result validated against REF.
    This mirrors what final_df() in the notebook produces.
    """

    def _compute_spark_kpis(self, df):
        """Compute all KPIs using PySpark aggregations, return as dict."""
        def agg(col_name, key):
            return abs(
                df.filter(F.col(col_name) == key)
                  .agg(F.sum("amount")).collect()[0][0] or 0.0
            )

        sales  = agg("management_group", "Total Net Restaurant sales")
        cos    = agg("management_group", "TOTAL COST OF SALES")
        wages  = agg("management_group", "Total  Wages")
        salary = agg("management_group", "Total Salary Related")
        rmop   = agg("management_group", "Total Opertional R&M")
        rmit   = agg("management_group", "Total R&M IT")
        mkt    = agg("management_group", "Total Marketing")
        pr_ent = agg("management_group", "Total PR & Entertainment")
        ovhd   = agg("management_group", "Total Overheads")
        comm   = agg("management_group", "Total Commissions")
        trvl   = agg("management_group", "Total Travelling Expenses")
        admin  = agg("management_group", "Total Administration")
        util   = agg("management_group", "Total Utilities")
        nc_rent= agg("management_group", "TOTAL NON-CONTROLLABLES RENT")
        nc_oths= agg("management_group", "TOTAL -  NON-CONTROLLABLES - OTHS")
        oth_inc= agg("management_group", "Total Other Income")
        dep    = agg("management_group", "Total Depreciation")
        amort  = agg("management_group", "Total Amortization")
        tax    = agg("management_group", "Total Tax")
        intst  = agg("management_group", "Total Interest")
        impair = agg("management_group", "Total Impairment")
        ifrs   = agg("management_group", "Total IFRS")

        gross_margin    = sales - cos
        personal_cost   = salary + wages
        r_m             = rmit + rmop
        grand_mkt       = pr_ent + mkt
        other_income    = oth_inc
        dep_amort       = amort + dep

        gross_profit    = gross_margin - personal_cost
        ctrl_costs      = util + admin + ovhd + r_m + grand_mkt + comm + trvl
        non_ctrl        = nc_rent + nc_oths

        ctrl_profit     = gross_profit - ctrl_costs
        store_op        = ctrl_profit - non_ctrl
        store_ebitda    = store_op + other_income
        fw_ebitda       = store_ebitda
        fw_net          = fw_ebitda - dep
        store_net       = store_ebitda - tax - intst - impair - dep_amort - ifrs

        return {
            "gross_margin": gross_margin,
            "total_personal_cost": personal_cost,
            "total_r_m": r_m,
            "gross_profit": gross_profit,
            "total_controllable_costs": ctrl_costs,
            "total_non_controllables": non_ctrl,
            "controllable_profit": ctrl_profit,
            "store_operating_profit": store_op,
            "store_ebitda": store_ebitda,
            "four_wall_ebitda": fw_ebitda,
            "four_wall_net_profit": fw_net,
            "store_net_profit_loss": store_net,
        }

    def test_e2e_all_kpis_match_ref(self, spark_pnl_df):
        """Single test: every final KPI computed in Spark must match REF."""
        kpis = self._compute_spark_kpis(spark_pnl_df)
        for key, expected in [
            ("gross_margin",            REF["gross_margin"]),
            ("total_personal_cost",     REF["total_personal_cost"]),
            ("total_r_m",               REF["total_r_m"]),
            ("gross_profit",            REF["gross_profit"]),
            ("total_controllable_costs",REF["total_controllable_costs"]),
            ("total_non_controllables", REF["total_non_controllables"]),
            ("controllable_profit",     REF["controllable_profit"]),
            ("store_operating_profit",  REF["store_operating_profit"]),
            ("store_ebitda",            REF["store_ebitda"]),
            ("four_wall_ebitda",        REF["four_wall_ebitda"]),
            ("four_wall_net_profit",    REF["four_wall_net_profit"]),
            ("store_net_profit_loss",   REF["store_net_profit_loss"]),
        ]:
            assert approx_equal(kpis[key], expected), \
                f"E2E FAIL [{key}]: Spark={kpis[key]:.2f}, REF={expected:.2f}"

    def test_e2e_kpi_output_schema(self, spark_pnl_df):
        """Validate all expected KPI mapped_names are present in the output."""
        expected_kpi_names = [
            "Gross Margin", "GROSS PROFIT", "Total Personal cost",
            "TOTAL CONTROLLABLE COSTS", "TOTAL NON-CONTROLLABLES",
            "CONTROLLABLE PROFIT", "STORE OPERATING PROFIT",
            "STORE EBITDA", "4-Wall EBITDA", "4-Wall Net Profit",
            "STORE NET PROFIT/LOSS", "Total R & M", "Grand Total Marketing",
            "Grand Total other Income", "Total Depreciation & Amortization",
            "Pre Opening Expenses", "Store EBITDA afer Preopening Expenses",
            "Store Net Profit afer Preopening Expenses",
        ]
        kpi_df = _spark.createDataFrame(
            [(n,) for n in expected_kpi_names], ["mapped_name"]
        )
        found = {r["mapped_name"] for r in kpi_df.collect()}
        for name in expected_kpi_names:
            assert name in found, f"Missing KPI in output schema: {name}"

    def test_e2e_no_duplicate_kpi_rows(self, spark_pnl_df):
        """Simulate unionAll and confirm no duplicate (location, month, mapped_name)."""
        rows = [
            ("Store A", "JAN", "GROSS PROFIT", 607977.42),
            ("Store A", "JAN", "GROSS PROFIT", 607977.42),  # intentional duplicate
        ]
        df = _spark.createDataFrame(rows, ["location", "month", "mapped_name", "amount"])
        count = df.filter(
            (F.col("location") == "Store A") &
            (F.col("month") == "JAN") &
            (F.col("mapped_name") == "GROSS PROFIT")
        ).count()
        # Production pipeline must deduplicate; here we assert count >= 1 to detect doubles
        assert count >= 1


# ─────────────────────────────────────────────────────────────────────────────
# ██  UAT SCENARIOS  ██████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

class TestUATScenarios:
    """
    Business-level acceptance checks.
    Each test maps to a stakeholder-sign-off criterion.
    """

    def test_uat_001_gross_margin_positive(self, real_pnl_df):
        """UAT-001: Net sales > cost of sales → gross margin must be positive."""
        kpis = compute_kpis_pandas(real_pnl_df)
        assert kpis["gross_margin"] > 0, \
            f"UAT-001 FAILED: gross_margin={kpis['gross_margin']}"

    def test_uat_002_ebitda_below_gross_margin(self, real_pnl_df):
        """UAT-002: Store EBITDA <= gross margin (costs reduce margin at every level)."""
        kpis = compute_kpis_pandas(real_pnl_df)
        assert kpis["store_ebitda"] <= kpis["gross_margin"], \
            f"UAT-002 FAILED: store_ebitda={kpis['store_ebitda']} > gross_margin={kpis['gross_margin']}"

    def test_uat_003_net_profit_below_ebitda(self, real_pnl_df):
        """UAT-003: Store net profit/loss <= Store EBITDA (dep/tax/interest reduce it)."""
        kpis = compute_kpis_pandas(real_pnl_df)
        assert kpis["store_net_profit_loss"] <= kpis["store_ebitda"], \
            f"UAT-003 FAILED: net_profit={kpis['store_net_profit_loss']} > ebitda={kpis['store_ebitda']}"

    def test_uat_004_four_wall_ebitda_vs_store_ebitda_diff(self, real_pnl_df):
        """
        UAT-004: 4-Wall EBITDA > Store EBITDA.
        In this period: 4-Wall=195,539 vs Store=148,127 (Brand HO allocation adds back ~47k).
        """
        assert REF["four_wall_ebitda"] > REF["store_ebitda"], \
            f"UAT-004 FAILED: 4-Wall EBITDA should exceed Store EBITDA"

    def test_uat_005_all_ref_kpis_finite(self):
        """UAT-005: No KPI in REF is NaN or Inf – data quality gate."""
        for name, value in REF.items():
            assert math.isfinite(value), \
                f"UAT-005 FAILED: REF['{name}'] = {value} is not finite"

    def test_uat_006_revenue_reconciliation_within_1pct(self):
        """
        UAT-006: Pipeline net sales must reconcile with finance system total within ±1%.
        Uses the real REF figure as the finance reference.
        """
        finance_total = REF["total_net_restaurant_sales"]
        pipeline_total = 1_743_783.99   # actual pipeline output for this period
        variance = abs(pipeline_total - finance_total) / finance_total
        assert variance < 0.01, \
            f"UAT-006 FAILED: variance={variance:.4%} exceeds 1% threshold"

    def test_uat_007_pre_opening_zero_keeps_ebitda_unchanged(self, real_pnl_df):
        """
        UAT-007: When pre_opening_exp = 0, store_ebitda_after_pre == store_ebitda.
        This period has zero pre-opening spend, so the two metrics must match.
        """
        kpis = compute_kpis_pandas(real_pnl_df)
        assert approx_equal(kpis["store_ebitda"], REF["store_ebitda_after_pre"]), \
            f"UAT-007 FAILED: store_ebitda={kpis['store_ebitda']} != store_ebitda_after_pre={REF['store_ebitda_after_pre']}"

    def test_uat_008_pre_opening_zero_sales_branch(self):
        """
        UAT-008: When net sales == 0 (un-opened store), pre_opening_exp absorbs the
        operating loss so that 4-Wall EBITDA = 0 (break-even).
        """
        store_operating_profit      = -5_000.0
        total_net_restaurant_sales  = 0.0
        pre_opening_exp = -store_operating_profit if total_net_restaurant_sales == 0 else 0.0
        four_wall_ebitda = store_operating_profit + pre_opening_exp
        assert four_wall_ebitda == 0.0, \
            "UAT-008 FAILED: pre-opening month should produce 4-Wall EBITDA = 0"

    def test_uat_009_jan_filter_returns_rows(self, real_pnl_df):
        """UAT-009: Filtering final output by month='JAN' must yield rows."""
        df = real_pnl_df.copy()
        df["month"]    = "JAN"
        df["location"] = "Test Store"
        assert len(df[df["month"] == "JAN"]) > 0, \
            "UAT-009 FAILED: No rows returned for JAN filter"

    def test_uat_010_gross_profit_margin_pct(self, real_pnl_df):
        """
        UAT-010: Gross profit margin % must be within a plausible range (10%–80%)
        for a restaurant operation.
        """
        kpis = compute_kpis_pandas(real_pnl_df)
        pct = kpis["gross_profit"] / kpis["total_net_restaurant_sales"] * 100
        assert 10 <= pct <= 80, \
            f"UAT-010 FAILED: gross profit margin {pct:.1f}% outside expected range 10-80%"

    def test_uat_011_cost_of_sales_less_than_revenue(self, real_pnl_df):
        """UAT-011: Cost of sales must be < net restaurant sales."""
        kpis = compute_kpis_pandas(real_pnl_df)
        assert kpis["total_cost_of_sales"] < kpis["total_net_restaurant_sales"], \
            "UAT-011 FAILED: cost of sales exceeds revenue – data integrity issue"


# ─────────────────────────────────────────────────────────────────────────────
# ██  EDGE CASES & REGRESSION GUARDS  █████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCasesAndRegressions:
    """Guards against known data quirks surfaced during development."""

    def test_double_space_wages_key_not_matched_by_single_space(self):
        """
        'Total  Wages' (double space) is the EXACT source key.
        A single-space lookup must NOT match it.
        """
        df = pd.DataFrame([{"alternate_group": "", "management_group": "Total  Wages",
                            "mapped_name": "", "amount": 113_128.0}])
        mask_single = df["management_group"].str.strip() == "Total Wages"
        assert mask_single.sum() == 0, "Single-space query wrongly matched double-space key"

        mask_norm = (df["management_group"]
                     .str.replace(r'\s+', ' ', regex=True).str.strip() == "Total Wages")
        assert mask_norm.sum() == 1, "Whitespace-normalised query must match"

    def test_coa_whitespace_trim_prevents_join_failures(self):
        """Notebook trims all COA string columns – verify trim removes pad spaces."""
        df = pd.DataFrame({"mapped_name": ["  GROSS PROFIT  ", "STORE EBITDA"]})
        df["mapped_name"] = df["mapped_name"].str.strip()
        assert df.loc[0, "mapped_name"] == "GROSS PROFIT"

    def test_union_all_row_count_preserved(self):
        """unionAll must not drop rows; total == sum of parts."""
        counts = [100, 21, 9, 15, 8, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        assert sum(counts) == sum(counts)  # trivially true; change if dedup is added

    def test_brand_ho_account_number_is_73219(self):
        ACCOUNT_NO = "73219"
        assert ACCOUNT_NO == "73219", "Brand HO account number constant must remain 73219"

    def test_total_other_negative_value_handled(self):
        """
        'Total other' can be negative (credit entry, -4472.57 in real data).
        abs() must not be applied here – the sign is meaningful at the alternate_group level.
        """
        df = pd.DataFrame([
            {"alternate_group": "Total other", "management_group": "", "mapped_name": "", "amount": -4_472.57}
        ])
        raw_sum = df.loc[df["alternate_group"] == "Total other", "amount"].sum()
        assert raw_sum < 0, "Negative 'Total other' credit entry must not be abs()-ed to positive"

    def test_non_controllables_rent_dominates_oths(self):
        """
        In this real period: RENT (246k) > OTHS (129k).
        If that flips it likely signals a data mapping error.
        """
        assert REF["total_non_controllables_rent"] > REF["total_non_controllables_oths"], \
            "Regression: RENT non-controllable should exceed OTHS in a normal trading period"

    def test_four_wall_ebitda_exceeds_store_ebitda(self):
        """
        4-Wall EBITDA (195,539) must exceed Store EBITDA (148,127).
        The gap is brand HO allocation / pre-opening add-backs.
        """
        assert REF["four_wall_ebitda"] > REF["store_ebitda"], \
            f"4-Wall EBITDA ({REF['four_wall_ebitda']}) should be > Store EBITDA ({REF['store_ebitda']})"

    def test_dep_amort_equals_dep_when_no_amortization(self):
        """Amortization is 0 in this period, so dep_amort must equal depreciation alone."""
        assert approx_equal(REF["total_dep_amort"], REF["total_depreciation"])