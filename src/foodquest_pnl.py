
# {path}/src/foodquest_pnl.py
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def create_total_row(df, group_cols, total_col_name, total_label, dimension_cols):
    """
    Create aggregated total rows (subgroup, group, major group totals).
    
    Parameters:
    - df: Input dataframe
    - group_cols: List of columns to group by (e.g., ['location', 'month', 'year', 'sub_group'])
    - total_col_name: Column name to use for the total label (e.g., 'sub_group')
    - total_label: Label type ('Total' or 'Grand Total')
    - dimension_cols: List of dimension columns to preserve
    """
    return df.groupBy(*group_cols).agg(
        sum("amount").alias("amount")
    ).select(
        col("location"),
        col("month"),
        col("year"),
        concat(lit("Total "), col(total_col_name)).alias("account_name"),
        concat(lit("Total "), col(total_col_name)).alias("name"),
        lit(None).cast("string").alias("accoun_type"),
        lit(None).alias("majour_group"),
        lit(None).alias("group"),
        lit(None).alias("sub_group"),
        col("amount"),
        *[col(c) for c in dimension_cols],
        lit(total_label).alias("Detail/Total")
    )


def create_calculated_metric(df, metric_name, calculation_expr, dimension_cols, total_label="Grand Total"):
    """
    Create calculated metric rows (Gross Profit, Operating Profit, EBITDA, Net Profit).
    
    Parameters:
    - df: Input dataframe
    - metric_name: Name of the metric (e.g., 'Gross Profit')
    - calculation_expr: PySpark column expression for the calculation
    - dimension_cols: List of dimension columns to preserve
    - total_label: Label type (default 'Grand Total')
    """
    group_cols = ["location", "month", "year"] + dimension_cols
    
    return df.groupBy(*group_cols).agg(
        calculation_expr.alias("amount")
    ).select(
        col("location"),
        col("month"),
        col("year"),
        lit(metric_name).alias("account_name"),
        lit(metric_name).alias("name"),
        lit(None).cast("string").alias("accoun_type"),
        lit(None).alias("majour_group"),
        lit(None).alias("group"),
        lit(None).alias("sub_group"),
        col("amount"),
        *[col(c) for c in dimension_cols],
        lit(total_label).alias("Detail/Total")
    )


def add_previous_year_data(df):
    """Add previous year (PY) amounts via self-join."""
    df_current = df.withColumn("year", year(col("year")))
    
    df_py = df_current.alias("py").select(
        (col("year") + 1).alias("year_join"),
        col("location").alias("py_location"),
        col("account_name").alias("py_account_name"),
        col("amount").alias("py_amount"),
        col("month").alias("py_month")
    )
    
    return df_current.alias("curr").join(
        df_py,
        (col("curr.year") == col("year_join")) &  
        (col("curr.location") == col("py_location")) &  
        (col("curr.account_name") == col("py_account_name")) &  
        (col("curr.month") == col("py_month")),
        "left"
    ).select(
        col("curr.*"),
        coalesce(col("py_amount"), lit(0.0)).alias("py_amount")
    )


def add_net_sales_calculations(df):
    """Add actual and PY net sales calculations at location, brand, and company levels."""
    window_location = Window.partitionBy("location", "year", "month")
    window_brand = Window.partitionBy("brand_id", "year", "month")
    window_company = Window.partitionBy("company_id", "year", "month")
    
    sales_condition = col("account_name").like("Total Sales")
    
    return df \
        .withColumn("actual_net_sales",
            sum(when(sales_condition, col("amount")).otherwise(0.0)).over(window_location)) \
        .withColumn("py_net_sales",
            sum(when(sales_condition, col("py_amount")).otherwise(0.0)).over(window_location)) \
        .withColumn("brand_act_net_sales",
            sum(when(sales_condition, col("amount")).otherwise(0.0)).over(window_brand)) \
        .withColumn("brand_py_net_sales",
            sum(when(sales_condition, col("py_amount")).otherwise(0.0)).over(window_brand)) \
        .withColumn("company_act_net_sales",
            sum(when(sales_condition, col("amount")).otherwise(0.0)).over(window_company)) \
        .withColumn("company_py_net_sales",
            sum(when(sales_condition, col("py_amount")).otherwise(0.0)).over(window_company))


def get_dimension_columns():
    """Return list of standard dimension columns used throughout transformations."""
    return [
        "netsuite_location_name", "type", "location_id", "brand_id",
        "company_id", "parent_company", "country_code", "zone", "store_type", "city"
    ]

def join_dataframes(base_df, join_configs):
    """
    Perform multiple joins on a base dataframe.
    
    Parameters:
    -----------
    base_df : DataFrame
        The base dataframe to join other dataframes to
    join_configs : list of dict
        List of join configurations, each containing:
        - 'df': DataFrame to join
        - 'left_key': column name or expression from left/base df
        - 'right_key': column name or expression from right df
        - 'join_type': 'inner', 'left', 'right', 'outer', etc.
        
    Returns:
    --------
    DataFrame: Result of all joins applied sequentially
    
    Example:
    --------
    join_configs = [
        {
            'df': df_coa_master,
            'left_key': col("accountNumber").cast("string"),
            'right_key': df_coa_master["account_number"].cast("string"),
            'join_type': 'inner'
        },
        {
            'df': df_location_master,
            'left_key': col("location"),
            'right_key': df_location_master["netsuite_location_name"],
            'join_type': 'left'
        }
    ]
    result = join_dataframes(df, join_configs)
    """
    result_df = base_df
    
    for config in join_configs:
        result_df = result_df.join(
            config['df'],
            config['left_key'] == config['right_key'],
            config['join_type']
        )
    
    return result_df
