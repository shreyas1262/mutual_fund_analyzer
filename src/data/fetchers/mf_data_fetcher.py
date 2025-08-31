from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import requests
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Union
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparkMutualFundDataFetcher:
    """
    A comprehensive PySpark-based class to fetch and process Indian mutual fund data
    """
    
    def __init__(self, app_name: str = "MutualFundAnalysis"):
        """
        Initialize Spark session and API configuration
        """
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        self.base_url = "https://api.mfapi.in"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Define schemas for type safety
        self.scheme_schema = StructType([
            StructField("schemeCode", StringType(), True),
            StructField("schemeName", StringType(), True),
            StructField("schemeType", StringType(), True)
        ])
        
        self.nav_schema = StructType([
            StructField("date", StringType(), True),
            StructField("nav", StringType(), True)
        ])
        
        logger.info("Spark session initialized successfully")
    
    def get_all_schemes(self, save_path: Optional[str] = None) -> "DataFrame":
        """
        Fetch all mutual fund schemes and return as Spark DataFrame
        
        Args:
            save_path: Optional path to save parquet file
            
        Returns:
            Spark DataFrame with scheme details
        """
        try:
            response = self.session.get(f"{self.base_url}/mf")
            response.raise_for_status()
            
            schemes_data = response.json()
            
            # Convert to Spark DataFrame
            df = self.spark.createDataFrame(schemes_data, schema=self.scheme_schema)
            
            if save_path:
                df.write.mode("overwrite").parquet(save_path)
                logger.info(f"Schemes data saved to {save_path}")
            
            logger.info(f"Fetched {df.count()} mutual fund schemes")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching schemes: {e}")
            return self.spark.createDataFrame([], schema=self.scheme_schema)
    
    def get_scheme_details(self, scheme_code: str) -> Dict:
        """
        Get detailed information about a specific scheme
        
        Args:
            scheme_code: The scheme code (e.g., '120503')
            
        Returns:
            Dictionary with scheme details and latest NAV
        """
        try:
            response = self.session.get(f"{self.base_url}/mf/{scheme_code}")
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching scheme {scheme_code}: {e}")
            return {}
    
    def get_historical_nav_spark(self, scheme_code: str, scheme_name: str = None,
                                start_date: Optional[str] = None, 
                                end_date: Optional[str] = None) -> "DataFrame":
        """
        Fetch historical NAV data and return as Spark DataFrame
        
        Args:
            scheme_code: The scheme code
            scheme_name: Optional scheme name for reference
            start_date: Start date in 'DD-MM-YYYY' format (optional)
            end_date: End date in 'DD-MM-YYYY' format (optional)
            
        Returns:
            Spark DataFrame with processed NAV data
        """
        try:
            # Get data from API
            response = self.session.get(f"{self.base_url}/mf/{scheme_code}")
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                logger.warning(f"No historical data found for scheme {scheme_code}")
                return self.spark.createDataFrame([], schema=self.nav_schema)
            
            # Convert to Spark DataFrame
            nav_df = self.spark.createDataFrame(data['data'], schema=self.nav_schema)
            
            # Process data with Spark SQL
            nav_df = nav_df.withColumn("scheme_code", lit(scheme_code))
            
            if scheme_name:
                nav_df = nav_df.withColumn("scheme_name", lit(scheme_name))
            
            # Convert date and NAV columns
            nav_df = nav_df.withColumn("date", 
                                     to_date(col("date"), "dd-MM-yyyy")) \
                           .withColumn("nav", 
                                     col("nav").cast(DoubleType()))
            
            # Filter by date range if provided
            if start_date:
                start_dt = datetime.strptime(start_date, '%d-%m-%Y').date()
                nav_df = nav_df.filter(col("date") >= lit(start_dt))
            
            if end_date:
                end_dt = datetime.strptime(end_date, '%d-%m-%Y').date()
                nav_df = nav_df.filter(col("date") <= lit(end_dt))
            
            # Sort by date
            nav_df = nav_df.orderBy("date")
            
            logger.info(f"Processed {nav_df.count()} NAV records for scheme {scheme_code}")
            return nav_df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {scheme_code}: {e}")
            return self.spark.createDataFrame([], schema=self.nav_schema)
    
    def get_multiple_schemes_spark(self, scheme_mapping: Dict[str, str],
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  save_path: Optional[str] = None) -> "DataFrame":
        """
        Fetch historical data for multiple schemes and combine into single DataFrame
        
        Args:
            scheme_mapping: Dict mapping scheme codes to scheme names
            start_date: Start date in 'DD-MM-YYYY' format
            end_date: End date in 'DD-MM-YYYY' format
            save_path: Optional path to save parquet file
            
        Returns:
            Combined Spark DataFrame with all schemes data
        """
        all_dataframes = []
        
        for i, (scheme_code, scheme_name) in enumerate(scheme_mapping.items()):
            logger.info(f"Fetching data for {scheme_name} ({scheme_code}) - {i+1}/{len(scheme_mapping)}")
            
            df = self.get_historical_nav_spark(scheme_code, scheme_name, start_date, end_date)
            
            if df.count() > 0:
                all_dataframes.append(df)
            
            # Rate limiting
            if i < len(scheme_mapping) - 1:
                time.sleep(0.5)
        
        if not all_dataframes:
            logger.warning("No data fetched for any schemes")
            return self.spark.createDataFrame([], schema=self.nav_schema)
        
        # Union all DataFrames
        combined_df = all_dataframes[0]
        for df in all_dataframes[1:]:
            combined_df = combined_df.union(df)
        
        # Add metadata columns
        combined_df = combined_df.withColumn("fetch_timestamp", current_timestamp())
        
        if save_path:
            combined_df.write.mode("overwrite").parquet(save_path)
            logger.info(f"Combined data saved to {save_path}")
        
        logger.info(f"Combined dataset created with {combined_df.count()} total records")
        return combined_df
    
    def create_pivot_nav_dataset(self, nav_df: "DataFrame", 
                                save_path: Optional[str] = None) -> "DataFrame":
        """
        Create a pivoted dataset with schemes as columns and dates as rows
        
        Args:
            nav_df: DataFrame with nav data in long format
            save_path: Optional path to save parquet file
            
        Returns:
            Pivoted Spark DataFrame
        """
        # Create pivot table
        pivot_df = nav_df.groupBy("date") \
                         .pivot("scheme_name") \
                         .agg(first("nav").alias("nav"))
        
        # Sort by date
        pivot_df = pivot_df.orderBy("date")
        
        if save_path:
            pivot_df.write.mode("overwrite").parquet(save_path)
            logger.info(f"Pivoted data saved to {save_path}")
        
        return pivot_df
    
    def search_schemes_spark(self, keyword: str, scheme_type: Optional[str] = None,
                           save_path: Optional[str] = None) -> "DataFrame":
        """
        Search for schemes using Spark SQL
        
        Args:
            keyword: Search keyword
            scheme_type: Optional scheme type filter
            save_path: Optional path to save results
            
        Returns:
            Filtered Spark DataFrame
        """
        all_schemes = self.get_all_schemes()
        
        # Create temporary view for SQL operations
        all_schemes.createOrReplaceTempView("schemes")
        
        # Build SQL query
        base_query = f"""
        SELECT * FROM schemes 
        WHERE LOWER(schemeName) LIKE LOWER('%{keyword}%')
        """
        
        if scheme_type:
            base_query += f" AND LOWER(schemeType) LIKE LOWER('%{scheme_type}%')"
        
        filtered_df = self.spark.sql(base_query)
        
        if save_path:
            filtered_df.write.mode("overwrite").parquet(save_path)
        
        logger.info(f"Found {filtered_df.count()} schemes matching '{keyword}'")
        return filtered_df
    
    def calculate_performance_metrics_spark(self, nav_df: "DataFrame") -> "DataFrame":
        """
        Calculate basic performance metrics using Spark SQL
        
        Args:
            nav_df: DataFrame with NAV data
            
        Returns:
            DataFrame with performance metrics
        """
        # Create temporary view
        nav_df.createOrReplaceTempView("nav_data")
        
        # Calculate metrics using Spark SQL
        metrics_query = """
        WITH scheme_stats AS (
            SELECT 
                scheme_name,
                scheme_code,
                COUNT(*) as total_observations,
                MIN(date) as start_date,
                MAX(date) as end_date,
                FIRST_VALUE(nav) OVER (PARTITION BY scheme_name ORDER BY date ASC) as first_nav,
                FIRST_VALUE(nav) OVER (PARTITION BY scheme_name ORDER BY date DESC) as latest_nav,
                MIN(nav) as min_nav,
                MAX(nav) as max_nav,
                AVG(nav) as avg_nav,
                STDDEV(nav) as nav_volatility
            FROM nav_data
            WHERE nav IS NOT NULL
            GROUP BY scheme_name, scheme_code
        ),
        performance_calc AS (
            SELECT *,
                ((latest_nav - first_nav) / first_nav) * 100 as total_return_pct,
                DATEDIFF(end_date, start_date) as period_days,
                ((max_nav - min_nav) / min_nav) * 100 as max_fluctuation_pct
            FROM scheme_stats
        )
        SELECT *,
            CASE 
                WHEN period_days > 0 
                THEN POW((latest_nav / first_nav), (365.0 / period_days)) - 1
                ELSE 0 
            END * 100 as annualized_return_pct
        FROM performance_calc
        """
        
        metrics_df = self.spark.sql(metrics_query)
        return metrics_df
    
    def stop_spark(self):
        """
        Stop the Spark session
        """
        self.spark.stop()
        logger.info("Spark session stopped")

# Utility functions for popular schemes and categories
def get_popular_schemes_mapping():
    """
    Returns a mapping of popular Indian mutual fund schemes
    """
    return {
        # Large Cap Funds
        '120503': 'Axis Bluechip Fund',
        '118989': 'Mirae Asset Large Cap Fund',
        '119533': 'SBI Bluechip Fund',
        '125497': 'ICICI Prudential Bluechip Fund',
        '120716': 'HDFC Top 100 Fund',
        
        # Mid Cap Funds
        '119551': 'Axis Midcap Fund',
        '125494': 'ICICI Prudential MidCap Fund',
        
        # Small Cap Funds
        '119552': 'Axis Small Cap Fund',
        '125498': 'ICICI Prudential SmallCap Fund',
        
        # Index Funds
        '119434': 'UTI Nifty 50 Index Fund',
        '120716': 'HDFC Index Fund Nifty 50 Plan',
        
        # Multi Cap / Flexi Cap
        '100127': 'Kotak Standard Multicap Fund',
        '118989': 'Parag Parikh Long Term Equity Fund'
    }

def demo_pyspark_usage():
    """
    Demonstrate PySpark mutual fund data fetching and processing
    """
    # Initialize the fetcher
    fetcher = SparkMutualFundDataFetcher()
    
    try:
        # Example 1: Get all schemes and save as parquet
        print("Fetching all schemes...")
        all_schemes_df = fetcher.get_all_schemes(save_path="data/all_schemes.parquet")
        
        print("Sample schemes data:")
        all_schemes_df.show(5, truncate=False)
        print(f"Total schemes: {all_schemes_df.count()}")
        
        # Example 2: Search for specific schemes
        print("\nSearching for 'Axis' equity schemes...")
        axis_schemes = fetcher.search_schemes_spark(
            keyword="Axis", 
            scheme_type="Equity",
            save_path="data/axis_schemes.parquet"
        )
        axis_schemes.show(5, truncate=False)
        
        # Example 3: Fetch historical data for multiple schemes
        print("\nFetching historical data for popular schemes...")
        popular_schemes = get_popular_schemes_mapping()
        
        # Take first 3 schemes for demo
        demo_schemes = dict(list(popular_schemes.items())[:3])
        
        combined_nav_df = fetcher.get_multiple_schemes_spark(
            scheme_mapping=demo_schemes,
            start_date='01-01-2023',
            end_date='31-07-2025',
            save_path="data/historical_nav_data.parquet"
        )
        
        print("Sample historical data:")
        combined_nav_df.show(10)
        
        # Example 4: Create pivot table for analysis
        print("\nCreating pivot table...")
        pivot_df = fetcher.create_pivot_nav_dataset(
            combined_nav_df,
            save_path="data/pivot_nav_data.parquet"
        )
        
        print("Pivot table sample:")
        pivot_df.show(5)
        
        # Example 5: Calculate basic performance metrics
        print("\nCalculating performance metrics...")
        performance_df = fetcher.calculate_performance_metrics_spark(combined_nav_df)
        
        print("Performance metrics:")
        performance_df.show(truncate=False)
        
        # Save performance metrics
        performance_df.write.mode("overwrite").parquet("data/performance_metrics.parquet")
        
        # Example 6: Data exploration with Spark SQL
        print("\nData exploration using Spark SQL...")
        
        # Register DataFrame for SQL queries
        combined_nav_df.createOrReplaceTempView("mf_nav_data")
        
        # Query 1: Latest NAV for all schemes
        latest_nav_query = """
        SELECT 
            scheme_name,
            scheme_code,
            date,
            nav,
            ROW_NUMBER() OVER (PARTITION BY scheme_code ORDER BY date DESC) as rn
        FROM mf_nav_data
        QUALIFY rn = 1
        ORDER BY scheme_name
        """
        
        latest_navs = fetcher.spark.sql(latest_nav_query)
        print("Latest NAVs:")
        latest_navs.show()
        
        # Query 2: Monthly aggregated data
        monthly_agg_query = """
        SELECT 
            scheme_name,
            YEAR(date) as year,
            MONTH(date) as month,
            FIRST_VALUE(nav) OVER (PARTITION BY scheme_name, YEAR(date), MONTH(date) ORDER BY date ASC) as month_start_nav,
            FIRST_VALUE(nav) OVER (PARTITION BY scheme_name, YEAR(date), MONTH(date) ORDER BY date DESC) as month_end_nav,
            COUNT(*) as trading_days
        FROM mf_nav_data
        GROUP BY scheme_name, YEAR(date), MONTH(date)
        ORDER BY scheme_name, year, month
        """
        
        monthly_data = fetcher.spark.sql(monthly_agg_query)
        print("Monthly aggregated data:")
        monthly_data.show(10)
        
        # Save monthly data
        monthly_data.write.mode("overwrite").parquet("data/monthly_nav_data.parquet")
        
        print("\nDemo completed successfully!")
        print("Files created:")
        print("- data/all_schemes.parquet")
        print("- data/historical_nav_data.parquet") 
        print("- data/pivot_nav_data.parquet")
        print("- data/performance_metrics.parquet")
        print("- data/monthly_nav_data.parquet")
        
        return {
            'schemes': all_schemes_df,
            'historical_nav': combined_nav_df,
            'pivot_nav': pivot_df,
            'performance': performance_df,
            'monthly_data': monthly_data
        }
        
    finally:
        # Clean up
        fetcher.stop_spark()

class SparkDataProcessor:
    """
    Advanced data processing utilities for mutual fund analysis using PySpark
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_returns_spark(self, nav_df: "DataFrame", 
                               return_periods: List[int] = [1, 7, 30]) -> "DataFrame":
        """
        Calculate returns for different periods using Spark window functions
        
        Args:
            nav_df: DataFrame with NAV data (should have date, scheme_name, nav columns)
            return_periods: List of periods to calculate returns for
            
        Returns:
            DataFrame with returns for different periods
        """
        from pyspark.sql.window import Window
        
        # Define window specification
        window_spec = Window.partitionBy("scheme_name").orderBy("date")
        
        returns_df = nav_df
        
        for period in return_periods:
            returns_df = returns_df.withColumn(
                f"return_{period}d",
                ((col("nav") - lag("nav", period).over(window_spec)) / 
                 lag("nav", period).over(window_spec)) * 100
            )
        
        return returns_df
    
    def calculate_rolling_metrics(self, nav_df: "DataFrame", 
                                 window_days: int = 252) -> "DataFrame":
        """
        Calculate rolling metrics using Spark window functions
        
        Args:
            nav_df: DataFrame with NAV data
            window_days: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        from pyspark.sql.window import Window
        
        window_spec = Window.partitionBy("scheme_name") \
                           .orderBy("date") \
                           .rowsBetween(-window_days + 1, 0)
        
        rolling_df = nav_df.withColumn("rolling_avg_nav", 
                                      avg("nav").over(window_spec)) \
                          .withColumn("rolling_min_nav", 
                                     min("nav").over(window_spec)) \
                          .withColumn("rolling_max_nav", 
                                     max("nav").over(window_spec)) \
                          .withColumn("rolling_volatility", 
                                     stddev("nav").over(window_spec))
        
        return rolling_df
    
    def create_scheme_summary(self, nav_df: "DataFrame") -> "DataFrame":
        """
        Create summary statistics for each scheme
        
        Args:
            nav_df: DataFrame with NAV data
            
        Returns:
            DataFrame with summary statistics
        """
        summary_df = nav_df.groupBy("scheme_name", "scheme_code") \
                          .agg(
                              count("nav").alias("total_observations"),
                              min("date").alias("start_date"),
                              max("date").alias("end_date"),
                              first("nav").alias("first_nav"),
                              last("nav").alias("latest_nav"),
                              min("nav").alias("min_nav"),
                              max("nav").alias("max_nav"),
                              avg("nav").alias("avg_nav"),
                              stddev("nav").alias("nav_volatility")
                          )
        
        # Calculate total return and annualized return
        summary_df = summary_df.withColumn(
            "total_return_pct",
            ((col("latest_nav") - col("first_nav")) / col("first_nav")) * 100
        ).withColumn(
            "period_days",
            datediff(col("end_date"), col("start_date"))
        ).withColumn(
            "annualized_return_pct",
            (pow(col("latest_nav") / col("first_nav"), 365.0 / col("period_days")) - 1) * 100
        )
        
        return summary_df

if __name__ == "__main__":
    # Run the demo
    print("Starting PySpark Mutual Fund Data Fetcher Demo...")
    results = demo_pyspark_usage()