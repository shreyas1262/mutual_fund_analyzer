#!/usr/bin/env python3
"""
Test script to verify Python and PySpark installation
"""

import sys
print(f"✅ Python version: {sys.version}")

try:
    import pyspark
    print(f"✅ PySpark version: {pyspark.__version__}")
except ImportError:
    print("❌ PySpark not installed")

try:
    import pandas as pd
    print(f"✅ Pandas version: {pd.__version__}")
except ImportError:
    print("❌ Pandas not installed")

try:
    import yfinance as yf
    print("✅ YFinance installed")
except ImportError:
    print("❌ YFinance not installed")

try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("TestApp").getOrCreate()
    print("✅ Spark session created successfully")
    
    # Test basic Spark operation
    data = [("Alice", 25), ("Bob", 30)]
    df = spark.createDataFrame(data, ["name", "age"])
    print(f"✅ Test DataFrame created with {df.count()} rows")
    
    spark.stop()
    print("✅ Spark session stopped successfully")
    
except Exception as e:
    print(f"❌ Spark test failed: {e}")

print("\n🎉 Installation test completed!")
