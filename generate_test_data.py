import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_large_dataset(num_rows=15000, filename="large_retail_dataset.csv"):
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating {num_rows} rows of data...")
    
    # 1. Dates (Time Series Trend)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 730)) for _ in range(num_rows)]
    
    # 2. Categories
    categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports"]
    category_choices = np.random.choice(categories, size=num_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    
    # 3. Age (with some anomalies)
    ages = np.random.normal(loc=35, scale=12, size=num_rows).astype(int)
    # Inject anomalies
    for i in range(100):
        ages[random.randint(0, num_rows-1)] = random.choice([-5, -10, 0, 110, 150])
        
    # 4. Price & Quantity
    prices = np.round(np.random.lognormal(mean=3.5, sigma=1.2, size=num_rows), 2)
    quantities = np.random.poisson(lam=2, size=num_rows) + 1
    
    # 5. Revenue (Highly correlated with Price * Quantity, but with some noise)
    # We want a strong correlation to test the Top Correlations chart!
    base_revenue = prices * quantities
    noise = np.random.normal(0, 10, size=num_rows)
    revenues = np.round(base_revenue + noise, 2)
    
    # Inject anomalies in revenue
    for i in range(50):
        revenues[random.randint(0, num_rows-1)] = random.choice([0, -50.5])
        
    # 6. Customer Satisfaction (Correlated with Age negatively slightly, just for fun)
    satisfaction = np.clip(np.round(10 - (ages / 15) + np.random.normal(0, 1, num_rows)), 1, 10)
    
    # 7. Missing Values
    # Inject 5% missing values into Discount
    discounts = np.round(np.random.uniform(0, 30, num_rows), 1)
    mask = np.random.choice([True, False], size=num_rows, p=[0.05, 0.95])
    discounts[mask] = np.nan
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "Transaction_ID": [f"TRX-{i+1:06d}" for i in range(num_rows)],
        "Date": dates,
        "Category": category_choices,
        "Customer_Age": ages,
        "Unit_Price": prices,
        "Quantity": quantities,
        "Total_Revenue": revenues,
        "Discount_Pct": discounts,
        "Satisfaction_Score": satisfaction
    })
    
    # Sort by date for nicer time series
    df = df.sort_values("Date").reset_index(drop=True)
    
    file_path = f"datasets/{filename}"
    df.to_csv(file_path, index=False)
    print(f"Successfully saved to {file_path}")

if __name__ == "__main__":
    import os
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    generate_large_dataset()
