"""
Generates a synthetic global e-commerce sales dataset (2023-01-01 to 2024-12-31).
Deliberately includes realistic data-quality issues (missing values, duplicates,
inconsistent text casing, a few negative quantities, outlier prices) so that the
analysis notebook has genuine cleaning work to demonstrate.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 7000

countries_regions = {
    "United States": "North America", "Canada": "North America",
    "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
    "Spain": "Europe", "Italy": "Europe",
    "India": "Asia", "Japan": "Asia", "Singapore": "Asia",
    "Australia": "Oceania", "Brazil": "South America", "Mexico": "South America",
}
countries = list(countries_regions.keys())
country_weights = np.array([0.22, 0.06, 0.14, 0.09, 0.08, 0.05, 0.04,
                             0.09, 0.05, 0.03, 0.05, 0.06, 0.04])
country_weights = country_weights / country_weights.sum()

categories = {
    "Electronics": ["Headphones", "Smartphone Case", "Bluetooth Speaker", "Laptop Stand", "USB-C Cable"],
    "Home & Kitchen": ["Coffee Maker", "Cutting Board", "Blender", "Storage Bin", "Throw Pillow"],
    "Apparel": ["T-Shirt", "Running Shoes", "Denim Jacket", "Wool Socks", "Baseball Cap"],
    "Beauty": ["Face Serum", "Shampoo", "Lip Balm", "Sunscreen SPF50", "Hair Dryer"],
    "Sports & Outdoors": ["Yoga Mat", "Water Bottle", "Camping Tent", "Resistance Bands", "Hiking Backpack"],
    "Office": ["Notebook Set", "Desk Organizer", "Wireless Mouse", "Standing Desk Mat", "Pen Pack"],
}
cat_list = list(categories.keys())
cat_weights = np.array([0.24, 0.18, 0.20, 0.14, 0.13, 0.11])
cat_weights = cat_weights / cat_weights.sum()

base_price = {
    "Electronics": (15, 220), "Home & Kitchen": (10, 150), "Apparel": (8, 90),
    "Beauty": (5, 60), "Sports & Outdoors": (10, 180), "Office": (4, 70),
}

segments = ["Consumer", "Small Business", "Enterprise"]
seg_weights = [0.6, 0.28, 0.12]

payment_methods = ["Credit Card", "PayPal", "Debit Card", "Gift Card", "Bank Transfer"]
pay_weights = [0.42, 0.24, 0.18, 0.06, 0.10]

statuses = ["Completed", "Returned", "Cancelled"]
status_weights = [0.88, 0.08, 0.04]

n_customers = 850
customer_ids = [f"CUST-{i:05d}" for i in range(1, n_customers + 1)]
first_names = ["James","Mary","Robert","Patricia","John","Jennifer","Michael","Linda","David","Elizabeth",
               "Wei","Priya","Ahmed","Sofia","Liam","Emma","Noah","Olivia","Lucas","Mia","Yuki","Carlos",
               "Ana","Hiro","Fatima","Chen","Raj","Ines","Tom","Grace"]
last_names = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
              "Kumar","Zhang","Silva","Muller","Tanaka","Khan","Costa","Rossi","Dubois","Nakamura"]

customer_lookup = {}
for cid in customer_ids:
    name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
    seg = rng.choice(segments, p=seg_weights)
    country = rng.choice(countries, p=country_weights)
    customer_lookup[cid] = (name, seg, country)

start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp("2024-12-31")
date_range_days = (end_date - start_date).days

rows = []
for i in range(1, N + 1):
    order_id = f"ORD-{100000 + i}"
    cid = rng.choice(customer_ids)
    name, seg, country = customer_lookup[cid]
    region = countries_regions[country]

    # seasonal weighting: bias more orders towards Nov-Dec (holiday) and slight yearly growth
    offset = int(rng.triangular(0, date_range_days * 0.85, date_range_days))
    order_date = start_date + pd.Timedelta(days=offset)

    cat = rng.choice(cat_list, p=cat_weights)
    product = rng.choice(categories[cat])
    lo, hi = base_price[cat]
    unit_price = round(float(rng.uniform(lo, hi)), 2)

    qty = int(rng.choice([1, 1, 1, 2, 2, 3, 4, 5], p=[0.32,0.18,0.15,0.14,0.09,0.06,0.04,0.02]))
    discount = float(rng.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20], p=[0.45,0.15,0.1,0.1,0.1,0.06,0.04]))
    ship_cost = round(float(rng.uniform(2, 25)), 2)
    pay = rng.choice(payment_methods, p=pay_weights)
    status = rng.choice(statuses, p=status_weights)

    rows.append([order_id, order_date, cid, name, seg, country, region,
                 cat, product, qty, unit_price, discount, ship_cost, pay, status])

df = pd.DataFrame(rows, columns=[
    "OrderID", "OrderDate", "CustomerID", "CustomerName", "Segment", "Country", "Region",
    "Category", "Product", "Quantity", "UnitPrice", "Discount", "ShippingCost",
    "PaymentMethod", "OrderStatus"
])

# ---------- Inject realistic messiness ----------

# 1. Missing CustomerName (~3%) and Segment (~4%)
mask = rng.random(len(df)) < 0.03
df.loc[mask, "CustomerName"] = np.nan
mask = rng.random(len(df)) < 0.04
df.loc[mask, "Segment"] = np.nan

# 2. Inconsistent country casing / whitespace (~5%)
mask = rng.random(len(df)) < 0.05
df.loc[mask, "Country"] = df.loc[mask, "Country"].str.upper()
mask2 = rng.random(len(df)) < 0.03
df.loc[mask2, "Country"] = " " + df.loc[mask2, "Country"].str.lower() + " "

# 3. A handful of negative / zero quantities (data entry errors) (~1%)
mask = rng.random(len(df)) < 0.01
df.loc[mask, "Quantity"] = -df.loc[mask, "Quantity"]

# 4. A few extreme unit price outliers (~0.5%)
mask = rng.random(len(df)) < 0.005
df.loc[mask, "UnitPrice"] = df.loc[mask, "UnitPrice"] * rng.uniform(15, 30)

# 5. Duplicate rows (~1.5% exact duplicates)
dup_frac = 0.015
n_dupes = int(len(df) * dup_frac)
dupe_rows = df.sample(n=n_dupes, random_state=1)
df = pd.concat([df, dupe_rows], ignore_index=True)

# 6. Missing ShippingCost (~2%)
mask = rng.random(len(df)) < 0.02
df.loc[mask, "ShippingCost"] = np.nan

# Shuffle rows so duplicates aren't obviously stacked at the end
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("/home/claude/project/ecommerce_sales_raw.csv", index=False)
print("Rows:", len(df))
print(df.head())
print(df.isna().sum())
