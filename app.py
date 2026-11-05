 # CarDekho Flask Web App
# Run:  python app.py   then open  http://127.0.0.1:5000
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # draw charts without opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request

here = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# ---------- 1. LOAD DATA ----------
csv_file = os.path.join(here, "cardekho_dataset.csv")
if not os.path.exists(csv_file):
    raise SystemExit("Put cardekho_dataset.csv in the same folder as app.py")

df = pd.read_csv(csv_file)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
df = df.drop_duplicates().dropna().reset_index(drop=True)
df["price_lakh"] = (df["selling_price"] / 100000).round(2)

# ---------- 2. MAKE CHARTS (once, when the app starts) ----------
chart_dir = os.path.join(here, "static", "charts")
os.makedirs(chart_dir, exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, name), dpi=110)
    plt.close()

def make_charts():
    plt.figure(figsize=(8, 5))
    plt.hist(df["price_lakh"], bins=50, color="steelblue")
    plt.title("Price Distribution"); plt.xlabel("Price (Lakh)"); plt.ylabel("Cars")
    save("price_distribution.png")

    plt.figure(figsize=(8, 5))
    df["brand"].value_counts().head(10).sort_values().plot(kind="barh", color="orange")
    plt.title("Top 10 Brands"); plt.xlabel("Number of cars")
    save("top_brands.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="fuel_type", y="price_lakh", data=df, showfliers=False)
    plt.title("Price by Fuel Type"); plt.ylabel("Price (Lakh)")
    save("price_by_fuel.png")

    plt.figure(figsize=(8, 5))
    df.groupby("vehicle_age")["price_lakh"].mean().plot(marker="o", color="crimson")
    plt.title("Average Price vs Vehicle Age"); plt.xlabel("Age (years)"); plt.ylabel("Price (Lakh)")
    save("price_vs_age.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(df["max_power"], df["price_lakh"], alpha=0.3, s=8)
    plt.title("Price vs Max Power"); plt.xlabel("Max power (bhp)"); plt.ylabel("Price (Lakh)")
    save("price_vs_power.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=np.number).drop(columns=["price_lakh"]).corr(),
                annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save("heatmap.png")

make_charts()

# ---------- 3. PRICE MODEL (simple linear regression with numpy) ----------
num_cols = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
fuels = sorted(df["fuel_type"].unique())
trans = sorted(df["transmission_type"].unique())

def build_features(frame):
    X = frame[num_cols].astype(float).copy()
    for f in fuels[1:]:
        X["fuel_" + f] = (frame["fuel_type"] == f).astype(float)
    for t in trans[1:]:
        X["trans_" + t] = (frame["transmission_type"] == t).astype(float)
    return X

X_all = build_features(df)
mean, std = X_all.mean(), X_all.std().replace(0, 1)
A = np.c_[np.ones(len(X_all)), ((X_all - mean) / std).to_numpy()]
y = np.log1p(df["selling_price"].to_numpy(dtype=float))
weights = np.linalg.lstsq(A, y, rcond=None)[0]
pred_all = A @ weights
r2 = 1 - ((y - pred_all) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def predict_price(row):
    X = build_features(pd.DataFrame([row]))
    a = np.r_[1.0, ((X - mean) / std).to_numpy()[0]]
    return float(np.expm1(a @ weights))

# ---------- 4. PAGES ----------
@app.route("/")
def home():
    stats = {
        "cars": f"{len(df):,}",
        "brands": df["brand"].nunique(),
        "median": f"{df['price_lakh'].median():.2f}",
        "top_brand": df["brand"].mode()[0],
        "priciest": df.groupby("brand")["price_lakh"].mean().idxmax(),
        "avg_age": f"{df['vehicle_age'].mean():.1f}",
    }
    fuel = df.groupby("fuel_type")["price_lakh"].mean().round(2).to_dict()
    trn = df.groupby("transmission_type")["price_lakh"].mean().round(2).to_dict()
    return render_template("home.html", stats=stats, fuel=fuel, trn=trn, page="home")

@app.route("/charts")
def charts():
    items = [
        ("price_distribution.png", "Most cars sell below 10 lakh; a few luxury cars stretch the right side."),
        ("top_brands.png", "Maruti, Hyundai and Honda dominate the listings."),
        ("price_by_fuel.png", "Diesel and electric cars cost more than petrol; LPG is cheapest."),
        ("price_vs_age.png", "Price falls steadily as the car gets older."),
        ("price_vs_power.png", "More engine power means a higher price."),
        ("heatmap.png", "Max power and engine size are the strongest drivers of price."),
    ]
    return render_template("charts.html", items=items, page="charts")

@app.route("/cars")
def cars():
    brand = request.args.get("brand", "")
    fuel = request.args.get("fuel", "")
    max_price = request.args.get("max_price", "")
    data = df
    if brand:
        data = data[data["brand"] == brand]
    if fuel:
        data = data[data["fuel_type"] == fuel]
    if max_price:
        try:
            data = data[data["price_lakh"] <= float(max_price)]
        except ValueError:
            pass
    shown = data.sort_values("price_lakh", ascending=False).head(100)
    cols = ["car_name", "brand", "vehicle_age", "km_driven", "fuel_type",
            "transmission_type", "mileage", "engine", "max_power", "seats", "price_lakh"]
    return render_template("cars.html", rows=shown[cols].values.tolist(),
                           total=len(data), brands=sorted(df["brand"].unique()),
                           fuels=fuels, sel_brand=brand, sel_fuel=fuel,
                           sel_price=max_price, page="cars")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result, form = None, {}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            row = {c: float(form[c]) for c in num_cols}
            row["fuel_type"] = form["fuel_type"]
            row["transmission_type"] = form["transmission_type"]
            lakh = predict_price(row) / 100000
            result = f"{max(lakh, 0.3):.2f}"
        except (ValueError, KeyError):
            result = "error"
    return render_template("predict.html", result=result, form=form, fuels=fuels,
                           trans=trans, r2=f"{r2:.2f}", page="predict")

if __name__ == "__main__":
    app.run(debug=True)
