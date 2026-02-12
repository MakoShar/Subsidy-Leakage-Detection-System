from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

def analyze_data():
    df = pd.read_csv("MP_Labour_DataSet.csv")

    district_df = df.groupby("district_name").agg({
        "Approved_Labour_Budget": "sum",
        "Total_Exp": "sum",
        "Number_of_Completed_Works": "sum",
        "percentage_payments_gererated_within_15_days": "mean"
    }).reset_index()

    # Financial anomaly
    district_df["exp_to_budget_ratio"] = (
        district_df["Total_Exp"] /
        (district_df["Approved_Labour_Budget"] + 1)
    )

    mean_ratio = district_df["exp_to_budget_ratio"].mean()
    std_ratio = district_df["exp_to_budget_ratio"].std()

    district_df["financial_anomaly"] = (
        abs((district_df["exp_to_budget_ratio"] - mean_ratio) / std_ratio) > 2
    ).astype(int)

    # Productivity anomaly
    district_df["exp_per_work"] = (
        district_df["Total_Exp"] /
        (district_df["Number_of_Completed_Works"] + 1)
    )

    mean_epw = district_df["exp_per_work"].mean()
    std_epw = district_df["exp_per_work"].std()

    district_df["productivity_anomaly"] = (
        abs((district_df["exp_per_work"] - mean_epw) / std_epw) > 2
    ).astype(int)

    # Composite risk
    district_df["risk_score"] = (
        district_df["financial_anomaly"] * 0.4 +
        district_df["productivity_anomaly"] * 0.4
    )

    district_df["high_risk"] = (
        district_df["risk_score"] >= 0.4
    ).astype(int)

    total = len(district_df)
    high_risk = district_df["high_risk"].sum()
    reduction = round((1 - high_risk / total) * 100, 2)

    high_risk_list = district_df[district_df["high_risk"] == 1]

    return df, total, high_risk, reduction, high_risk_list

@app.route("/")
def dashboard():
    df, total, high_risk, reduction, high_risk_list = analyze_data()

    preview_limit = 50
    df_preview = df.head(preview_limit).fillna("")
    return render_template(
        "dashboard.html",
        total=total,
        high_risk=high_risk,
        reduction=reduction,
        table=high_risk_list.to_dict(orient="records"),
        csv_filename="MP_Labour_DataSet.csv",
        csv_preview_count=len(df_preview),
        csv_columns=[str(c) for c in df_preview.columns.tolist()],
        csv_rows=df_preview.to_dict(orient="records"),
    )

if __name__ == "__main__":
    app.run(debug=True)