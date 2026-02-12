import pandas as pd
from sklearn.metrics import classification_report
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# LOAD DATA
df = pd.read_excel("synthetic_subsidy_data.xlsx")
print(df.head())

bank_counts = df.groupby("bank_account").size().reset_index(name="count")

# Flag suspicious accounts (more than 3 beneficiaries)
suspicious_accounts = bank_counts[bank_counts["count"] > 3]


# Merge back
df["shared_account_flag"] = df["bank_account"].isin(
    suspicious_accounts["bank_account"]
).astype(int)

print(df[["bank_account", "shared_account_flag"]].head(15))
officer_counts = df.groupby("officer_id").size().reset_index(name="count")

suspicious_officers = officer_counts[officer_counts["count"] > 15]

df["suspicious_officer_flag"] = df["officer_id"].isin(
    suspicious_officers["officer_id"]
).astype(int)

df["high_income_flag"] = (df["income"] > 600000).astype(int)

df["rule_risk_score"] = (
    df["shared_account_flag"] * 0.4 +
    df["suspicious_officer_flag"] * 0.3 +
    df["high_income_flag"] * 0.3
)

df["predicted_fraud"] = (df["rule_risk_score"] >= 0.3).astype(int)

print(df[["fraud_flag", "predicted_fraud"]].head(20))

print(confusion_matrix(df["fraud_flag"], df["predicted_fraud"]))

print(classification_report(df["fraud_flag"], df["predicted_fraud"]))

#pichart info
fraud_amount = df[df["predicted_fraud"] == 1]["subsidy_amount"].sum()
total_amount = df["subsidy_amount"].sum()
print("Total Distributed:", total_amount)
print("Amount At Risk:", fraud_amount)
print("Leakage %:", (fraud_amount/total_amount)*100)


"""
G = nx.Graph()

suspicious_df = df[df["predicted_fraud"] == 1]

for _, row in suspicious_df.iterrows():
    beneficiary = f"B_{row['beneficiary_id']}"
    bank = f"BA_{row['bank_account']}"
    officer = f"O_{row['officer_id']}"

    G.add_node(beneficiary, type="beneficiary")
    G.add_node(bank, type="bank")
    G.add_node(officer, type="officer")

    G.add_edge(beneficiary, bank)
    G.add_edge(beneficiary, officer)

plt.figure(figsize=(10,8))
pos = nx.spring_layout(G, seed=42)

node_colors = []
for node in G.nodes(data=True):
    if node[1]["type"] == "beneficiary":
        node_colors.append("blue")
    elif node[1]["type"] == "bank":
        node_colors.append("red")
    else:
        node_colors.append("green")

nx.draw(G, pos, node_size=150, node_color=node_colors, with_labels=False)
plt.title("Detected Fraud Network Cluster")
plt.show()"""

# Make pie
plt.figure()
plt.pie(
    [fraud_amount, total_amount - fraud_amount],
    labels=["At Risk", "Clean"],
    autopct="%1.1f%%"
)
plt.title("Estimated Subsidy Leakage")
plt.show()