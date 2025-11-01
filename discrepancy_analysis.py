import pandas as pd
import igraph as ig
from tqdm import tqdm
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ===========================================
# 1. Matplotlib settings (English font)
# ===========================================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ===========================================
# 2. Load Data
# ===========================================
print("Loading data...")
nodes = pd.read_csv("musae_git_target.csv")
edges = pd.read_csv("musae_git_edges_fixed.csv")

print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")

# ===========================================
# 3. Build Graph (igraph)
# ===========================================
print("Building graph with igraph...")
g = ig.Graph.DataFrame(edges, directed=False, vertices=nodes)

# ===========================================
# 4. Compute Centrality Metrics (with tqdm)
# ===========================================
print("\nComputing centrality metrics (with progress bars)...")

metrics = {}

# ---- Degree ----
print(" -> Calculating Degree centrality...")
deg_values = g.degree()
for _ in tqdm(range(len(deg_values)), desc="Degree Centrality"):
    pass
metrics["degree_centrality"] = np.array(deg_values) / (len(g.vs) - 1)

# ---- Betweenness ----
print(" -> Calculating Betweenness centrality...")
btw_values = g.betweenness(vertices=None, directed=False)
for _ in tqdm(range(len(btw_values)), desc="Betweenness Centrality"):
    pass
metrics["betweenness"] = np.array(btw_values)

# ---- Closeness ----
print(" -> Calculating Closeness centrality...")
clo_values = g.closeness(vertices=None, normalized=True)
for _ in tqdm(range(len(clo_values)), desc="Closeness Centrality"):
    pass
metrics["closeness"] = np.array(clo_values)

# ---- Eigenvector ----
print(" -> Calculating Eigenvector centrality...")
eig_values = g.eigenvector_centrality(directed=False)
for _ in tqdm(range(len(eig_values)), desc="Eigenvector Centrality"):
    pass
metrics["eigenvector"] = np.array(eig_values)

# Assign metrics to dataframe
for k, v in metrics.items():
    nodes[k] = v

print("\n✅ Centrality calculation completed!")

# ===========================================
# 5. Group statistics
# ===========================================
metrics_list = ["degree_centrality", "betweenness", "closeness", "eigenvector"]
print("\n=== Mean centralities by ml_target ===")
group_means = nodes.groupby("ml_target")[metrics_list].mean()
print(group_means)

# ===========================================
# 6. Statistical tests
# ===========================================
print("\n=== Statistical Tests (t-test + Cohen's d) ===")

def cohen_d(x, y):
    nx_ = len(x)
    ny_ = len(y)
    dof = nx_ + ny_ - 2
    pooled_std = np.sqrt(((nx_ - 1) * np.var(x, ddof=1) + (ny_ - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

results = []
for metric in metrics_list:
    g0 = nodes.loc[nodes["ml_target"]==0, metric].dropna()
    g1 = nodes.loc[nodes["ml_target"]==1, metric].dropna()
    t_stat, p_val = ttest_ind(g0, g1, equal_var=False)
    d_val = cohen_d(g0, g1)
    results.append([metric, t_stat, p_val, d_val])

df_results = pd.DataFrame(results, columns=["Metric", "t-value", "p-value", "Cohen's d"])
print(df_results.to_string(index=False))

# ===========================================
# 7. Visualization (Violin Plots)
# ===========================================
print("\nGenerating plots...")
sns.set(style="whitegrid", font="Arial", font_scale=1.1)
for metric in metrics_list:
    plt.figure(figsize=(6,4))
    sns.violinplot(x="ml_target", y=metric, data=nodes, inner="quartile", palette="Set2")
    plt.title(f"{metric} distribution by ml_target")
    plt.xlabel("ml_target")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"discrepancy_analysis/{metric}.png", dpi=300)
    plt.close()
print("  ✓ Plots saved")

# ===========================================
# 8. PDF Report (English)
# ===========================================
print("\nGenerating PDF report...")

doc = SimpleDocTemplate("discrepancy_analysis/network_analysis_report_igraph.pdf", pagesize=A4)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Social Network Analysis Report (igraph)", styles["Title"]))
story.append(Spacer(1,12))
story.append(Paragraph(f"Nodes: {len(nodes)}, Edges: {len(edges)}", styles["Normal"]))
story.append(Spacer(1,12))

# Group means
story.append(Paragraph("1. Centrality metrics by ml_target", styles["Heading2"]))
tbl_data = [group_means.columns.tolist()] + group_means.reset_index().values.tolist()
table = Table(tbl_data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
    ('GRID', (0,0), (-1,-1), 0.25, colors.grey)
]))
story.append(table)
story.append(Spacer(1,12))

# Statistical results
story.append(Paragraph("2. Statistical test results (t-test + Cohen's d)", styles["Heading2"]))
tbl_data2 = [df_results.columns.tolist()] + df_results.values.tolist()
table2 = Table(tbl_data2)
table2.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.lightgreen),
    ('GRID', (0,0), (-1,-1), 0.25, colors.grey)
]))
story.append(table2)
story.append(Spacer(1,12))

# Plots
story.append(Paragraph("3. Violin Plots", styles["Heading2"]))
for metric in metrics_list:
    story.append(Paragraph(metric, styles["Heading3"]))
    story.append(RLImage(f"{metric}.png", width=400, height=300))
    story.append(Spacer(1,12))

doc.build(story)
print("✅ PDF report generated: network_analysis_report_igraph.pdf")

print("\n🎯 All tasks completed successfully.")