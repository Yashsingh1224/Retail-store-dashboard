import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Online Retail Dashboard", layout="wide")
st.title("🛍️ Online Retail Customer Segmentation Dashboard")

# Load data
df = pd.read_excel("Online_retail.xlsx")
df.dropna(subset=['CustomerID'], inplace=True)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# RFM features
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Standardize
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = pca_components[:, 0]
rfm['PCA2'] = pca_components[:, 1]

# Sidebar filters
st.sidebar.header("Filters")
cluster_option = st.sidebar.selectbox("Select Cluster", options=["All"] + sorted(rfm['Cluster'].unique().tolist()))

# KPI metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(rfm))
col2.metric("Total Revenue", f"${df['TotalPrice'].sum():,.2f}")
col3.metric("Avg Order Size", f"${df.groupby('InvoiceNo')['TotalPrice'].sum().mean():.2f}")

# Cluster distribution
st.subheader("📈 Cluster Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=rfm, x='Cluster', ax=ax1)
st.pyplot(fig1)

# PCA scatter plot
st.subheader("🔍 Customer Segmentation (PCA View)")
fig2, ax2 = plt.subplots()
plot_data = rfm if cluster_option == "All" else rfm[rfm['Cluster'] == cluster_option]
sns.scatterplot(data=plot_data, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax2)
st.pyplot(fig2)

# Cluster summary table
st.subheader("📋 Cluster Summary")
st.dataframe(rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(2))

# Download segmented data
st.subheader("📥 Download Segmented Customer Data")
st.download_button("Download CSV", data=rfm.to_csv(index=False), file_name="segmented_customers.csv", mime="text/csv")

# Load the additional anomalous customer data
anomalous_data = pd.read_csv("anomalous_customers_categorized.csv")

# Merge all relevant columns
rfm_with_anomalies = pd.merge(rfm, anomalous_data, on='CustomerID', how='left')

# Display categorized anomalous customers
st.subheader("📊 Categorized Anomalous Customers")
anomalous_customers = rfm_with_anomalies[rfm_with_anomalies['AnomalyCategory'].notna()]
st.dataframe(anomalous_customers[[
    'CustomerID', 'TotalRevenue', 'TotalQuantity', 'NumTransactions', 'NumUniqueProducts', 'AnomalyCategory'
]])

# Optional: Download segmented anomalous customer data
st.subheader("📥 Download Categorized Anomalous Customer Data")
st.download_button("Download CSV", data=anomalous_customers.to_csv(index=False), file_name="categorized_anomalous_customers.csv", mime="text/csv")
