import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Website Ka Title
st.title("🎬 Netflix Data Analysis Dashboard")
st.write("This is my first Python Web App for analyzing Netflix data.")

# 2. Data Load Karna
# (Dhyan rakhein ki 'netflix_titles.csv' aapki app.py wale folder mein hi ho)
df = pd.read_csv('netflix_titles.csv.zip')

# 3. Data Table Dikhana (Checkbox ke saath)
if st.checkbox("Raw Data Dekhein"):
    st.write(df.head())

# 4. Graph Banana (Movies vs TV Shows)
st.subheader("What is more on Netflix: Movies or TV Shows?")

fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(x='type', data=df, palette='Set2', ax=ax)

# Graph ko app par dikhana
st.pyplot(fig)