import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set dark background for Matplotlib to match Streamlit's dark theme
plt.style.use('dark_background')

# 1. Page Configuration
st.set_page_config(page_title="Netflix Dashboard", page_icon="🎬", layout="wide")

# Load and clean data
df = pd.read_csv('netflix_titles.csv.zip')
df['country'] = df['country'].fillna('Unknown')

# ----------------- SIDEBAR UI -----------------
# Netflix logo and filters in the sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.title("Filters ⚙️")
st.sidebar.write("Customize the data view:")

# Country selection dropdown
country_list = df['country'].unique().tolist()
selected_country = st.sidebar.selectbox("Select a Country", ["Worldwide (All)"] + country_list)

# ----------------- MAIN PAGE UI -----------------
st.title("🎬 Netflix Data Analysis Dashboard")
st.markdown("---") # Horizontal separator line

# Filter data based on selected country
if selected_country != "Worldwide (All)":
    df = df[df['country'] == selected_country]

# Split the screen into two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Data Overview: {selected_country}")
    st.write(f"**Total Movies & Shows:** {len(df)}")
    
    # Checkbox to display raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(df[['title', 'type', 'release_year']].head())

with col2:
    st.subheader("Movies vs TV Shows")
    
    # Generate the bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='type', data=df, palette='Set1', ax=ax)
    
    # Display the chart
    st.pyplot(fig)

# ----------------- AI RECOMMENDATION SYSTEM -----------------
st.markdown("---")
st.header("🤖 AI Movie Recommender")
st.write("Select a movie or show, and our AI will suggest 5 similar titles based on their descriptions!")

# Clean the description column to avoid NLP errors
df['description'] = df['description'].fillna('')

# 1. AI Logic (NLP Content-Based Filtering)
@st.cache_data # Caches the data to make the app run faster
def get_recommendations(title, dataframe):
    # TF-IDF algorithm converts text descriptions into numerical vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['description'])
    
    # Cosine Similarity calculates the mathematical distance between titles
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Find the index of the user-selected movie
    idx = dataframe[dataframe['title'] == title].index[0]
    
    # Calculate similarity scores for all other movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Extract the top 5 most similar movies (skipping the 1st one, which is the movie itself)
    movie_indices = [i[0] for i in sim_scores[1:6]]
    return dataframe['title'].iloc[movie_indices]

# 2. User Input
movie_list = df['title'].unique().tolist()
selected_movie = st.selectbox("Choose a Movie or TV Show:", movie_list)

# 3. Display Recommendations on Button Click
if st.button("Get Recommendations 🚀"):
    st.write(f"**Top 5 recommendations similar to '{selected_movie}':**")
    
    # Call the ML function
    recommendations = get_recommendations(selected_movie, df)
    
    # Print the results
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")