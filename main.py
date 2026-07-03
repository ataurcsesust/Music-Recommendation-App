import streamlit as st
from recommend import df, recommend_songs

st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="centered"
)

st.title("🎶 Instant Music Recommender")

song_list = sorted(df["song"].dropna().unique())

selected_song = st.selectbox(
    "Select a Song",
    song_list
)

if st.button("Recommend"):

    with st.spinner("Finding similar songs..."):

        recommendations = recommend_songs(selected_song)

    if recommendations is None:
        st.error("Song not found.")
    else:
        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=False
        )