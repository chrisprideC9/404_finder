import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
from urllib.parse import urlparse
import numpy as np

# Function to extract path from URL
def extract_path(url):
    try:
        return urlparse(url).path.rstrip('/')
    except:
        return ''

# Function to strip path progressively
def strip_path(path):
    parts = path.split('/')
    # Remove the last segment iteratively
    while len(parts) > 1:
        parts.pop()
        new_path = '/'.join(parts)
        if not new_path.startswith('/'):
            new_path = '/' + new_path
        yield new_path

# Function to perform exact, fuzzy, and stripped path matching
def match_urls(old_df, new_df, similarity_threshold):
    # Extract paths
    old_df['Path'] = old_df['URL'].apply(extract_path)
    new_df['Path'] = new_df['URL'].apply(extract_path)

    # Exact Path Matching
    exact_matches = pd.merge(
        old_df, new_df, on='Path', suffixes=('_old', '_new')
    )
    exact_matches = exact_matches[['URL_old', 'URL_new', 'Path']].copy()
    exact_matches['Match_Type'] = 'Exact Path'
    exact_matches['Confidence_Score'] = 100  # Exact matches are highly confident

    # Identify URLs without exact matches
    old_no_exact = old_df[~old_df['URL'].isin(exact_matches['URL_old'])].copy()
    new_no_exact = new_df[~new_df['URL'].isin(exact_matches['URL_new'])].copy()

    # Fuzzy Path Matching
    fuzzy_matches = []

    for old_url, old_path in zip(old_no_exact['URL'], old_no_exact['Path']):
        # Find the best match in new paths
        match, score, _ = process.extractOne(
            old_path, new_no_exact['Path'], scorer=fuzz.ratio
        )
        if score >= similarity_threshold:
            # Retrieve the new URL corresponding to the matched path
            new_url = new_df[new_df['Path'] == match]['URL'].values[0]
            fuzzy_matches.append({
                'URL_old': old_url,
                'URL_new': new_url,
                'Path': match,
                'Match_Type': 'Fuzzy Path',
                'Confidence_Score': score  # Use the actual similarity score
            })

    fuzzy_matches_df = pd.DataFrame(fuzzy_matches)

    # Combine exact and fuzzy matches
    all_matches = pd.concat([
        exact_matches,
        fuzzy_matches_df[['URL_old', 'URL_new', 'Path', 'Match_Type', 'Confidence_Score']]
    ], ignore_index=True)

    # Identify unmatched old URLs
    matched_old_urls = all_matches['URL_old'].unique()
    unmatched_old = old_df[~old_df['URL'].isin(matched_old_urls)].copy()

    # Progressive Path Stripping Matching
    stripped_matches = []
    for idx, row in unmatched_old.iterrows():
        old_url = row['URL']
        old_path = row['Path']
        match_found = False

        for stripped_path in strip_path(old_path):
            # Attempt exact match with stripped path
            potential_matches = new_df[new_df['Path'] == stripped_path]
            if not potential_matches.empty:
                new_url = potential_matches['URL'].values[0]
                stripped_matches.append({
                    'URL_old': old_url,
                    'URL_new': new_url,
                    'Path': stripped_path,
                    'Match_Type': 'Stripped Path',
                    'Confidence_Score': 80  # Assign a fixed confidence score for stripped matches
                })
                match_found = True
                break  # Stop stripping once a match is found

        if not match_found:
            # Attempt fuzzy matching on stripped paths
            for stripped_path in strip_path(old_path):
                match, score, _ = process.extractOne(
                    stripped_path, new_df['Path'], scorer=fuzz.ratio
                )
                if score >= similarity_threshold:
                    new_url = new_df[new_df['Path'] == match]['URL'].values[0]
                    stripped_matches.append({
                        'URL_old': old_url,
                        'URL_new': new_url,
                        'Path': match,
                        'Match_Type': 'Fuzzy Stripped Path',
                        'Confidence_Score': score
                    })
                    match_found = True
                    break  # Stop stripping once a match is found

        if not match_found:
            # If no match is found even after stripping, mark as No Match
            stripped_matches.append({
                'URL_old': old_url,
                'URL_new': 'No Match Found',
                'Path': row['Path'],
                'Match_Type': 'No Match',
                'Confidence_Score': 0  # Indicate no confidence
            })

    stripped_matches_df = pd.DataFrame(stripped_matches)

    # Combine all matches
    all_matches = pd.concat([
        all_matches,
        stripped_matches_df[['URL_old', 'URL_new', 'Path', 'Match_Type', 'Confidence_Score']]
    ], ignore_index=True)

    return all_matches

# Streamlit App
def main():
    st.title("📄 Enhanced URL Matching Tool for Website Migration")

    st.markdown("""
    This tool assists in matching URLs for website migration by comparing old and new URLs based on their path structure using exact, fuzzy, and stripped path matching.
    """)

    st.header("1. Upload Required CSV Files")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_old = st.file_uploader("Upload Old URLs CSV", type=["csv"], key="old")
    with col2:
        uploaded_new = st.file_uploader("Upload New URLs CSV", type=["csv"], key="new")

    if uploaded_old and uploaded_new:
        try:
            old_df = pd.read_csv(uploaded_old)
            new_df = pd.read_csv(uploaded_new)

            # Validate required column
            if 'URL' not in old_df.columns or 'URL' not in new_df.columns:
                st.error("❌ Both CSV files must contain a 'URL' column.")
                return

            st.success("✅ CSV files uploaded and validated successfully.")

            st.header("2. Configure Matching Parameters")

            similarity_threshold = st.slider(
                "Set Path Similarity Threshold (%) for Fuzzy Matching",
                min_value=60, max_value=100, value=90, step=5
            )
            st.markdown(f"**Current Similarity Threshold:** {similarity_threshold}%")

            if st.button("🔄 Start Matching"):
                with st.spinner("🔄 Matching URLs..."):
                    matches = match_urls(old_df, new_df, similarity_threshold)

                st.success("✅ URL matching completed.")

                st.header("3. Matched URLs")

                if not matches.empty:
                    # Define color mapping for Match_Type
                    def highlight_match_type(val):
                        if val == 'Exact Path':
                            color = '#d3ffd3'  # Light green
                        elif val == 'Fuzzy Path':
                            color = '#ffe4b3'  # Light orange
                        elif val == 'Stripped Path':
                            color = '#cce5ff'  # Light blue
                        elif val == 'Fuzzy Stripped Path':
                            color = '#e6ccff'  # Light purple
                        elif val == 'No Match':
                            color = '#ffb3b3'  # Light red
                        else:
                            color = ''
                        return f'background-color: {color}'

                    # Apply conditional formatting
                    styled_df = matches.style.applymap(
                        highlight_match_type, subset=['Match_Type']
                    ).format({
                        'Confidence_Score': "{:.0f}"
                    })

                    st.dataframe(styled_df)

                    # Option to download the results
                    def convert_df(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv = convert_df(matches)

                    st.download_button(
                        label="📥 Download Matched URLs as CSV",
                        data=csv,
                        file_name='matched_urls.csv',
                        mime='text/csv',
                    )
                else:
                    st.warning("⚠️ No matches found based on the provided threshold.")
        except Exception as e:
            st.error(f"❌ An error occurred while processing the files: {e}")
    else:
        st.info("🛠️ Please upload both Old URLs and New URLs CSV files to begin the URL matching process.")

if __name__ == "__main__":
    main()
