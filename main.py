import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
from urllib.parse import urlparse
import numpy as np
import re

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

# Function to determine if a URL is internal based on the base domain
def is_internal(url, base_domain):
    try:
        return urlparse(url).netloc == base_domain
    except:
        return False

# Function to filter out URLs based on a list of patterns
def filter_urls(data_df, filter_patterns):
    """
    Excludes URLs that match any of the patterns in filter_patterns.

    Parameters:
    - data_df (pd.DataFrame): The DataFrame containing URLs.
    - filter_patterns (list): List of regex patterns to filter out.

    Returns:
    - filtered_df (pd.DataFrame): DataFrame after excluding filtered URLs.
    - excluded_df (pd.DataFrame): DataFrame of excluded URLs.
    """
    # Compile regex patterns for efficiency
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in filter_patterns]
    
    # Function to check if a URL matches any of the patterns
    def matches_filter(url):
        return any(pattern.search(url) for pattern in compiled_patterns)
    
    # Apply the filter
    excluded_df = data_df[data_df['Address'].apply(matches_filter)].copy()
    filtered_df = data_df[~data_df['Address'].apply(matches_filter)].copy()
    
    return filtered_df, excluded_df

# Function to perform exact, fuzzy, and stripped path matching for internal redirects
def match_internal_redirects(data_df, similarity_threshold, base_domain):
    # Filter URLs based on status codes
    old_df = data_df[data_df['Status Code'] == 404].copy()
    new_df = data_df[data_df['Status Code'] == 200].copy()

    # Ensure only internal URLs are considered
    old_df = old_df[old_df['Address'].apply(lambda x: is_internal(x, base_domain))]
    new_df = new_df[new_df['Address'].apply(lambda x: is_internal(x, base_domain))]

    # Extract paths
    old_df['Path'] = old_df['Address'].apply(extract_path)
    new_df['Path'] = new_df['Address'].apply(extract_path)

    # Exact Path Matching
    exact_matches = pd.merge(
        old_df, new_df, on='Path', suffixes=('_old', '_new')
    )
    exact_matches = exact_matches[['Address_old', 'Address_new', 'Path']].copy()
    exact_matches['Match_Type'] = 'Exact Path'
    exact_matches['Confidence_Score'] = 100  # Exact matches are highly confident

    # Identify URLs without exact matches
    matched_old_urls = exact_matches['Address_old'].unique()
    old_no_exact = old_df[~old_df['Address'].isin(matched_old_urls)].copy()

    # Fuzzy Path Matching
    fuzzy_matches = []

    for _, row in old_no_exact.iterrows():
        old_url = row['Address']
        old_path = row['Path']
        # Find the best match in new paths
        match, score, _ = process.extractOne(
            old_path, new_df['Path'], scorer=fuzz.ratio
        )
        if score >= similarity_threshold:
            # Retrieve the new URL corresponding to the matched path
            new_url = new_df[new_df['Path'] == match]['Address'].values[0]
            fuzzy_matches.append({
                'Address_old': old_url,
                'Address_new': new_url,
                'Path': match,
                'Match_Type': 'Fuzzy Path',
                'Confidence_Score': score  # Use the actual similarity score
            })

    fuzzy_matches_df = pd.DataFrame(fuzzy_matches)

    # Combine exact and fuzzy matches
    all_matches = pd.concat([
        exact_matches,
        fuzzy_matches_df[['Address_old', 'Address_new', 'Path', 'Match_Type', 'Confidence_Score']]
    ], ignore_index=True)

    # Identify unmatched old URLs
    matched_old_urls = all_matches['Address_old'].unique()
    unmatched_old = old_df[~old_df['Address'].isin(matched_old_urls)].copy()

    # Progressive Path Stripping Matching
    stripped_matches = []
    for _, row in unmatched_old.iterrows():
        old_url = row['Address']
        old_path = row['Path']
        match_found = False

        for stripped_path in strip_path(old_path):
            # Attempt exact match with stripped path
            potential_matches = new_df[new_df['Path'] == stripped_path]
            if not potential_matches.empty:
                new_url = potential_matches['Address'].values[0]
                stripped_matches.append({
                    'Address_old': old_url,
                    'Address_new': new_url,
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
                    new_url = new_df[new_df['Path'] == match]['Address'].values[0]
                    stripped_matches.append({
                        'Address_old': old_url,
                        'Address_new': new_url,
                        'Path': match,
                        'Match_Type': 'Fuzzy Stripped Path',
                        'Confidence_Score': score
                    })
                    match_found = True
                    break  # Stop stripping once a match is found

        if not match_found:
            # If no match is found even after stripping, mark as No Match
            stripped_matches.append({
                'Address_old': old_url,
                'Address_new': 'No Match Found',
                'Path': row['Path'],
                'Match_Type': 'No Match',
                'Confidence_Score': 0  # Indicate no confidence
            })

    stripped_matches_df = pd.DataFrame(stripped_matches)

    # Combine all matches
    all_matches = pd.concat([
        all_matches,
        stripped_matches_df[['Address_old', 'Address_new', 'Path', 'Match_Type', 'Confidence_Score']]
    ], ignore_index=True)

    return all_matches

# Streamlit App
def main():
    st.set_page_config(page_title="🔄 Internal Redirect Finder", layout="wide")
    st.title("🔄 Internal Redirect Finder for 404 Errors")

    st.markdown("""
    This tool identifies internal redirects for URLs that return a 404 status code within your site. It filters out common non-IA URLs and matches the remaining URLs to existing internal URLs that return a 200 status code using exact, fuzzy, and stripped path matching.
    """)

    st.header("1. Upload Your URLs CSV")

    uploaded_file = st.file_uploader("Upload CSV of HTML export from Screaming Frog", type=["csv"])

    if uploaded_file:
        try:
            data_df = pd.read_csv(uploaded_file)

            # Validate required columns
            if 'Address' not in data_df.columns or 'Status Code' not in data_df.columns:
                st.error("❌ The CSV file must contain 'Address' and 'Status Code' columns.")
                return

            st.success("✅ CSV file uploaded and validated successfully.")

            # Define filter patterns (basic list)
            filter_patterns = [
                r'\.jpg$',      # URLs ending with .jpg
                r'\.jpeg$',     # URLs ending with .jpeg
                r'\.png$',      # URLs ending with .png
                r'\.gif$',      # URLs ending with .gif
                r'\.css$',      # URLs ending with .css
                r'\.js$',       # URLs ending with .js
                r'\.atom$',     # URLs ending with .atom
                r'/cache/',     # URLs containing /cache/
                r'\.pdf$',      # URLs ending with .pdf
                r'\.ico$',      # URLs ending with .ico
                r'\.svg$',      # URLs ending with .svg
                r'\.mp3$',      # URLs ending with .mp3
                r'\.mp4$',      # URLs ending with .mp4
                r'\.zip$',      # URLs ending with .zip
                r'\.tar$',      # URLs ending with .tar
                r'\.gz$',       # URLs ending with .gz
                r'/images/',    # URLs containing /images/
                r'/assets/',    # URLs containing /assets/
                r'/scripts/',   # URLs containing /scripts/
                r'/styles/',    # URLs containing /styles/
                r'\.json$',     # URLs ending with .json
                r'\.xml$',      # URLs ending with .xml
                r'\.txt$',      # URLs ending with .txt
                r'\.md$',       # URLs ending with .md
                r'\.exe$',      # URLs ending with .exe
                r'\.bat$',      # URLs ending with .bat
            ]

            st.header("2. Filter Non-IA URLs")

            st.markdown("""
            The following filters are applied to exclude common non-IA URLs such as static assets, feeds, and cache-related resources. These patterns are based on file extensions and specific path segments.
            """)

            # Display the filter patterns (optional)
            st.subheader("🔍 Applied Filter Patterns:")
            st.write(", ".join([pattern.replace("\\", "") for pattern in filter_patterns]))

            # Optionally, allow users to add/remove patterns (currently commented out for simplicity)
            # st.subheader("🛠️ Customize Filter Patterns")
            # user_patterns = st.text_area("Add custom regex patterns (one per line):").splitlines()
            # if user_patterns:
            #     filter_patterns.extend(user_patterns)

            # Apply filtering
            filtered_df, excluded_df = filter_urls(data_df, filter_patterns)

            st.markdown(f"**Total URLs Uploaded:** {len(data_df)}")
            st.markdown(f"**URLs Excluded by Filters:** {len(excluded_df)}")
            st.markdown(f"**URLs to be Processed:** {len(filtered_df)}")

            # Optionally, display excluded URLs (collapsed)
            with st.expander("📂 View Excluded URLs"):
                st.dataframe(excluded_df[['Address', 'Status Code']].reset_index(drop=True))

            st.header("3. Configure Matching Parameters")

            # Extract base domain from the first URL to determine internal redirects
            if not filtered_df['Address'].dropna().empty:
                sample_url = filtered_df['Address'].dropna().iloc[0]
                base_domain = urlparse(sample_url).netloc
                st.write(f"**Assumed Base Domain:** `{base_domain}`")
            else:
                st.error("❌ No URLs available for processing after applying filters.")
                return

            similarity_threshold = st.slider(
                "Set Path Similarity Threshold (%) for Fuzzy Matching",
                min_value=60, max_value=100, value=90, step=5
            )
            st.markdown(f"**Current Similarity Threshold:** {similarity_threshold}%")

            if st.button("🔄 Start Finding Internal Redirects"):
                with st.spinner("🔄 Matching internal redirects..."):
                    matches = match_internal_redirects(filtered_df, similarity_threshold, base_domain)

                st.success("✅ Internal redirect matching completed.")

                st.header("4. Matched Internal Redirects")

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

                    st.dataframe(styled_df, use_container_width=True)

                    # Option to download the results
                    def convert_df(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv = convert_df(matches)

                    st.download_button(
                        label="📥 Download Internal Redirects as CSV",
                        data=csv,
                        file_name='internal_redirects.csv',
                        mime='text/csv',
                    )
                else:
                    st.warning("⚠️ No redirects found based on the provided threshold.")
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")
    else:
        st.info("🛠️ Please upload a CSV file containing 'Address' and 'Status Code' columns to begin.")

if __name__ == "__main__":
    main()


st.markdown("---")
st.markdown("© 2025 Calibre Nine | [GitHub Repository](https://github.com/chrisprideC9)")