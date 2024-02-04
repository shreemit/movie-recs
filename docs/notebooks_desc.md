# Adding TMDb ID to Movie Data

# Notebooks
- `linking.ipynb`
- `data_preproc.ipynb`
- 'collab_filter.ipynb'
- 'content_based.ipynb'
- 'evaluation.ipynb'

## Notebook: `linking.ipynb`

This notebook integrates data from multiple sources (links, movies, IMDb, TMDb) to create a comprehensive movie dataset with consistent formatting and TMDb IDs.
### Key steps:
1. Load and prepare data:

    - Imports necessary libraries (pandas, re).
    - Loads data from CSV files.
    - Renames columns for consistency.

2. Merges datasets:
    - Combine given movie data with link data and IMDb data.
    - Extract year and cleans movie titles.
    - Merge TMDb data with the main movie data.
3. Handle missing data:
    - Identifies and addresses missing values.
4. Save the merged data.
    - Creates a merged CSV file with TMDb IDs (merge_movies.csv).

The main utility of adding TMDb IDs is that they can be used with the TMDb API to fetch additional movie data.

## Notebook: `data_preproc.ipynb`
