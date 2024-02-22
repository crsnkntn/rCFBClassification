import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# NOTE: filter out team names
# NOTE: 

# ROUND 1: banned_words = {"et", "am", "pm", "football", "vs", "game", "thread", "state", "field", "stadium"}
# ROUND 2:
banned_words = {"week"}
# NOTE: This will change for future datasets
# NOTE: add all teams to this
team_names = {"michigan", "ohiostate", "georgia", "oklahoma", "texas", "floridastate", "oregon", "alabama", "notredame", "iowa"}

stop_words.update(banned_words)
stop_words.update(team_names)
MAX_WORD_LEN = 16


def clean_csv_files(csv_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each CSV file in the directory
    for file_name in os.listdir(csv_dir):
        if file_name.endswith(".csv"):
            # Construct the full path to the CSV file
            file_path = os.path.join(csv_dir, file_name)

            # Read CSV file into DataFrame
            df = pd.read_csv(file_path)

            # Clean comment_body column
            df['comment_body'] = df['comment_body'].apply(lambda x: clean_text(str(x), stop_words))
            df['post_title'] = df['post_title'].apply(lambda x: clean_text(str(x), stop_words))
            df['post_selftext'] = df['post_selftext'].apply(lambda x: clean_text(str(x), stop_words))

            # Save cleaned DataFrame to a new CSV file
            output_file_path = os.path.join(output_dir, file_name)
            df.to_csv(output_file_path, index=False)

            print(f"CSV file cleaned and saved to: {output_file_path}")


def clean_text(text, stop_words):
    # Remove stopwords
    words = text.split()

    filtered_words = [word for word in words if word not in stop_words and len(word) < MAX_WORD_LEN]

    # Join filtered words back into a single string
    text = ' '.join(filtered_words)

    return text

# Example usage:
csv_dir = "datasets/combined_classdata_csv"
output_dir = "datasets/cleaned_combined_classdata_csv"

clean_csv_files(csv_dir, output_dir)
