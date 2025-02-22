import os
import pandas as pd

def read_imdb_data(folder_path):
    data = []
    for sentiment in ['pos', 'neg']:
        sentiment_path = os.path.join(folder_path, sentiment)
        for filename in os.listdir(sentiment_path):
            if filename.endswith('.txt'):
                with open(os.path.join(sentiment_path, filename), 'r', encoding='utf-8') as file:
                    review = file.read()
                    data.append((review, sentiment))
    return data

# Define the folder path
folder_path = 'aclImdb/train'

# Read the data
imdb_data = read_imdb_data(folder_path)

# Create a DataFrame
df = pd.DataFrame(imdb_data, columns=['review', 'sentiment'])

# Save to CSV
df.to_csv('imdb_reviews.csv', index=False)


