import pandas as pd
import ast
from flair.models import TextClassifier
from flair.data import Sentence

sia = TextClassifier.load('en-sentiment')

def preprocess_data():
    df = pd.read_csv('datasets/professors_dataset.csv')
    
    df = df[df['type'] == 'professor']
    df = df.drop(columns=['type'])
    
    df['courses'] = df['courses'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['num_courses'] = df['courses'].apply(len)
    
    df['reviews'] = df['reviews'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['num_reviews'] = df['reviews'].apply(len)
    df = df[df['num_reviews'] >= 10]
    
    def extract_review_text(reviews):
        return [review['review'] for review in reviews]
    
    df['review_texts'] = df['reviews'].apply(extract_review_text)
    
    def flair_prediction(x):
        sentence = Sentence(x)
        sia.predict(sentence)
        score = sentence.labels[0]
        if score.value == 'POSITIVE':
            return score.score
        else:
            return -score.score
    
    def get_sentiment(reviews):
        scores = []
        for review in reviews:
            if not review or not isinstance(review, str):
                continue
            try:
                score = flair_prediction(review)
                scores.append(score)
            except Exception:
                continue
        if not scores:
            return 0
            
        return sum(scores) / len(scores)
    
    df['sentiment'] = df['review_texts'].apply(get_sentiment)
    df = df.drop(columns=['reviews', 'review_texts', 'courses'])
    df.to_csv('datasets/filtered_dataset.csv', index=False)
    print(f"Successfully saved filtered dataset with {len(df)} professors to filtered_dataset.csv")

if __name__ == "__main__":
    preprocess_data() 