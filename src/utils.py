import pandas as pd 
from sklearn.model_selection import train_test_split

def to_sentiment(rating):
  rating = int(rating)
  if rating <= 2:
    return 0
  elif rating == 3:
    return 1
  else: 
    return 2



def read_data_from_path(data_path):
    #function to read the data from the data_path
    #modify this function depending on your dataset
    df = pd.read_csv(data_path)
    df['sentiment'] = df.score.apply(to_sentiment)
    # class_names = ['negative', 'neutral', 'positive']
    RANDOM_SEED=42
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

    print("Loading  Dataset")
    train_reviews=df_train.content.to_numpy()
    train_labels=df_train.sentiment.to_numpy()

    val_reviews=df_val.content.to_numpy()
    val_labels=df_val.sentiment.to_numpy()

    test_reviews=df_test.content.to_numpy()
    test_labels=df_test.sentiment.to_numpy()

    return train_reviews, train_labels, val_reviews, val_labels, test_reviews, test_labels