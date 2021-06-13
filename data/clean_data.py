import pandas as pd

df_list = []

# load twitter data
twitter_df = pd.read_csv('twitter_dataset.csv', usecols=['class', 'tweet'])
twitter_df.columns = ['classification', 'text']
twitter_df['label'] = twitter_df.apply(lambda row: 0 if row.classification==2 else 1, axis = 1)
df_list.append(twitter_df[['text', 'label']])

# load fox news comment data
fox_news_df = pd.read_json('fox_news_dataset.json', orient='records', lines=True)
df_list.append(fox_news_df[['text', 'label']])

# load wikipedia talk page data
wikipedia_df = pd.read_csv('wikipedia_dataset.csv', usecols=['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
wikipedia_df.columns = ['text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
wikipedia_df['label'] = wikipedia_df.apply(lambda row: 1 if row.toxic or row.severe_toxic or row.obscene or row.threat or row.insult or row.identity_hate else 0, axis=1)
df_list.append(wikipedia_df[['text', 'label']])

df = pd.concat(df_list)

df.to_csv('clean_data.csv', index=False)