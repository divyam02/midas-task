import pandas as pd
import jsonlines
import json
"""
Pending:
	1. Add image count
		@Note: Retweet image are not to be included!
		Look in entities->media only.
"""
tweets = []

with jsonlines.open('temp.jsonl', mode='r') as reader:
	for obj in reader.iter(type=dict, skip_invalid=True):
		tweets.append(obj)

tweets_data = pd.DataFrame(tweets)
"""
print(tweets_data.head())
print(tweets_data.columns)
for i in tweets_data.columns:
	print(tweets_data[i][9])

a = tweets_data['entities'][31]
print(a['media'])
"""
actual = pd.DataFrame()
actual['text'] = tweets_data['full_text']
actual['date'] = tweets_data['created_at']
actual['favorite_count'] = tweets_data['favorite_count']
actual['retweets'] = tweets_data['retweet_count']
#actual['images']

print(actual)

actual.to_csv('midas_twitter_feed_stats.csv')