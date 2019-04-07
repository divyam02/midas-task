from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import *
from collections import Counter
import sys


consumer_key=" "
consumer_secret=" "
access_token=" "
access_token_secret=" "

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth)

item = auth_api.get_user(target)
tweets = item.statuses_count

tweet_count = 0

end_date = datetime.utcnow() - timedelta(days=30)
for status in Cursor(auth_api.user_timeline, id=target).items():
	tweet_count+=1
	"""
	Add code to get relevant details
	"""
	if status.created_at < end_date:
		break

	