from dotenv import dotenv_values
from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# APIキーの情報
config = dotenv_values('.env')
key = config.get('key')
secret = config.get('secret')
wait_time = 1

# 保存フォルダの指定
target = sys.argv[1]
savedir = "./"  + target

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = target,
    per_page = 10,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence'
)

photos = result['photos']
# pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + 'jpg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
