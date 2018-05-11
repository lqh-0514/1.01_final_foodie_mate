import pymongo
from pymongo import MongoClient



#connect to mongodb, start client session
client = MongoClient()
#connect to specific database
db = client.olive
#connect to specific collection
collection = db.yelp_reviews
#finding needs from full collection, data = collection.find({'restaurant_id':'neptune-oyster-boston'}) will find only neptune, etc.
data = collection.find() #finds all

#extract data from cursor
itembuffer=[]
for item in data:
     itembuffer.append(item)

#itembuffer has exactly the same structure as set_data in your code, can directly plug in, regardless of find() or find({some criteria})

