import pymongo
from pymongo import InsertOne
import pandas as pd
import json

MONGO_DB = "cryptocurrency"
MONGO_COLLECTION = "deribit_transactions"


def connect_db(port=27017, db=MONGO_DB):
    client = pymongo.MongoClient(port=port)
    db = client[db]
    return db


def get_as_df(collection, query):
    cursor = collection.find(query)
    df = pd.DataFrame(list(cursor))
    return df


# def write_in_db(coll, df):
#     """writes the dataframe in the collection

#     Args:
#         coll (mongodb.collection): the collection to write in
#         df (pd.DataFrame): the data to write
#     """
#     records = json.loads(df.T.to_json()).values()
#     coll.insert_many(records)


# def bulk_write(coll, df, ordered=False):
#     records = json.loads(df.T.to_json()).values()
#     operations = []
#     for doc in records:
#         op = InsertOne(doc)
#         operations.append(op)

#     try:
#         coll.bulk_write(operations, ordered=ordered)
#     except pymongo.errors.BulkWriteError as bwe:
#         print(coll.name, len(bwe.details["writeErrors"]))
