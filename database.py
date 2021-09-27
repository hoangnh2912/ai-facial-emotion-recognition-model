
from time import time
from pymongo import MongoClient
CONNECTION_STRING = "mongodb+srv://thanh:thanhtn123@cluster0.un40r.mongodb.net/data_mining?retryWrites=true&w=majority"
client = MongoClient(CONNECTION_STRING)


def get_customer():
    return client.get_database().get_collection('customer')


def insert_fer(emotion,id_name):
    print('Thêm ',emotion)
    new_item = {
        'emotion':emotion,
        'emotion_id':id_name,
        'create_date':int(time()*1000)
    }
    get_customer().insert_one(new_item)


# get_customer().delete_many({'emotion_id':{
#     '$ne':-1
# }})
