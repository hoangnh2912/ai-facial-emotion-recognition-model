from fastapi import FastAPI, Form, UploadFile, File
app = FastAPI(docs_url="/data_mining", redoc_url=None)
from database import get_customer
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import calendar
import pandas as pd
import numpy as np
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8081",
    "http://128.199.72.145:8081",
    "http://128.199.72.145:8082",
]
class_name = ["bình thường", "Vui", "Ngạc nghiên", "Buồn", "Tức giận", "Kinh tởm", "Sợ hãi", "Khinh thường"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/get_emotions')
async def _get_emotions(day:Optional[int]=None,month:Optional[int]=None,year:Optional[int]=None):
    result = []
    label = []
    if day is None:
        _,total_days_in_month = calendar.monthrange(int(year),int(month))
        label = list(range(1,total_days_in_month+1)) 
        f_ts = datetime(year,month,1).timestamp()
        t_ts = datetime(year,month,total_days_in_month,23,59,59).timestamp()
        list_emotion = get_customer().find({
            'create_date':{'$gte':f_ts*1000,'$lte':t_ts*1000}
        })
        list_emotion = list(list_emotion).copy()
        for i in range(len(list_emotion)):
            list_emotion[i]['_id'] = str(list_emotion[i]['_id'])
            date = datetime.fromtimestamp(list_emotion[i]['create_date']/1000)
            list_emotion[i]['day'] = date.day
        df = pd.DataFrame(list_emotion)
        for id_emo,emo in enumerate(class_name):
            if id_emo != 0:
                data_count = np.zeros(total_days_in_month)
                for d in range(total_days_in_month):
                    try:
                        data_count[d] = df[(df['day']==(d+1)) & (df['emotion_id']==id_emo)].__len__()
                    except:
                        data_count[d] = 0
                result.append({
                    'data':list(data_count),
                    'label':emo
                })
        return {
            "data":result,
            "label":label,
        }
    if day is not None:
        result = []
        f_ts = datetime(year,month,day).timestamp()
        t_ts = datetime(year,month,day,23,59,59).timestamp()
        list_emotion = get_customer().find({
            'create_date':{'$gte':f_ts*1000,'$lte':t_ts*1000}
        })
        list_emotion = list(list_emotion).copy()
        for i in range(len(list_emotion)):
            list_emotion[i]['_id'] = str(list_emotion[i]['_id'])
        if list_emotion.__len__() >0:
            df = pd.DataFrame(list_emotion)
            for id_emo,emo in enumerate(class_name):
                if id_emo!=0:
                    result.append(df[df['emotion_id']==id_emo].__len__())
        return {
            'data':list(result)
        }



