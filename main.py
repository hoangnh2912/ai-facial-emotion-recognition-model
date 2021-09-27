from fastapi import FastAPI, Form, UploadFile, File
app = FastAPI(docs_url="/data_mining", redoc_url=None)
from database import get_customer
from typing import Optional

@app.get('/get_emotions')
async def _get_emotions(from_date:Optional[str]=None,to_date:Optional[str]=None):
    list_emotion = get_customer().find()
    list_emotion = list(list_emotion).copy()
    for i in range(len(list_emotion)):
        list_emotion[i]['_id'] = str(list_emotion[i]['_id'])
    return {
        "data":list_emotion
    }


