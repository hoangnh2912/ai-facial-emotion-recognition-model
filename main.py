from fastapi import FastAPI, Form, UploadFile, File
app = FastAPI(docs_url="/data_mining", redoc_url=None)
from database import get_customer
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8081",
    "http://128.199.72.145:8081",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/get_emotions')
async def _get_emotions(from_date:Optional[str]=None,to_date:Optional[str]=None):
    list_emotion = get_customer().find()
    list_emotion = list(list_emotion).copy()
    for i in range(len(list_emotion)):
        list_emotion[i]['_id'] = str(list_emotion[i]['_id'])
    return {
        "data":list_emotion
    }


