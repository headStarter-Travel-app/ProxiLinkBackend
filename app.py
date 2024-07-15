# app.py
from dotenv import load_dotenv
import os

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
