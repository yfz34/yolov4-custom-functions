import uvicorn
from typing import Union
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
from werkzeug.utils import secure_filename
import uuid
import pathlib
import detect_server

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

upload_path = "./temp/uploads"
detect_path = "./temp/uploads"

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(file: UploadFile):
    # if file.content_type != "image/jpg":
    #     raise HTTPException(400, detail="檔案類型錯誤")

    id = str(uuid.uuid4().hex)
    fileName = secure_filename(file.filename)
    newFileName = id + pathlib.Path(fileName).suffix
    fullFilePath = os.path.join(upload_path, newFileName)

    with open(fullFilePath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detectResult = detect_server.run(fullFilePath)

    response = {
        "id": id,
        "detectFileUrl": "/file/detections/" + detectResult["fileName"],
        "licensePlate": detectResult["licensePlate"]
    }

    return response

@app.get("/detail/{id}")
def get_detail(id: str):
    return [
        {
            "name": "graypre",
            "url": "/file/graypre/" + id + ".png",
        }, 
        {
            "name": "edgedpre",
            "url": "/file/edgedpre/" + id + ".png",
        },
        {
            "name": "warp",
            "url": "/file/warp/" + id + ".png",
        },
        {
            "name": "smoothened",
            "url": "/file/smoothened/" + id + ".png",
        },
        {
            "name": "thresh",
            "url": "/file/thresh/" + id + ".png",
        },
        {
            "name": "erosion",
            "url": "/file/erosion/" + id + ".png",
        },
        {
            "name": "blur",
            "url": "/file/blur/" + id + ".png",
        }
    ]

@app.get("/file/{type}/{name_file}")
def get_file(type: str, name_file: str):
    return FileResponse(path=os.getcwd() + "/temp/" + type + "/" + name_file)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, log_level="info")