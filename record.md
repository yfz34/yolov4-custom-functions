# 系統環境
- Ubuntu 20.04.5 LTS
- python(3.9.15)

# 建立虛擬環境

```shell
python -m venv car
cd car
source bin/activate
```

# Tesseract OCR 5
## 安裝
```
sudo add-apt-repository ppa:alex-p/tesseract-ocr-devel
sudo apt install -y tesseract-ocr
```

## tessdata_best
https://github.com/tesseract-ocr/tessdata_best/blob/main/eng.traineddata
下載"eng.traineddata"放至/usr/share/tesseract-ocr/5/tessdata/

# 安裝其他套件

```
sudo apt-get install libgtk2.0-dev
sudo apt-get install pkg-config
sudo apt install libtesseract-dev
pip install -r requirements.txt
```

# 專案配置

1. image folder copy intto ./data/
2. ./data/classes 建立"carno.names"檔案 內文"carno"
3. ./data/ 放"carno.weights"

# 建立model
```
python save_model.py --weights ./data/carno.weights --output ./checkpoints/carno-416 --input_size 416 --model yolov4 
```

# 本機測試
```
python detect.py --weights ./checkpoints/carno-416 --size 416 --model yolov4 --images ./data/images/plate1.jpg --ocr
```

# 服務啟動
```
python main.py
```
