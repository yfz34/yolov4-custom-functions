# 系統環境
- Ubuntu 20.04.5 LTS
- python(3.9.15)

# 建立虛擬環境

```shell
python -m venv car
cd car
source bin/activate
```

# 安裝套件

```
sudo apt-get install libgtk2.0-dev
sudo apt-get install pkg-config
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
pip install -r requirements.txt
```

# 配置

1. image folder copy intto ./data/
2. ./data/classes 建立"carno.names"檔案 內文"carno"
3. ./data/ 放"carno.weights"
