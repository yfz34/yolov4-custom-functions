for ((i=1;i<=150;i++))
do
  echo "$i"
  python detect.py --weights ./checkpoints/carno-416 --size 416 --model yolov4 --images ./data/images/plate"$i".jpg --plate --dont_show
done