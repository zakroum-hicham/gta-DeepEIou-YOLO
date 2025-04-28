# gta-DeepEIou-YOLO

## Setup Instructions

* Clone this repo
* Install dependencies.
```
git clone https://github.com/zakroum-hicham/gta-DeepEIou-YOLO.git
cd gta-DeepEIou-YOLO
```
```
cd reid
pip install -r requirements.txt
pip install cython_bbox
python setup.py develop
```

## RUN DEMO

```

cd ..
video_path = "video.mp4"
!python demo.py --path "{video_path}" --ckpt "checkpoints/best.pt"

```
