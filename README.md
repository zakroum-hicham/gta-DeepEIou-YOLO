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
pip install -q ultralytics 
```


## RUN DeepEIoU DEMO

```

cd ..
video_path = "video.mp4"
python demo.py --path "{video_path}" --ckpt "checkpoints/best.pt"

```
### DEMO
[Demo DeepEIoU](https://github.com/zakroum-hicham/gta-DeepEIou-YOLO/blob/main/demo1.mp4)

## RUN DeepEIoU + GTA DEMO

1.Generate tracklets
```

video_path = "video.mp4"
python gen_tracklet.py --path "{video_path}" --pred_file "ex: out/2025_04_29_13_08_00/video.txt"

```
2.Refine tracklets

```

video_path = "video.mp4"
python refine_tracklets.py --video_path "{video_path}" --track_src "ex: out/tracklets" --use_split --min_len 100 --eps 0.6 --min_samples 10 --max_k 3 --use_connect --spatial_factor 1.0 --merge_dist_thres 0.4

```
### DEMO
