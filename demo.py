import argparse
import os
import os.path as osp
import torch
from loguru import logger
from ultralytics import YOLO
import time
import cv2
import sys
sys.path.append('.')
from reid.torchreid.utils import FeatureExtractor
from tracker.Deep_EIoU import Deep_EIoU
from yolox.utils.visualize import plot_tracking


def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU Demo")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="../demo.mp4", help="path to images or video"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        help="whether to save the inference result of image/video",
    )

    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # reid args
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    return parser



def imageflow_demo(predictor, extractor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, args.path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = Deep_EIoU(args, frame_rate=30)
    frame_id = 0
    results = []
    while True:
        if frame_id % 30 == 0:
            logger.info('Processing frame {})'.format(frame_id))
        ret_val, frame = cap.read()
        if ret_val:
            yolo_results = predictor(frame,verbose=False)[0]
            det = yolo_results.boxes.data.cpu().numpy()
            det  = det[:, :5]
            if len(det) > 0:
                mask = (det[:, 0] >= 0) & (det[:, 1] >= 0) & (det[:, 2] <= width) & (det[:, 3] <= height)
                det = det[mask]
                
                # Get cropped images for ReID
                cropped_imgs = [
                    frame[int(y1):int(y2), int(x1):int(x2)] 
                    for x1, y1, x2, y2,_ in det
                ]
                embs = extractor(cropped_imgs)
                embs = embs.cpu().detach().numpy()
                
                online_targets = tracker.update(det, embs)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.last_tlwh
                    tid = t.track_id
                    if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                online_im = plot_tracking(
                    frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=30
                )
            else:
                online_im = frame
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")



def main(args):

    output_dir = "/content/out"
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

  
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    model = YOLO(args.ckpt)


    current_time = time.localtime()
    
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path = 'checkpoints/sports_model.pth.tar-60',
        device='cuda'
    )   

    imageflow_demo(model, extractor, vis_folder, current_time, args)



if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)