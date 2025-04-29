import cv2
import numpy as np
import os
import argparse
import pickle
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms as T
from Tracklet import Tracklet
from reid.torchreid.utils import FeatureExtractor


def main(args):
    # Initialize feature extractor
    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path = 'checkpoints/sports_model.pth.tar-60',
        device='cuda'
    )   

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.path))[0]
    output_file = os.path.join(args.output_dir, f'{video_name}.pkl')

    # Load tracking results
    track_res = np.loadtxt(args.pred_file, delimiter=',')
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {args.path}")
    
    seq_tracks = {}
    last_frame = int(track_res[-1, 0]) if len(track_res) > 0 else 0

    for frame_id in tqdm(range(1, last_frame + 1), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections for current frame
        frame_dets = track_res[track_res[:, 0] == frame_id]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        input_batch = None
        tid2idx = {}

        for idx, det in enumerate(frame_dets):
            frame_num, track_id = int(det[0]), int(det[1])
            l, t, w, h = det[2:6]
            score = det[6]
            bbox = [l, t, w, h]

            # Update tracklet
            if track_id not in seq_tracks:
                seq_tracks[track_id] = Tracklet(track_id, frame_num, score, bbox)
            else:
                seq_tracks[track_id].append_det(frame_num, score, bbox)
            tid2idx[track_id] = idx

            # Extract appearance features
            im = img.crop((l, t, l+w, t+h)).convert('RGB')
            im = val_transforms(im).unsqueeze(0)
            input_batch = im if input_batch is None else torch.cat([input_batch, im], dim=0)

        if input_batch is not None:
            features = extractor(input_batch).cpu().numpy()
            for tid, idx in tid2idx.items():
                feat = features[idx]
                feat /= np.linalg.norm(feat)  # L2 normalize
                seq_tracks[tid].append_feat(feat)

    cap.release()

    # Save results
    with open(output_file, 'wb') as f:
        pickle.dump(seq_tracks, f)
    print(f"Saved tracklets to {output_file}")


def make_parser():
    parser = argparse.ArgumentParser("Generate tracklets from video")
    parser.add_argument(
        "--path",type=str, default="video.mp4", help="path to images or video"
    )
    parser.add_argument(
        "--pred_file",required=True, type=str, help="path to MOT prediction text file"
    )
    parser.add_argument(
        "--output_dir",
        default="out/tracklets",
        type=str,
        help="output directory to save the results",
    )
    return parser




if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
