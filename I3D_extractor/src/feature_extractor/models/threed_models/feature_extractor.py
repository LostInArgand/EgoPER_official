import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models      # type: ignore
from torch.utils.data import Dataset
from i3d_resnet import i3d_resnet
from tqdm import tqdm
from collections import Counter, deque

# Using RESNET50 convert [f,c,h,w] -> [f,2048]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class npyLoader(Dataset[object]):
    def __init__(self, video_path: str):
        
        self.videos_dir = video_path

        videos_names = os.listdir(self.videos_dir)
        self.video_paths = [
            os.path.join(self.videos_dir, item) for item in videos_names
        ]

        self.model = i3d_resnet(50, 400, 0.5, without_t_stride=False).cuda()        # type: ignore
        self.model.eval()
        print("I3D Loaded", flush=True)

    def read_video_clip(self, video_path: str, start: int, clip_length: int, resize: "tuple[int, int]" = (224, 224)):
        """
        Read a clip of frames from the video starting at 'start' frame.
        """
        videocap = cv2.VideoCapture(video_path)
        videocap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        for _ in range(clip_length):
            success, frame = videocap.read()
            if not success:
                break
            frame = cv2.resize(frame, resize)
            # Convert BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        videocap.release()
        if len(frames) == clip_length:
            return np.stack(frames)
        return None
    
    def normalize(self, frames: np.ndarray, mean: "list[float]", std: "list[float]") -> torch.Tensor:
        """
        Normalize the clip with the given mean and std.
        """
        frames = torch.from_numpy(frames).float() / 255.0  # T, H, W, C
        frames = frames.permute(3, 0, 1, 2)  # C, T, H, W
        for c in range(3):
            frames[c] = (frames[c] - mean[c]) / std[c]
        return frames.unsqueeze(0)  # (1, C, T, H, W)

    def _get_label(self, labels: "list[str]", start: int, clip_length: int) -> str:
        return Counter(labels[start:start + clip_length]).most_common(1)[0][0]

    def _extract_features(self, video_path: str, feature_path: str, annotations_path:str, gt_path: str) -> int:
        """
        This method is to extract npy frames from the video
        """

        print(video_path, flush=True)
        video_name = video_path.split('/')[-1].split('.')[0]
        print(video_name, flush=True)
    
        npy_file_path = os.path.join(feature_path, f'{video_name}.npy')
        gt_file_path = os.path.join(gt_path, f'{video_name}.txt')

        videocap = cv2.VideoCapture(video_path)
        total_frames = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))

        # frame_count = 0
        gt = []
        clip_length = 16

        mean = self.model.mean()
        std = self.model.std()

        annotations_file_path = os.path.join(annotations_path, f'{video_name}-phase.txt')
        with open(annotations_file_path, 'r') as f:
            annotations = f.read().rstrip().split('\n')
            del annotations[0]  # Remove the first line which is not needed
            labels = [label.split()[1] for label in annotations]
            # print(labels[0], labels[1], flush=True)

        assert len(labels) == total_frames, "Labels length does not match total frames in video."

        batch_size = 128
        features_extracted = []
        all_clips = []  # type: ignore

        frame_q = deque(maxlen=clip_length)  # To store the last 'clip_length' frames
        for i in range(clip_length):
            success, frame = videocap.read()
            if success:
                frame = cv2.resize(frame, (224, 224))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_q.append(frame_rgb)
        gt_label = self._get_label(labels, 0, clip_length)
        gt.append(gt_label)
        all_clips.append(self.normalize(np.stack(frame_q), mean, std))  # type: ignore
        # print(len(gt), all_clips[0].shape, flush=True)

        for start in tqdm(range(clip_length, total_frames), desc=f"Processing {video_name}"):
            success, frame = videocap.read()
            # print(len(gt), len(features_extracted) * batch_size + len(all_clips), flush=True)
            if success:
                frame = cv2.resize(frame, (224, 224))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_q.append(frame_rgb)
                all_clips.append(self.normalize(np.stack(frame_q), mean, std))  # type: ignore
                gt_label = self._get_label(labels, start - clip_length + 1, clip_length)
                gt.append(gt_label)
                if len(all_clips) == batch_size:
                    all_clips_tensor = torch.cat(all_clips, dim=0)
                    # print(f"Batch clips tensor shape: {all_clips_tensor.shape}, {start}", flush=True)
                    # Extract features for the batch
                    with torch.no_grad():
                        feature = self.model(all_clips_tensor.cuda())
                        features_extracted.append(feature.cpu())
                    all_clips = []
                    # print(f"Features extracted so far: {len(features_extracted) * 128}", flush=True)
        if len(all_clips) > 0:
            all_clips_tensor = torch.cat(all_clips, dim=0)
            # print(f"Final batch clips tensor shape: {all_clips_tensor.shape}", flush=True)
            # Extract features for the remaining clips
            with torch.no_grad():
                feature = self.model(all_clips_tensor.cuda())
                features_extracted.append(feature.cpu())
            all_clips = []
        videocap.release()
        # print(len(gt), len(features_extracted) * batch_size + len(all_clips), flush=True)
        features_extracted = torch.cat(features_extracted, dim=0).numpy()  # shape: (N_clips, 2048)
        print(features_extracted.shape, len(gt), flush=True)
        # print(f"Features extracted shape: {features_extracted.shape}", flush=True)
        assert len(features_extracted) == len(gt), "Features extracted length does not match labels length."
        torch.cuda.empty_cache()
        
        videocap.release()

        print(f"Frames_conv size: {features_extracted.shape}", flush=True)

        np.save(npy_file_path, features_extracted)            # type: ignore

        with open(gt_file_path, 'w') as f:
            f.write('\n'.join(gt))

        print("NPY Dataset Created", flush=True)

        return 0
    

if __name__ == '__main__':
    video_path = "/home/nano01/a/dalwis/cholec/videos"
    feature_path = "/home/nano01/a/dalwis/cholec/features/I3D"
    annotations_path = "/home/nano01/a/dalwis/cholec/annotations"
    gt_path = "/home/nano01/a/dalwis/cholec/I3D_gt"

    obj = npyLoader(video_path)
    vid_list_file = "./data/splits/test_split1.txt"
    with open(vid_list_file, 'r') as vid_list_file:
        vid_list = vid_list_file.read().rstrip().split('\n')
        video_ids = list(map(int, [vid.split('.')[0][-2:] for vid in vid_list]))

    for vid in obj.video_paths:
        video_name = vid.split('/')[-1].split('.')[0]
        if int(video_name[-2:]) not in video_ids:
            continue
        print(video_name)
        obj._extract_features(vid, feature_path, annotations_path, gt_path)          # type: ignore
    print("Done !!!", flush=True)