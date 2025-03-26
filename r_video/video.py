import cv2 as cv
import torch
def read_video(video_path):
    torch.cuda.empty_cache()
    skip_frames=3
    cap=cv.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame=cv.resize(frame,(640,480))
        for cap in range(skip_frames-1):
            ret, frame = cap.read()
        frames.append(frame)   
    cap.release()
def save_video(video_path, frames):
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(video_path, fourcc, 20.0, (640, 480))
    for frame in frames:
        out.write(frame)
    out.release()