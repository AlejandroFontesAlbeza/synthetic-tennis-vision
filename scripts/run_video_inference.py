import os

import cv2

from inference.video_io import VideoReader, VideoWriter

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "..", "data", "tennisMatch", "clips", "clip1.mp4")
output_path = os.path.join(script_dir, "..", "data", "tennisMatch", "clips", "clip1_output.mp4")

reader = VideoReader(video_path, frame_skip=1)
frame_size = (int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
writer = VideoWriter(output_path, frame_size=frame_size, fps=30)
for frame, frame_number in reader:
    print(f"Processing: {frame_number}")
    frame = cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    writer.write(frame)
reader.release()
writer.release()
print("Done processing video.")