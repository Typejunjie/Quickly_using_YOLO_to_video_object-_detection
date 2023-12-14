from ultralytics import YOLO
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--videos',type=str , default='video', help='')
parser.add_argument('--results', type=str, default='results', help='')
opt = parser.parse_args()

model = YOLO("yolov8n.pt")
model.train(data="coco128.yaml", epochs=3)  # train the model

video_path = f'./{opt.videos}'
video_names = os.listdir(video_path)
video_list = list(os.path.join(video_path, file) for file in video_names)

for i, video_path in enumerate(video_list):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建VideoWriter对象以保存提取的帧为新的视频文件
    output = cv2.VideoWriter(f'./{opt.results}/{video_names[i]}', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 循环读取视频帧并保存
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 在这里对帧进行处理（可选）
        results = model(frame)
        frame = results[0].plot()
        # 将帧写入输出文件
        output.write(frame)
        
        # 显示帧（可选）
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    output.release()
    cv2.destroyAllWindows()