# converts video to images at a certain interval 
import os
import cv2

video_path=r'C:\Users\aquar\OneDrive\Desktop\code\internship\master_pr\steel_plate_prjkt\train\plate loading.mkv' #path of target vid
output_dir=r'C:\Users\aquar\OneDrive\Desktop\code\internship\master_pr\steel_plate_prjkt\train\frames' 
frame_rate= 5

os.makedirs(output_dir,exist_ok=True)

cap=cv2.VideoCapture(video_path)
fps=cap.get(cv2.CAP_PROP_FPS)
frames_intervals=int(fps*frame_rate)

frame_count=0
saved_count=0

while cap.isOpened():
    ret, frame=cap.read()
    if not ret:
        break
    if frame_count % frames_intervals==0:
        filename=os.path.join(output_dir,f'frame_{saved_count:05d}.jpg')
        cv2.imwrite(filename, frame)
        saved_count +=1
    frame_count +=1
    
cap.release()
print(f"saved {saved_count} frames.")

