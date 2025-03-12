import argparse
import cv2
import time
import threading
from playsound import playsound
from ultralytics import YOLO

ALARM_COOLDOWN = 2  # seconds
last_alarm_time = 0

def play_alarm():
    playsound("sound/alarm.wav")

def detect_in_image(model, image_path):
    global last_alarm_time
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return

    results = model(img)
    drowning_detected = False

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            if label.lower() in ["drowning", "drown"]:
                drowning_detected = True
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "Drowning", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # Play alarm sound if cooldown period has passed
                if time.time() - last_alarm_time > ALARM_COOLDOWN:
                    threading.Thread(target=play_alarm, daemon=True).start()
                    last_alarm_time = time.time()
            elif label.lower() in ["swimming"]:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, "Swimming", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif label.lower() in ["out of water"]:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, "Out of Water", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    if drowning_detected:
        print("Drowning detected in the image!")
    else:
        print("No drowning detected in the image.")

    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_in_video(model, video_path):
    global last_alarm_time
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_drowning = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                label = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                if label.lower() in ["drowning", "drown"]:
                    frame_drowning = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Drowning", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    # Play the alarm sound if enough time has passed
                    if time.time() - last_alarm_time > ALARM_COOLDOWN:
                        threading.Thread(target=play_alarm, daemon=True).start()
                        last_alarm_time = time.time()
                elif label.lower() in ["swimming"]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Swimming", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                elif label.lower() in ["out of water"]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "Out of Water", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if frame_drowning:
            print("Drowning detected in current frame.")

        cv2.imshow("Video Detection", frame)
        # Press 'q' to exit the video loop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Drowning Detection App")
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help="Path to an image or video file. E.g., 'video.mp4' or 'image.jpg'"
    )
    args = parser.parse_args()

    model = YOLO("best.pt")

    if args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        detect_in_image(model, args.source)
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detect_in_video(model, args.source)
    else:
        print("Unsupported file format. Please provide an image or video file.")

if __name__ == "__main__":
    main()
