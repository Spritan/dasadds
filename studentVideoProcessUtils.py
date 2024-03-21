import cv2
from moviepy.editor import ImageSequenceClip

net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

classes = []

with open("models/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


def zoomIntoPerson(
    vid_path: str = "/home/spritan/Downloads/IMG_6583.MOV",
    out_path: str = "/run/media/spritan/New Volume/Github/Sparts/SPARTS_video_analyser/SPART_VIDEO_ANALYSER/demo/data/output_video2.mp4",
):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_frames = []
    count = 0
    cropped_image = None

    while True:
        if cropped_image is not None:
            del cropped_image

        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        # print("height, width :", height, width)
        try:
            # Get active buttons list
            active_buttons = "person"
            class_ids, scores, bboxes = model.detect(
                frame, confThreshold=0.3, nmsThreshold=0.4
            )
            x_, y_, w_, h_ = bboxes[0]
            for bbox in bboxes:
                x, y, w, h = bbox
                if x<x_:
                    x_=x
                if y<y_:
                    y_=y
                if w>w_:
                    w_=w
                if h>h_:
                    h_=h
            for class_id, bbox in zip(class_ids, bboxes):
                x, y, w, h = bbox

                class_name = classes[class_id]

                if class_name in active_buttons:
                    # cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
                    cropped_image = frame[
                        max(0, int(y_ * 0.9)) : min(int(y_ + (h_ * 1.1)), height),
                        max(0, int(x_ * 0.9)) : min(int(x_ + (w_ * 1.3)), width),
                    ]

                    h, w, _ = cropped_image.shape

                    h_ratio = h_ / height
                    w_ratio = w_ / width

                    if h_ratio >= w_ratio:
                        resized_image = cv2.resize(
                            cropped_image, (int(w * (1 / h_ratio)), int(height))
                        )
                        pad_h = 0
                        pad_w = max(0, width - int(w * (1 / h_ratio)))
                    else:
                        resized_image = cv2.resize(
                            cropped_image, (int(w), int(h * (1 / h_ratio)))
                        )
                        pad_w = 0
                        pad_h = max(0, height - int(h * (1 / w_ratio)))

                    padded_image = cv2.copyMakeBorder(
                        cropped_image,
                        0,
                        pad_h,
                        0,
                        pad_w,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                    resized_image = cv2.resize(padded_image, (width, height))

            # cv2.imwrite(f"/run/media/spritan/New Volume/Github/Sparts/SPARTS_video_analyser/SPART_VIDEO_ANALYSER/demo/data/{count}.png", padded_image)
            video_frames.append(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))  # type: ignore
            count += 1
        except Exception as e:
            print(e)

    try:
        video_clip = ImageSequenceClip(video_frames, fps=int(fps))  # type: ignore
        video_clip.write_videofile(
            out_path,
            codec="libx264",
        )
        return out_path
    except Exception as e:
        print(e)


if __name__ == "__main__":
    zoomIntoPerson()
