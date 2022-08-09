import cv2
import torch


class YoloObjectDetection:

    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)

    def load_model(self):
        model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path='/content/best.pt', force_reload=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(
                    row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(
                    labels[i]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):

        cap = cv2.VideoCapture(0)

        count = 0
        writer = None
        (Width, Height) = (None, None)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            Width = frame.shape[1]
            Height = frame.shape[0]

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"DIVX")
                writer = cv2.VideoWriter("output.mp4", fourcc, 20,
                                         (Width, Height), True)

            writer.write(frame)

        cap.release()
        writer.release()
        cv2.destroyAllWindows()


detection = YoloObjectDetection()
detection()
