import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        print(f"Camera found at index: {i}")

        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(3000)

        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"No camera at index: {i}")