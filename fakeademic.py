import cv2
import webbrowser
import time
import numpy as np
import pygetwindow as gw
from pywinauto.application import Application

triggered = False
cam = cv2.VideoCapture(0)
time.sleep(2)

print("🎮 Sit normally and press 's' to start detection...")

# Wait until user presses 's' and capture multiple reference frames
reference_frames = []
while True:
    ret, frame = cam.read()
    cv2.imshow("Press 's' when ready", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("📸 Capturing reference frames...")
        for _ in range(10):
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            reference_frames.append(gray)
            time.sleep(0.2)
        break

cv2.destroyAllWindows()

# Average the reference frames
gray_base = np.mean(reference_frames, axis=0).astype(dtype=np.uint8)

# Get frame size for safe zone
height, width = gray_base.shape
safe_zone = (width // 5, height // 5, width * 3 // 5, height * 3 // 5)

print("📷 Watching for intruders behind you...")

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frame_delta = cv2.absdiff(gray_base, gray)
    thresh = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    intruder_found = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Large movement only
            x, y, w, h = cv2.boundingRect(contour)

            # Ignore motion inside safe zone
            if (x > safe_zone[0] and y > safe_zone[1] and
                x + w < safe_zone[2] and y + h < safe_zone[3]):
                continue

            # Draw rectangle around intruder
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            intruder_found = True

    # Draw safe zone (green box)
    cv2.rectangle(frame, (safe_zone[0], safe_zone[1]), (safe_zone[2], safe_zone[3]), (0, 255, 0), 2)

    # Overlay status text
    if intruder_found:
        cv2.putText(frame, "⚠ Intruder Detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 3, cv2.LINE_AA)

        if not triggered:
            print("⚠ Intruder Detected! Switching to Study Mode...")
            triggered = True
            webbrowser.open("https://www.khanacademy.org/science")
            time.sleep(3)

            try:
                windows = gw.getWindowsWithTitle('Khan Academy')
                if windows:
                    app = Application().connect(title=windows[0].title)
                    app.top_window().set_focus()
            except Exception as e:
                print("🔴 Could not bring window to front:", e)
    else:
        cv2.putText(frame, "✅ All Clear", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the live feed
    cv2.imshow("Fakeademic Live Feed", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()