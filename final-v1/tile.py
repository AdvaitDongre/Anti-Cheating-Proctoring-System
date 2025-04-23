import cv2
import dlib

# Load face detector
detector = dlib.get_frontal_face_detector()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set a larger fixed tile position
TILE_X1, TILE_Y1, TILE_X2, TILE_Y2 = 150, 80, 500, 420  # Increased size

def check_user_position(frame, faces):
    """
    Checks if the user's face is inside the fixed tile.
    If no face is detected, shows a warning.
    """
    if faces:
        # Use the largest detected face (closest to camera)
        face = max(faces, key=lambda rect: rect.width() * rect.height())

        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Expand detection box slightly for better accuracy
        padding = 10
        x, y, w, h = x - padding, y - padding, w + (2 * padding), h + (2 * padding)

        # Check if face is outside the tile
        if x < TILE_X1 or y < TILE_Y1 or (x + w) > TILE_X2 or (y + h) > TILE_Y2:
            color = (0, 0, 255)  # Red (out of bounds)
            warning_text = "WARNING: Stay inside the tile!"
        else:
            color = (0, 255, 0)  # Green (inside bounds)
            warning_text = None
    else:
        color = (0, 0, 255)  # Red when no face is detected
        warning_text = "WARNING: No person detected!"

    # Draw the fixed tile (larger now)
    cv2.rectangle(frame, (TILE_X1, TILE_Y1), (TILE_X2, TILE_Y2), color, 3)

    # Display warning if needed
    if warning_text:
        cv2.putText(frame, warning_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)  # Bigger, clearer warning

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    frame = check_user_position(frame, faces)

    cv2.imshow("Fixed Tile Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
