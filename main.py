import cv2
import pytesseract
import csv
import datetime
import time

# Specify the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Set up the CSV file
csv_file = open('recognized_text.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Recognized Text'])

# Function to perform text recognition and draw bounding boxes
def recognize_text_and_draw_boxes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = pytesseract.image_to_string(gray)
    return text

try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Perform text recognition and draw bounding boxes
        text = recognize_text_and_draw_boxes(frame)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write the recognized text to the CSV file
        csv_writer.writerow([timestamp, text])

        # Display the frame with bounding boxes
        cv2.imshow('Live Feed', frame)

        # Wait for 0.2 seconds
        elapsed_time = time.time() - start_time
        if elapsed_time < 0.2:
            time.sleep(0.2 - elapsed_time)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the video capture object and close the CSV file
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
