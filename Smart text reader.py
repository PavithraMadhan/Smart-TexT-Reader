"""

This script continuously captures frames from a webcam video stream, processes each frame using Optical Character Recognition (OCR) to detect and extract text, and then displays the processed frame with bounding boxes around the detected text. The extracted text is also saved to a text file in a specified directory. Additionally, the script converts the extracted text into speech using Google Text-to-Speech (gTTS) library and saves the generated speech as an MP3 file, which is played back using the system's default media player. The script runs until the user presses 'q' to quit the video stream.

Imports:
- time: Provides functionality for adding delays between displaying frames.
- cv2: OpenCV library for computer vision tasks, including video capture, processing, and display.
- pytesseract: Python wrapper for Tesseract OCR engine, enabling text extraction from images.
- imutils: Collection of convenience functions for working with OpenCV, such as image resizing.
- os: Operating system interface for directory operations, used for creating directories to save images, text outputs, and speech files.
- gtts: Google Text-to-Speech library for converting text into spoken language.

Functions:
- process_video_frame: Processes a single frame of a video for OCR, detecting and extracting text, and drawing bounding boxes around the detected text.
- save_text_to_speech: Reads the content of a text file, converts it to speech using gTTS library, saves the speech as an MP3 file, and plays it using the system's default media player.
- run_ocr_video_stream: Continuously captures frames from the webcam, processes each frame for OCR, displays the resulting frames with text overlay, saves extracted text to a file, and converts the text to speech.

"""

import time
import cv2
import pytesseract
from pytesseract import Output
import imutils
import os
from gtts import gTTS


def process_video_frame(frame):
    """
    Process a single frame of a video to perform Optical Character Recognition (OCR) on text.
    :Parameters: frame (A numpy array representing the input frame of the video)
    :Returns: resized_frame (A numpy array representing the processed frame with bounding boxes drawn around detected text),
              extracted_text (A string containing the extracted text from the frame)
    """
    try:
        # Resize the frame for faster processing
        resized_frame = imutils.resize(frame, width=600)
        # Convert the frame to grayscale for OCR
        grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Apply binarization and thresholding to enhance text visibility
        thresh = cv2.threshold(grayscale_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Use Tesseract to do OCR on the grayscale frame with PSM 6 (Single line)
        ocr_data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config='--psm 6')

        # Draw bounding boxes around detected text and update text properties
        num_boxes = len(ocr_data['level'])
        extracted_text = ""  # Initialize a variable to store extracted text

        for i in range(num_boxes):
            if int(ocr_data['conf'][i]) > 60:  # Confidence threshold
                # Extract bounding box coordinates and text from OCR data
                (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                # Draw a rectangle around the detected text on the resized frame
                resized_frame = cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = ocr_data['text'][i].strip()

                if text:  # Only draw text if there's something to draw
                    font_scale = 0.6  # Adjust font scale if necessary
                    resized_frame = cv2.putText(resized_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                (255, 255, 255), 6)
                    extracted_text += text + " "  # Append extracted text to the variable

        # Print extracted text to the console
        print("Extracted text:", extracted_text)
        return resized_frame, extracted_text

    except Exception as e:
        print("Error in process_video_frame, please check the input frame and ensure that it is valid:", str(e))
        return frame, ""


def save_text_to_speech(file_path):
    """
    Reads the content of the text file specified by `file_path` and the generated speech is saved as an MP3 file
    :Parameters: File path ( String that denotes the path to the text file)
    :Returns: None
    """
    try:
        file_content = open(file_path, "r").read().replace("\n", " ")
        speech = gTTS(text=str(file_content), lang='en', slow=False)
        speech_file_path = file_path.replace(".txt", ".mp3")
        speech.save(speech_file_path)
        os.system(f"start {speech_file_path}")
    except Exception as e:
        print("Error in save_text_to_speech:", str(e))


def run_ocr_video_stream():
    """
    The function continuously captures frames from the camera device, processes each frame using the 'process_video_frame' function,
    and displays the resulting frame with bounding boxes around detected text.
    Press 'q' to quit the video stream.
    :Returns: None
    """
    try:
        # Create a parent directory for all reports
        reports_directory = "reports"
        if not os.path.exists(reports_directory):
            os.makedirs(reports_directory)

        # Create a directory with the current date and time
        current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
        directory_name = f"report_{current_datetime}"
        directory_path = os.path.join(reports_directory, directory_name)
        os.makedirs(directory_path)

        # Create a text file to save extracted text
        text_file_path = os.path.join(directory_path, "extracted_text.txt")
        text_file = open(text_file_path, "a")

        # Initialize the video stream from the webcam
        video_capture = cv2.VideoCapture(0)
        frame_count = 0

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            if not ret:
                print("Failed to initialize video capture. Please check if the camera is connected and accessible.")
                break

            # Process the frame
            frame, extracted_text = process_video_frame(frame)
            # Save the image
            image_path = os.path.join(directory_path, f"frame_{frame_count}.jpg")
            cv2.imwrite(image_path, frame)
            frame_count += 1
            # Display the resulting frame with OCR text
            cv2.imshow('OCR', frame)
            time.sleep(1)
            # Append extracted text to the text file
            text_file.write(extracted_text + "\n")

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Error in run_ocr_video_stream:", str(e))

    finally:
        # Release the capture when everything is done
        video_capture.release()
        cv2.destroyAllWindows()
        # Close the text file
        text_file.close()
        # Save extracted text to speech at the end
        save_text_to_speech(text_file_path)


if __name__ == "__main__":
    run_ocr_video_stream()