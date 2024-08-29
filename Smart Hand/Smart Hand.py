import cv2
import os, sys
import mediapipe as mp
import math
import time
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
window_name = 'Hand Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 500)

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
def apply_icon(image, icon_path, pos, size):
    if getattr(sys, 'frozen', False):
        # PyInstaller
        base_path = sys._MEIPASS
    else:
        # normal script
        base_path = os.path.abspath(".")

    # full path to the image
    full_icon_path = os.path.join(base_path, icon_path)

    # Load and resize the image
    icon = cv2.imread(full_icon_path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        raise FileNotFoundError(f"Icon file not found at path: {full_icon_path}")
    icon = cv2.resize(icon, (size, size))

    b_channel, g_channel, r_channel, alpha_channel = cv2.split(icon)
    mask = alpha_channel
    roi = image[pos[1]:pos[1] + size, pos[0]:pos[0] + size]
    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    icon_fg = cv2.merge((b_channel, g_channel, r_channel))
    combined = cv2.add(roi_bg, icon_fg)
    image[pos[1]:pos[1] + size, pos[0]:pos[0] + size] = combined

def showInfo(image):
    # Large rectangle
    info_box_top_left = (85, 85)
    info_box_bottom_right = (image.shape[1] - 80, image.shape[0] - 80)

    # Draw large rectangle
    cv2.rectangle(image, info_box_top_left, info_box_bottom_right, (0, 0, 0), -1)

    cv2.putText(image, 'Smart Hand v1.0', (info_box_top_left[0] + 20, info_box_top_left[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.putText(image, '- To select a button, hold your', (info_box_top_left[0] + 10, info_box_top_left[1] + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, 'index finger on the icon for 3 seconds.', (info_box_top_left[0] + 10, info_box_top_left[1] + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, '- To adjust, move the thumb away', (info_box_top_left[0] + 10, info_box_top_left[1] + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, 'from the index finger and bring them', (info_box_top_left[0] + 10, info_box_top_left[1] + 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, 'closer together.', (info_box_top_left[0] + 10, info_box_top_left[1] + 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, '- Select STOP to deselect an option.', (info_box_top_left[0] + 10, info_box_top_left[1] + 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Size and position of the "Close" button
    close_button_size = 40
    close_button_pos = (info_box_bottom_right[0] + 15, info_box_top_left[1])

    # Draw the "Close" button
    cv2.rectangle(image, close_button_pos,
                  (close_button_pos[0] + close_button_size, close_button_pos[1] + close_button_size), (0, 0, 255), -1)
    cv2.putText(image, 'X', (close_button_pos[0] + 10, close_button_pos[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    return close_button_pos, close_button_size
def close_info_button_hover(index_point, close_button_pos, close_button_size):
    if close_button_pos[0] < index_point[0] < close_button_pos[0] + close_button_size and close_button_pos[1] < index_point[1] < close_button_pos[1] + close_button_size:
        return True
    return False

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()  # Get the volume range (min, max)
min_volume = volume_range[0]
max_volume = volume_range[1]
initial_volume = (min_volume + (max_volume - min_volume) * 0.5)  # Calculate the value for 50%


min_distance = 50
max_distance = 300

# Icon positions on the screen
icon_info_pos = (560, 20)  # Top right
icon_brightness_pos = (100, 230)  # Left
icon_volume_pos = (450, 230)  # Right
icon_esc_pos = (20, 20)  # Top left
icon_stop_pos = (300, 400)  # Bottom center

# Icon sizes
icon_stop_size = 60
icon_selected_stop_size = 70
icon_info_size = 60
icon_selected_info_size = 70
icon_esc_size = 60
icon_selected_esc_size = 70
icon_size = 100
icon_selected_size = 110

# Variable for the selected option
selected_option = None  # 'brightness' or 'volume'
hover_start_time = None  # Time when the finger started hovering over the icon
info_displayed = False
close_button_pos = None
close_button_size = None

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Could not access the camera.")
            break

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        # Hand detection
        results = hands.process(image)

        # Convert the image back to BGR for display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if info_displayed:
            # The large rectangle and the "Close" button
            close_button_pos, close_button_size = showInfo(image)

        # Variables to indicate if the icons are selected
        brightness_hover = False
        volume_hover = False
        esc_hover = False
        info_hover = False
        stop_hover = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert relative coordinates to pixel coordinates
                thumb_point = (int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0]))
                index_point = (int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0]))

                if info_displayed and close_info_button_hover(index_point, close_button_pos, close_button_size):
                    selected_option = 'close_info'
                    break

                # Detect if the hand is on one of the icons
                if icon_brightness_pos[0] < index_point[0] < icon_brightness_pos[0] + icon_size and icon_brightness_pos[
                    1] < index_point[1] < icon_brightness_pos[1] + icon_size:
                    brightness_hover = True
                    if hover_start_time is None:
                        hover_start_time = time.time()
                    elif time.time() - hover_start_time >= 3:
                        selected_option = 'brightness'
                        hover_start_time = None  # Reset for the next selection
                elif icon_volume_pos[0] < index_point[0] < icon_volume_pos[0] + icon_size and icon_volume_pos[1] < \
                        index_point[1] < icon_volume_pos[1] + icon_size:
                    volume_hover = True
                    if hover_start_time is None:
                        hover_start_time = time.time()
                    elif time.time() - hover_start_time >= 3:
                        selected_option = 'volume'
                        hover_start_time = None  # Reset for the next selection
                elif icon_esc_pos[0] < index_point[0] < icon_esc_pos[0] + icon_esc_size and icon_esc_pos[1] < \
                        index_point[1] < icon_esc_pos[1] + icon_esc_size:
                    esc_hover = True
                    if hover_start_time is None:
                        hover_start_time = time.time()
                    elif time.time() - hover_start_time >= 2:
                        selected_option = 'esc'
                        hover_start_time = None  # Reset for the next selection
                elif icon_info_pos[0] < index_point[0] < icon_info_pos[0] + icon_info_size and icon_info_pos[1] < \
                        index_point[1] < icon_info_pos[1] + icon_info_size:
                    info_hover = True
                    if hover_start_time is None:
                        hover_start_time = time.time()
                    elif time.time() - hover_start_time >= 1:
                        selected_option = 'info'
                        hover_start_time = None  # Reset for the next selection
                elif icon_stop_pos[0] < index_point[0] < icon_stop_pos[0] + icon_stop_size and icon_stop_pos[1] < \
                        index_point[1] < icon_stop_pos[1] + icon_stop_size:
                    stop_hover = True
                    if hover_start_time is None:
                        hover_start_time = time.time()
                    elif time.time() - hover_start_time >= 1:
                        selected_option = 'stop'
                        hover_start_time = None  # Reset for the next selection
                else:
                    hover_start_time = None  # Reset if the finger moves away from the icons

        # Draw the icons and loading bar
        if brightness_hover:
            current_size = icon_selected_size
            current_pos = (icon_brightness_pos[0] - (current_size - icon_size) // 2,
                           icon_brightness_pos[1] - (current_size - icon_size) // 2)
            apply_icon(image, 'brightness.png', current_pos, current_size)

            if hover_start_time is not None:
                elapsed_time = time.time() - hover_start_time
                load_fraction = elapsed_time / 3
                cv2.rectangle(image, current_pos, (current_pos[0] + int(current_size * load_fraction), current_pos[1]),
                              (255, 255, 255), 3)
        else:
            apply_icon(image, 'brightness.png', icon_brightness_pos, icon_size)

        if volume_hover:
            current_size = icon_selected_size
            current_pos = (icon_volume_pos[0] - (current_size - icon_size) // 2,
                           icon_volume_pos[1] - (current_size - icon_size) // 2)
            apply_icon(image, 'volume.png', current_pos, current_size)

            if hover_start_time is not None:
                elapsed_time = time.time() - hover_start_time
                load_fraction = elapsed_time / 3
                cv2.rectangle(image, current_pos, (current_pos[0] + int(current_size * load_fraction), current_pos[1]),
                              (255, 255, 255), 3)
        else:
            apply_icon(image, 'volume.png', icon_volume_pos, icon_size)

        if esc_hover:
            current_size = icon_selected_esc_size
            current_pos = (icon_esc_pos[0] - (current_size - icon_esc_size) // 2,
                           icon_esc_pos[1] - (current_size - icon_esc_size) // 2)
            if hover_start_time is not None:
                elapsed_time = time.time() - hover_start_time
                cv2.rectangle(image, current_pos,
                              (current_pos[0] + current_size, current_pos[1] + current_size), (0, 0, 255), -1)
                cv2.putText(image, 'ESC', (current_pos[0] + 5, current_pos[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
                load_fraction = elapsed_time / 2
                cv2.rectangle(image, current_pos, (current_pos[0] + int(current_size * load_fraction), current_pos[1]),
                              (255, 255, 255), 3)
        else:
            cv2.rectangle(image, icon_esc_pos, (icon_esc_pos[0] + icon_esc_size, icon_esc_pos[1] + icon_esc_size),
                          (0, 0, 255), -1)
            cv2.putText(image, 'ESC', (icon_esc_pos[0], icon_esc_pos[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

        if info_hover:
            current_size = icon_selected_info_size
            radius = current_size // 2
            center_pos = (icon_info_pos[0] + radius, icon_info_pos[1] + radius)
            if hover_start_time is not None:
                elapsed_time = time.time() - hover_start_time
                cv2.circle(image, center_pos, radius, (0, 0, 0), -1)
                cv2.putText(image, 'INFO', (center_pos[0] - 25, center_pos[1] + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                load_fraction = elapsed_time / 1
                cv2.circle(image, center_pos, int(radius * load_fraction), (255, 255, 255), 3)
        else:
            radius = icon_info_size // 2
            center_pos = (icon_info_pos[0] + radius, icon_info_pos[1] + radius)
            cv2.circle(image, center_pos, radius, (0, 0, 0), -1)
            cv2.putText(image, 'INFO', (center_pos[0] - 25, center_pos[1] + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

        if stop_hover:
            current_size = icon_selected_stop_size
            current_pos = (icon_stop_pos[0] - (current_size - icon_stop_size) // 2,
                           icon_stop_pos[1] - (current_size - icon_stop_size) // 2)
            if hover_start_time is not None:
                elapsed_time = time.time() - hover_start_time
                cv2.rectangle(image, current_pos,
                              (current_pos[0] + current_size, current_pos[1] + current_size), (128, 128, 128), -1)
                cv2.putText(image, 'STOP', (current_pos[0] + 5, current_pos[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                            (255, 255, 255), 2)
                load_fraction = elapsed_time / 1
                cv2.rectangle(image, current_pos, (current_pos[0] + int(current_size * load_fraction), current_pos[1]),
                              (255, 255, 255), 3)
        else:
            cv2.rectangle(image, icon_stop_pos, (icon_stop_pos[0] + icon_stop_size, icon_stop_pos[1] + icon_stop_size),
                          (128, 128, 128), -1)
            cv2.putText(image, 'STOP', (icon_stop_pos[0], icon_stop_pos[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                        (255, 255, 255), 2)

        if info_displayed:
            close_button_pos, close_button_size = showInfo(image)

        if selected_option:
            distance = calculate_distance(thumb_point, index_point)
            distance_normalized = min(max(distance, min_distance), max_distance)
            if selected_option == 'brightness':
                brightness = ((distance_normalized - min_distance) / (max_distance - min_distance)) * 100
                sbc.set_brightness(brightness)

                cv2.putText(image, f'BRIGHT: {brightness:.0f}%', (icon_esc_pos[0] + icon_esc_size + 20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                print(f"Brightness set to: {brightness:.2f}%")
            elif selected_option == 'volume':
                volume_scalar = (distance_normalized - min_distance) / (max_distance - min_distance)
                volume_scalar = min(max(volume_scalar, 0.0), 1.0)
                volume.SetMasterVolumeLevelScalar(volume_scalar, None)
                volume_percent = volume_scalar * 100

                cv2.putText(image, f'VOL: {volume_percent:.0f}%', (icon_info_pos[0] - 140, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                print(f"Volume set to: {volume_percent:.2f}%")
            elif selected_option == 'esc':
                break
            elif selected_option == 'info':
                info_displayed = True
            elif selected_option == 'close_info':
                info_displayed = False
                selected_option = None
            elif selected_option == 'stop':
                selected_option = None

        cv2.imshow('Hand Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
