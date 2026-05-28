import cv2
from easyocr import Reader
from PIL import ImageFont, ImageDraw, Image
import numpy as np


languages_list = ["en", "es"]
gpu = True
reader = Reader(languages_list, gpu)
font_path = "calibri.ttf"


def write_text(text, x, y, img, color=(50, 50, 255), font_size=22):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y - font_size), text, font=font, fill=color)
    img = np.array(img_pil)
    return img


def draw_box(img, lt, br, color=(255, 0, 50), thickness=2):
    cv2.rectangle(img, lt, br, color=color, thickness=thickness)
    return img


def box_cords(box):
    (lt, rt, br, bl) = box
    lt = (int(lt[0]), int(lt[1]))
    rt = (int(rt[0]), int(rt[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    return lt, rt, br, bl


cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 24.0, (frame_width, frame_height))

text = ""

while True:
    ret, frame = cam.read()

    out.write(frame)
    frame_copy = frame.copy()
    result = reader.readtext(frame_copy)
    for box, text, conf in result:
        lt, rt, br, bl = box_cords(box)
        frame_copy = write_text(text, lt[0], lt[1], frame_copy)
        frame_copy = draw_box(frame_copy, lt, br)
        text += text + " "

    print(text)

    cv2.imshow("Camera", frame_copy)

    if cv2.waitKey(1) == ord("q"):
        break


cam.release()
out.release()
cv2.destroyAllWindows()
