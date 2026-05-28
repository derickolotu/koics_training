import os
import imghdr


def rename_images(directory):
    if not os.path.isdir(directory):
        print(f"The specified directory '{directory}' does not exist.")
        return

    files = os.listdir(directory)
    image_count = 1

    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and imghdr.what(file_path):
            extension = os.path.splitext(file)[1]
            new_name = f"Img{image_count}{extension}"
            new_path = os.path.join(directory, new_name)

            os.rename(file_path, new_path)
            image_count += 1


rename_images("custom_object/data_images")
