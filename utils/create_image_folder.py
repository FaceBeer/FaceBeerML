import argparse
from pathlib import Path
import sys
import time

from picamera import PiCamera


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name")
    args = parser.parse_args()
    name = args.name
    assert name is not None

    data_path = Path("../data")
    new_path = data_path / name
    if new_path.exists():
        new_path.rmdir()
    new_path.mkdir()

    camera = PiCamera()
    camera.resolution = (1920, 1080)
    time.sleep(2)

    for i in range(500):
        if i % 50 == 0:
            print(f"Image {i}/500.")
        filename = str(new_path/str(i).zfill(5))+'.jpg'
        camera.capture(filename)
#
# camera.capture("/home/pi/Pictures/img.jpg")
# print("Done.")