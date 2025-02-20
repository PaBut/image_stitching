import os
from cv2 import Mat
import cv2
import natsort


class TilesLoader:
    __tiles: dict[(int, int), Mat]
    __width: int
    __height: int
    def __init__(self, path: str, width: int, height: int):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        files = natsort.natsorted(files)
        self.__tiles = {}
        self.__width = width
        self.__height = height

        for i in range(len(files)):
            self.__tiles[(i % (width + 1), i // (width + 1))] = cv2.imread(files[i])

    def get_tile(self, x: int, y: int) -> Mat:
        return self.__tiles[(x, y)]
    
    def width(self) -> int:
        return self.__width
    
    def height(self) -> int:
        return self.__height