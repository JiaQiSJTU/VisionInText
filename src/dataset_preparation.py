# encoding = "utf-8"
import json
import os
from tqdm import tqdm
from collections import defaultdict
import argparse
import random
random.seed(1)
from PIL import Image, ImageDraw, ImageFont

def create_image_from_ascii(ascii_art, font_path, image_size=(1600, 1200), margin=10):
    font_size = 18
    image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    dummy_text = "A"
    (_, _, _, height) = draw.textbbox((0, 0), dummy_text, font=font)
    line_height = height + 2

    y_position = 0
    for art_line in ascii_art.split('\n'):
        draw.text((10, y_position), art_line, font=font, fill='black')
        y_position += line_height

    pixels = image.load()
    width, height = image.size

    def is_row_blank(row):
        return all(pixels[x, row] == (255, 255, 255) for x in range(width))

    def is_column_blank(column):
        return all(pixels[column, y] == (255, 255, 255) for y in range(height))

    top = next((y for y in range(height) if not is_row_blank(y)), None)
    bottom = next((y for y in range(height - 1, -1, -1) if not is_row_blank(y)), None)
    left = next((x for x in range(width) if not is_column_blank(x)), None)
    right = next((x for x in range(width - 1, -1, -1) if not is_column_blank(x)), None)

    if None not in (top, bottom, left, right):
        top = max(0, top - margin)
        bottom = min(height, bottom + margin)
        left = max(0, left - margin)
        right = min(width, right + margin)
        image = image.crop((left, top, right, bottom))

    return image



if __name__=="__main__":

    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--data_dir", type=str, default="", help="data directory")
    parser.add_argument("--file_name", type=str, default="", help="train or test filename")
    parser.add_argument("--font_path", type=str, default="./dejavu-sans-mono/DejaVuSansMono.ttf")
    args = parser.parse_args()

    samples = []
    with open(os.path.join(args.data_dir, args.file_name), "r") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    if not os.path.exists(os.path.join(args.data_dir, "img")):
        os.mkdir(os.path.join(args.data_dir, "img"))

    for sample in tqdm(samples):
        
        ascii_art = sample["ascii_art"]
        image_path = sample["image_path"]

        image = create_image_from_ascii(ascii_art, args.font_path)
        image.save(os.path.join(args.data_dir, "img", image_path))

