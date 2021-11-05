"""Have a look at the data"""
from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
import PIL

DATA_PATH = Path(__file__).parents[2] / 'elixir_data'


def main():
    for tile_path in (DATA_PATH / 'camelyon16tiles').glob('*.tif'):
        tile_image = PIL.Image.open(tile_path)
        tile_image.thumbnail(tile_image.size)
    data_index = pd.read_csv(DATA_PATH / 'index_camelyon16_with_metadata_all.csv')
    print(data_index)

if __name__ == '__main__':
    main()