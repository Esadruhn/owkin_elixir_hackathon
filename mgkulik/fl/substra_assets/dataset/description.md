# Dataset opener

This is the script to load the data from the Camelyon challenge.
It returns the path to the data, so that instead of loading everything into memory at once, they
are loaded at runtime in the algo.

How to load the data:

```python
import PIL

tile_image = PIL.Image.open(tile_path)
```

For the labels, it returns 0 if normal, 1 if tumoral.
