"""
https://google-images-download.readthedocs.io/en/latest/examples.html
"""

from google_images_download import google_images_download
import os

# Define function to download images
def download_images(query, num_images, output_dir):
    response = google_images_download.googleimagesdownload()
    arguments = {
        "keywords": query,
        "limit": num_images,
        "print_urls": False,
        "format": "jpg",
        "size": "medium",  # to get images between 100x100 and 1000x1000
        "output_directory": output_dir
    }
    paths = response.download(arguments)
    return paths

# Download 500 cat images
download_images("cat", 5, "cat_images")

