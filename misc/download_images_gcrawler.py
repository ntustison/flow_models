"""
https://icrawler.readthedocs.io/en/latest/builtin.html
"""

from icrawler.builtin import GoogleImageCrawler
import os

# Step 1: Download images
output_dir = 'data/train/images'
os.makedirs(output_dir, exist_ok=True)

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=1,
    downloader_threads=4,
    storage={'root_dir': output_dir})
# filters = dict(
#     size='large',
#     color='orange',
#     license='commercial,modify',
#     date=((2017, 1, 1), (2017, 11, 30))
# )
google_crawler.crawl(
    keyword='cat photos',
    #filters=filters,
    offset=0,
    max_num=5,  # 1000,
    min_size=(128, 128),
    max_size=None,  # (1500, 1500),
    file_idx_offset=0
)

print("Download completed.")
