For Yahoo Learning-to-Rank dataset

1. Download the dataset from [HuggingFace](https://huggingface.co/datasets/YahooResearch/Yahoo-Learning-to-Rank-Challenge).
2. Create the tfds manual download directory:
   ```bash
   mkdir -p ~/tensorflow_datasets/downloads/manual
   ```
3. Extract the downloaded `dataset.tgz` and place `ltrc_yahoo.tar.bz2` into that directory.

MSLR can be downloaded via tfds, but at the moment the download function is [bugged](https://github.com/tensorflow/datasets/pull/11146). 
To download the data:
1. Navigate to [MSLR official website](https://www.microsoft.com/en-us/research/project/mslr/) using your browser.
2. Accept the online agreement.
3. Click on MSLR-WEB30K, opening a new tab.
4. In the new tab, right-click and choose Inspect/Developer Tools, click Network (and if needed reload the page).
5. Click Download, which will create a new `download.aspx?UniqueId=...` request. Copy the whole request, including `https://my.microsoftpersonalcontent.com/personal/...`.
6. Navigate to your local `tensorflow-datasets` installation and open the `mslr_web.py` file (e.g. `/opt/conda/lib/python3.11/site-packages/tensorflow_datasets/ranking/mslr_web/mslr_web.py`) in a text editor.
7. Replace `_URLS`'s `"30K"` entry on line 65 with the copied request string (make sure it's still a valid python string).