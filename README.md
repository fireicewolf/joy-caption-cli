# Joy Caption Cli
A Python base cli tool for tagging images with [joy-caption-pre-alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha) models.

### Only support cuda devices in current.

## Introduce

I make this repo because I want to caption some images cross-platform (On My old MBP, my game win pc or docker base linux cloud-server(like Google colab))

But I don't want to install a huge webui just for this little work. And some cloud-service are unfriendly to gradio base ui.

So this repo born.


## Model source

Huggingface are original sources, modelscope are pure forks from Huggingface(Because HuggingFace was blocked in Some place).

|               Model               |                                HuggingFace Link                                |                                       ModelScope Link                                        |
|:---------------------------------:|:------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
|       joy-caption-pre-alpha       |     [HuggingFace](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)      |       [ModelScope](https://www.modelscope.cn/models/fireicewolf/joy-caption-pre-alpha)       |
| siglip-so400m-patch14-384(Google) |       [HuggingFace](https://huggingface.co/google/siglip-so400m-patch14-384)       |        [ModelScope](https://www.modelscope.cn/models/fireicewolf/siglip-so400m-patch14-384)        |
|         Meta-Llama-3.1-8B         |    [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)     |        [ModelScope](https://www.modelscope.cn/models/fireicewolf/Meta-Llama-3.1-8B)         |

## TO-DO

make a simple ui by Jupyter widget(When my lazy cancer cured😊)

## Installation
Python 3.10 works fine.

Open a shell terminal and follow below steps:
```shell
# Clone this repo
git clone https://github.com/fireicewolf/joy-caption-cli.git
cd joy-caption-cli

# create a Python venv
python -m venv .venv
.\venv\Scripts\activate

# Install torch
# Install torch base on your GPU driver. ex.
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
 
# Base dependencies, models for inference will download via python request libs.
pip install -U -r requirements.txt

# If you want to download or cache model via huggingface hub, install this.
pip install -U -r huggingface-requirements.txt

# If you want to download or cache model via modelscope hub, install this.
pip install -U -r modelscope-requirements.txt
```

### Take a notice
This project use llama-cpp-python as base lib, and it needs to be complied.

## Simple usage
__Make sure your python venv has been activated first!__
```shell
python caption.py your_datasets_path
```
To run with more options, You can find help by run with this or see at [Options](#options)
```shell
python caption.py -h
```

##  <span id="options">Options</span>
<details>
    <summary>Advance options</summary>
`data_path`

path for data

`--recursive`

Will include all support images format in your input datasets path and its sub-path.

`config`

config json for llava models, default is "default.json"

[//]: # (`--use_cpu`)

[//]: # ()
[//]: # (Use cpu for inference.)

`--model_name MODEL_NAME`

model name for inference, default is "Joy-Caption-Pre-Alpha", please check configs/default.json)

`--model_site MODEL_SITE`

Model site where onnx model download from(huggingface or modelscope), default is huggingface.

`--models_save_path MODEL_SAVE_PATH`

Path for models to save, default is models(under project folder).

`--download_method SDK`

Download models via sdk or url, default is sdk.

If huggingface hub or modelscope sdk not installed or download failed, will auto retry with url download.

`--use_sdk_cache`

Use huggingface or modelscope sdk cache to store models, this option need huggingface_hub or modelscope sdk installed.

If this enabled, `--models_save_path` will be ignored.

`--custom_caption_save_path CUSTOM_CAPTION_SAVE_PATH`

Save caption files to a custom path but not with images(But keep their directory structure)

`--log_level LOG_LEVEL`

Log level for terminal console and log file, default is `INFO`(`DEBUG`,`INFO`,`WARNING`,`ERROR`,`CRITICAL`)

`--save_logs`

Save logs to a file, log will be saved at same level with `data_dir_path`

`--caption_extension CAPTION_EXTENSION`

Caption file extension, default is `.txt`

`--not_overwrite`

Do not overwrite caption file if it existed.

`--user_prompt USER_PROMPT`

user prompt for caption.

`--temperature TEMPERATURE`

temperature for Llama model,default is 0.5.

`--max_tokens MAX_TOKENS`

max tokens for output, default is 300.

</details>

## Credits
Base on [oy-caption-pre-alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)

Without their works(👏👏), this repo won't exist.
