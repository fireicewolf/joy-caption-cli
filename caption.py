import argparse
import os
from datetime import datetime
from pathlib import Path

from utils.download import download
from utils.joy import Joy
from utils.logger import Logger

DEFAULT_USER_PROMPT = """
A descriptive caption for this image:\n
"""


def main(args):
    # Set logger
    workspace_path = os.getcwd()
    data_dir_path = Path(args.data_path)

    log_file_path = data_dir_path.parent if os.path.exists(data_dir_path.parent) else workspace_path

    if args.custom_caption_save_path:
        log_file_path = Path(args.custom_caption_save_path)

    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # caption_failed_list_file = f'Caption_failed_list_{log_time}.txt'

    if os.path.exists(data_dir_path):
        log_name = os.path.basename(data_dir_path)

    else:
        print(f'{data_dir_path} NOT FOUND!!!')
        raise FileNotFoundError

    if args.save_logs:
        log_file = f'Caption_{log_name}_{log_time}.log' if log_name else f'test_{log_time}.log'
        log_file = os.path.join(log_file_path, log_file) \
            if os.path.exists(log_file_path) else os.path.join(os.getcwd(), log_file)
    else:
        log_file = None

    if str(args.log_level).lower() in 'debug, info, warning, error, critical':
        my_logger = Logger(args.log_level, log_file).logger
        my_logger.info(f'Set log level to "{args.log_level}"')

    else:
        my_logger = Logger('INFO', log_file).logger
        my_logger.warning('Invalid log level, set log level to "INFO"!')

    if args.save_logs:
        my_logger.info(f'Log file will be saved as "{log_file}".')

    # Check custom models path
    config_file = os.path.join(Path(__file__).parent, 'configs', 'default.json') \
        if args.config == "default.json" else Path(args.config)

    # Download models
    if os.path.exists(Path(args.models_save_path)):
        models_save_path = Path(args.models_save_path)
    else:
        models_save_path = Path(os.path.join(Path(__file__).parent, args.models_save_path))

    image_adapter_path, clip_path, llm_path = download(
        logger=my_logger,
        config_file=config_file,
        model_name=str(args.model_name),
        model_site=str(args.model_site),
        models_save_path=models_save_path,
        use_sdk_cache=True if args.use_sdk_cache else False,
        download_method=str(args.download_method)
    )

    # Load models
    my_joy = Joy(
        logger=my_logger,
        args=args,
        image_adapter_path=image_adapter_path,
        clip_path=clip_path,
        llm_path=llm_path
    )
    my_joy.load_model()

    # Inference
    my_joy.inference()

    # Unload models
    my_joy.unload_model()


def setup_args() -> argparse.ArgumentParser:
    args = argparse.ArgumentParser()

    args.add_argument(
        'data_path',
        type=str,
        help='path for data.'
    )
    args.add_argument(
        '--recursive',
        action='store_true',
        help='Include recursive dirs'
    )
    args.add_argument(
        '--config',
        type=str,
        default='default.json',
        help='config json for llava models, default is "default.json"'
    )
    # args.add_argument(
    #     '--use_cpu',
    #     action='store_true',
    #     help='use cpu for inference.'
    # )
    args.add_argument(
        '--image_size',
        type=int,
        default=1024,
        help='resize image to suitable, default is 1024.'
    )
    args.add_argument(
        '--model_name',
        type=str,
        default='Joy-Caption-Pre-Alpha',
        help='model name for inference, default is "Joy-Caption-Pre-Alpha", please check configs/default.json'
    )
    args.add_argument(
        '--model_site',
        type=str,
        choices=['huggingface', 'modelscope'],
        default='huggingface',
        help='download model from model site huggingface or modelscope, default is "huggingface".'
    )
    args.add_argument(
        '--models_save_path',
        type=str,
        default="models",
        help='path to save models, default is "models".'
    )
    args.add_argument(
        '--use_sdk_cache',
        action='store_true',
        help='use sdk\'s cache dir to store models. \
            if this option enabled, "--models_save_path" will be ignored.'
    )
    args.add_argument(
        '--download_method',
        type=str,
        choices=["SDK", "URL"],
        default='SDK',
        help='download method via SDK or URL, default is "SDK".'
    )
    args.add_argument(
        '--custom_caption_save_path',
        type=str,
        default=None,
        help='Input custom caption file save path.'
    )
    args.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='set log level, default is "INFO"'
    )
    args.add_argument(
        '--save_logs',
        action='store_true',
        help='save log file.'
    )
    args.add_argument(
        '--caption_extension',
        type=str,
        default='.txt',
        help='extension of caption file, default is ".txt"'
    )
    args.add_argument(
        '--not_overwrite',
        action='store_true',
        help='not overwrite caption file if exist.'
    )
    args.add_argument(
        '--user_prompt',
        type=str,
        default=DEFAULT_USER_PROMPT,
        help='user prompt for caption.'
    )
    args.add_argument(
        '--temperature',
        type=float,
        default=0.5,
        help='temperature for Llama model.'
    )
    args.add_argument(
        '--max_tokens',
        type=int,
        default=300,
        help='max tokens for output.'
    )

    return args


if __name__ == "__main__":
    args = setup_args()
    args = args.parse_args()
    main(args)
