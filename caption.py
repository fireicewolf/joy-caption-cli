import argparse
import os
import time
from datetime import datetime
from pathlib import Path

from utils.download import download_models
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

    image_adapter_path, clip_path, llm_path = download_models(
        logger=my_logger,
        args=args,
        config_file=config_file,
        models_save_path=models_save_path,
    )
    # Load models
    my_joy = Joy(
        logger=my_logger,
        args=args,
        image_adapter_path=image_adapter_path,
        clip_path=clip_path,
        llm_path=llm_path,
        use_gpu=True if not args.llm_use_cpu else False
    )
    my_joy.load_model()

    # Inference
    start_inference_time = time.monotonic()
    my_joy.inference()
    total_inference_time = time.monotonic() - start_inference_time
    days = total_inference_time // (24 * 3600)
    total_inference_time %= (24 * 3600)
    hours = total_inference_time // 3600
    total_inference_time %= 3600
    minutes = total_inference_time // 60
    seconds = total_inference_time % 60
    days = f"{days} Day(s) " if days > 0 else ""
    hours = f"{hours} Hour(s) " if hours > 0 or (days and hours == 0) else ""
    minutes = f"{minutes} Min(s) " if minutes > 0 or (hours and minutes == 0) else ""
    seconds = f"{seconds:.1f} Sec(s)"
    my_logger.info(f"All work done with in {days}{hours}{minutes}{seconds}.")

    # Unload models
    my_joy.unload_model()


def setup_args() -> argparse.ArgumentParser:
    parsed_args = argparse.ArgumentParser()
    base_args = parsed_args.add_argument_group("Base")
    base_args.add_argument(
        'data_path',
        type=str,
        help='path for data.'
    )
    base_args.add_argument(
        '--recursive',
        action='store_true',
        help='Include recursive dirs'
    )

    log_args = parsed_args.add_argument_group("Logs")
    log_args.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='set log level, default is "INFO"'
    )
    log_args.add_argument(
        '--save_logs',
        action='store_true',
        help='save log file.'
    )

    download_args = parsed_args.add_argument_group("Download")
    download_args.add_argument(
        '--config',
        type=str,
        default='default.json',
        help='config json for llava models, default is "default.json"'
    )
    download_args.add_argument(
        '--model_name',
        type=str,
        default='Joy-Caption-Pre-Alpha',
        help='model name for inference, default is "Joy-Caption-Pre-Alpha", please check configs/default.json'
    )
    download_args.add_argument(
        '--model_site',
        type=str,
        choices=['huggingface', 'modelscope'],
        default='huggingface',
        help='download model from model site huggingface or modelscope, default is "huggingface".'
    )
    download_args.add_argument(
        '--models_save_path',
        type=str,
        default="models",
        help='path to save models, default is "models".'
    )
    download_args.add_argument(
        '--use_sdk_cache',
        action='store_true',
        help='use sdk\'s cache dir to store models. \
            if this option enabled, "--models_save_path" will be ignored.'
    )
    download_args.add_argument(
        '--download_method',
        type=str,
        choices=["SDK", "URL"],
        default='SDK',
        help='download method via SDK or URL, default is "SDK".'
    )
    download_args.add_argument(
        '--force_download',
        action='store_true',
        help='force download even file exists.'
    )
    download_args.add_argument(
        '--skip_download',
        action='store_true',
        help='skip download if exists.'
    )
    download_args.add_argument(
        '--custom_caption_save_path',
        type=str,
        default=None,
        help='Input custom caption file save path.'
    )

    inference_args = parsed_args.add_argument_group("Inference")
    inference_args.add_argument(
        '--llm_use_cpu',
        action='store_true',
        help='use cpu for inference.'
    )
    inference_args.add_argument(
        '--llm_dtype',
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default='fp16',
        help='choice joy LLM load dtype, default is `auto`.'
    )
    inference_args.add_argument(
        '--llm_qnt',
        type=str,
        choices=["none", "4bit", "8bit"],
        default='none',
        help='Enable quantization for LLM ["none","4bit", "8bit"]. default is `none`.'
    )
    inference_args.add_argument(
        '--image_size',
        type=int,
        default=1024,
        help='resize image to suitable, default is 1024.'
    )
    inference_args.add_argument(
        '--caption_extension',
        type=str,
        default='.txt',
        help='extension of caption file, default is ".txt"'
    )
    inference_args.add_argument(
        '--not_overwrite',
        action='store_true',
        help='not overwrite caption file if exist.'
    )
    inference_args.add_argument(
        '--user_prompt',
        type=str,
        default=DEFAULT_USER_PROMPT,
        help='user prompt for caption.'
    )
    inference_args.add_argument(
        '--temperature',
        type=float,
        default=0.5,
        help='temperature for Llama model.'
    )
    inference_args.add_argument(
        '--max_tokens',
        type=int,
        default=300,
        help='max tokens for output.'
    )
    return parsed_args


if __name__ == "__main__":
    get_args = setup_args()
    get_args = get_args.parse_args()
    main(get_args)
