import logging, os

def prepare_logging(file_name: str = "logs/app.log"):
    # Get the directory where your script is located
    log_path = os.path.join(os.getcwd(), file_name)

    # 1. Setup the File Handler
    file_handler = logging.FileHandler(log_path,  mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 2. Setup the Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 3. Apply the Config
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[file_handler, console_handler]
    )

