from dotenv import load_dotenv
from pyaml_env import parse_config
load_dotenv()
config = parse_config('config.yml', encoding = 'utf-8')
config_ml = parse_config('config_ml.yml', encoding = 'utf-8')