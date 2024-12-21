import configparser

def get_file_path_from_config():
  current_directory = os.path.dirname(os.path.abspath(__file__))
  config_file = Path(current_directory) / 'config.ini'

  config = configparser.ConfigParser()
  config.read(config_file)
  file_path = config.get('DEFAULT', 'FILE_PATH')
  return file_path