from config import read_config

config = read_config()
model_type = config.get("ModelType", "BAO")

PG_OPTIMIZER_INDEX = 0

if model_type == "GNTO":
    DEFAULT_MODEL_PATH = "bao_default_model_gnto"
    TMP_MODEL_PATH = "bao_tmp_model_gnto"
    OLD_MODEL_PATH = "bao_previous_model_gnto"
else:
    DEFAULT_MODEL_PATH = "bao_default_model"
    TMP_MODEL_PATH = "bao_tmp_model"
    OLD_MODEL_PATH = "bao_previous_model"
