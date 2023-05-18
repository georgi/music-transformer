from src.data import convert_maestro_to_proto
import dotenv
import hydra
from omegaconf import DictConfig
from src.data import PerformanceEncoder

"""
This scripts downloads the MAESTRO dataset and converts it to the proto format.
"""

data_dir = "data"

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    encoder = PerformanceEncoder.from_config(config)
    convert_maestro_to_proto(data_dir, encoder)

if __name__ == "__main__":
    main()

