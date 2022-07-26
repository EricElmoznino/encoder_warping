import hydra
from omegaconf import DictConfig
from models import get_model


@hydra.main(version_base=None, config_path="configurations", config_name="encoding_dim")
def main(cfg: DictConfig) -> None:
    model, layer_groups, image_transform = get_model(cfg.model.arch, cfg.model.layer)


if __name__ == "__main__":
    main()
