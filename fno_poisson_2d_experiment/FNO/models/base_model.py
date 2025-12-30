import inspect
import torch
import warnings
from pathlib import Path


class BaseModel(torch.nn.Module):
    """Base class for all Models"""

    _models = dict()
    _version = "0.1.0"

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            BaseModel._models[name.lower()] = cls
            cls._name = name
        else:
            BaseModel._models[cls.__name__.lower()] = cls
            cls._name = cls.__name__

    def __new__(cls, *args, **kwargs):
        sig = inspect.signature(cls)
        model_name = cls.__name__
        verbose = kwargs.get("verbose", False)

        # Warn on unexpected kwargs
        for key in kwargs:
            if key not in sig.parameters:
                if verbose:
                    print(
                        f"Given argument key={key} "
                        f"that is not in {model_name}'s signature."
                    )

        # Fill defaults
        for key, value in sig.parameters.items():
            if (value.default is not inspect._empty) and (key not in kwargs):
                kwargs[key] = value.default

        if hasattr(cls, "_version"):
            kwargs["_version"] = cls._version
        kwargs["args"] = args
        kwargs["_name"] = cls._name

        instance = super().__new__(cls)
        instance._init_kwargs = kwargs
        return instance

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        if state_dict.get("_metadata") is None:
            state_dict["_metadata"] = self._init_kwargs
        else:
            warnings.warn(
                "Attempting to update metadata for a module with metadata already present."
            )
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        metadata = state_dict.pop("_metadata", None)

        if metadata is not None:
            saved_version = metadata.get("_version", None)
            if saved_version != self._version:
                warnings.warn(
                    f"Loading model version {saved_version}, "
                    f"but current version is {self._version}"
                )

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def save_checkpoint(self, save_folder, save_name):
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.state_dict(),
            save_folder / f"{save_name}_state_dict.pt",
        )
        torch.save(
            self._init_kwargs,
            save_folder / f"{save_name}_metadata.pkl",
        )

    def load_checkpoint(self, save_folder, save_name, map_location=None):
        state_dict = torch.load(
            Path(save_folder) / f"{save_name}_state_dict.pt",
            map_location=map_location,
            weights_only=False,
        )
        self.load_state_dict(state_dict)

    @classmethod
    def from_checkpoint(cls, save_folder, save_name, map_location=None):
        init_kwargs = torch.load(
            Path(save_folder) / f"{save_name}_metadata.pkl",
            weights_only=False,
        )

        init_kwargs.pop("_name", None)
        init_kwargs.pop("_version", None)

        args = init_kwargs.pop("args", [])
        instance = cls(*args, **init_kwargs)
        instance.load_checkpoint(save_folder, save_name, map_location)
        return instance


def available_models():
    return list(BaseModel._models.keys())


def get_model(config):
    if "model" not in config:
        raise KeyError("Expected config.model")

    arch = config.model["model_arch"].lower()
    model_config = config.model.copy()
    model_config.pop("model_arch", None)

    data_channels = model_config.pop("data_channels")
    patching_levels = config.get("patching", {}).get("levels", 0)
    if patching_levels:
        data_channels *= patching_levels + 1

    model_config["in_channels"] = data_channels

    try:
        return BaseModel._models[arch](**model_config)
    except KeyError:
        raise ValueError(
            f"Got model_arch={arch}, expected one of {available_models()}"
        )
