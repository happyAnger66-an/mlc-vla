from inspect import signature, Parameter
from typing import Dict, Any
from importlib import import_module


def build_from_cfg(cfg: Dict, auto_instantiation: bool = True, type_key: str = 'type') -> Any:
    """
    Build and return an object based on a configuration dictionary.

    Args:
        cfg (dict): A configuration dictionary that must include the key "type", 
                    specifying the fully qualified name of the target class.
                    Additional key-value pairs are treated as arguments for instantiation.
        auto_instantiation (bool, optional): If True, automatically instantiate the class 
                                        with the provided arguments. If False, 
                                        return the class reference instead.
                                        Default is True.
        type_key (str, optional): The key in the configuration dictionary that specifies 
                                  the fully qualified name of the target class.
                                  Default is 'type'.
    Returns:
        Any: The constructed object if `auto_instantiation` is True, 
            or the class reference if False.
    """
    if type_key not in cfg:
        raise KeyError(f"The configuration dictionary must contain the key '{type_key}'.")

    # Copy the configuration to avoid modifying the original
    args = cfg.copy()
    obj_type = args.pop(type_key)  # Extract and remove the "type" key from the arguments

    # Split the fully qualified name into module path and object name
    try:
        module_path, object_name = obj_type.rsplit('.', 1)
    except ValueError:
        raise ValueError(
            f"Invalid '{type_key}' format: '{obj_type}'. It must be in 'module.submodule.ClassName' format."
        )

    try:
        module = import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}'. Error: {e}")

    tgt_cls = getattr(module, object_name)

    if auto_instantiation:
        init_params = signature(tgt_cls.__init__).parameters
        # Check if **kwargs is present in __init__
        has_kwargs = any(
            param.kind == Parameter.VAR_KEYWORD for param in init_params.values()
        )
        if not has_kwargs:
            # Get the valid parameters for the class constructor
            args = {k: v for k, v in args.items() if k in init_params}
        return tgt_cls(**args)
    else:
        return tgt_cls