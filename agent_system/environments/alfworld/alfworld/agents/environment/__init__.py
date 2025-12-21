def get_environment(env_type):
    if env_type == 'AlfredTWEnv':
        from .alfred_tw_env import AlfredTWEnv
        return AlfredTWEnv
    elif env_type == 'AlfredThorEnv':
        from .alfred_thor_env import AlfredThorEnv
        return AlfredThorEnv
    elif env_type == 'AlfredHybrid':
        from .alfred_hybrid import AlfredHybrid
        return AlfredHybrid
    else:
        raise NotImplementedError(f"Environment {env_type} is not implemented.")
