import pathlib

from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[str(pathlib.Path(__file__).parent / 'settings.toml')]
)