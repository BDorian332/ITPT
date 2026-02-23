from dataclasses import dataclass

@dataclass
class PipelineSettings:
    version: str = "v1"

    # preprocessing
    cropping: bool = True
    denoising: bool = True

    # postprocessing
    post_clean: bool = True
    post_merge: bool = False


SETTINGS = PipelineSettings()
