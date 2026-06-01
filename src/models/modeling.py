# Compatibility shim: Wang/TALL head_*.pt files reference src.models.modeling.
# Re-exports from src.vision.modeling so the pickles unpickle here.
from src.vision.modeling import *  # noqa: F401, F403
