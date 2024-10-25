from . import tts
from .tts import TTS
from .tts_v2 import TTSV2
from .version import __version__
from .sentence_tokenizer import ChineseSentenceTokenizer

__all__ = [
    "TTS",
    "TTSV2",
    "__version__",
    "ChineseSentenceTokenizer",
]

from livekit.agents import Plugin
from .log import logger

class DashScopePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

Plugin.register_plugin(DashScopePlugin())
