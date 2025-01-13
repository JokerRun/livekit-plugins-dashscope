import asyncio
from dataclasses import dataclass
from typing import AsyncIterator
from websocket import WebSocketConnectionClosedException
import dashscope
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.tts_v2 import *


@dataclass
class CosyTTSConfig():
    api_key: str = ""
    voice: str = "longxiaochun"
    model: str = "cosyvoice-v1"
    sample_rate: int = 16000


class AsyncIteratorCallback(ResultCallback):
    def __init__(self, queue: asyncio.Queue) -> None:
        self.closed = False
        self.loop = asyncio.get_event_loop()
        self.queue = queue

    def close(self):
        self.closed = True

    # def on_open(self):
    #
    # def on_complete(self):
    #
    # def on_error(self, message: str):

    def on_close(self):
        self.close()

    # def on_event(self, message: str) -> None:
    #     self.ten_env.log_debug(f"received event: {message}")

    def on_data(self, data: bytes) -> None:
        if self.closed:
            return
        asyncio.run_coroutine_threadsafe(self.queue.put(data), self.loop)


class CosyTTS:
    def __init__(self, config: CosyTTSConfig) -> None:
        self.config = config
        self.synthesizer = None  # Initially no synthesizer
        self.queue = asyncio.Queue()
        dashscope.api_key = config.api_key

    def _create_synthesizer(self, callback: AsyncIteratorCallback):
        if self.synthesizer:
            self.synthesizer = None
        
        ten_env.log_info("Creating new synthesizer")
        self.synthesizer = SpeechSynthesizer(
            model=self.config.model,
            voice=self.config.voice,
            format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            callback=callback,
        )

    async def get_audio_bytes(self) -> bytes:
        return await self.queue.get()

    def text_to_speech_stream(
        self,  text: str, end_of_segment: bool
    ) -> None:
        try:
            callback = AsyncIteratorCallback( self.queue)

            if not self.synthesizer or end_of_segment:
                self._create_synthesizer( callback)

            self.synthesizer.streaming_call(text)

            if end_of_segment:
                self.synthesizer.streaming_complete()
                self.synthesizer = None
        except WebSocketConnectionClosedException as e:
            self.synthesizer = None
        except Exception as e:
            self.synthesizer = None

    def cancel(self) -> None:
        if self.synthesizer:
            try:
                self.synthesizer.streaming_cancel()
            except WebSocketConnectionClosedException as e:
                pass
            except Exception as e:
                pass
            self.synthesizer = None
