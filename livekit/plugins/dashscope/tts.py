from __future__ import annotations

import os
import base64
from dataclasses import dataclass
from typing import AsyncContextManager, Literal

import dashscope
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse

from dashscope.audio.tts import SpeechSynthesizer, SpeechSynthesisResult, ResultCallback
from livekit.agents import tts, utils
from livekit import rtc

from .log import logger

# DashScope TTS 默认音频参数
DASHSCOPE_TTS_SAMPLE_RATE = 24000
DASHSCOPE_TTS_CHANNELS = 1

AudioFormat = Literal["wav", "mp3", "pcm"]

@dataclass 
class _TTSOptions:
    model: str
    sample_rate: int
    format: AudioFormat
    volume: int  # 音量 0-100
    rate: float  # 语速 0.5-2.0
    pitch: float  # 音调 0.5-2.0
    word_timestamp_enabled: bool  # 是否开启字级别时间戳
    phoneme_timestamp_enabled: bool  # 是否开启音素级别时间戳

class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = "sambert-zhichu-v1",
        sample_rate: int = DASHSCOPE_TTS_SAMPLE_RATE,
        format: AudioFormat = "wav",
        volume: int = 50,
        rate: float = 1.0,
        pitch: float = 1.0,
        word_timestamp_enabled: bool = False,
        phoneme_timestamp_enabled: bool = False,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=DASHSCOPE_TTS_CHANNELS,
        )

        # 验证参数
        if not (0 <= volume <= 100):
            raise ValueError("Volume must be between 0 and 100")
        if not (0.5 <= rate <= 2.0):
            raise ValueError("Rate must be between 0.5 and 2.0")
        if not (0.5 <= pitch <= 2.0):
            raise ValueError("Pitch must be between 0.5 and 2.0")

        # 验证API key
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError("DashScope API key is required")

        dashscope.api_key = api_key
        
        self._opts = _TTSOptions(
            model=model,
            sample_rate=sample_rate,
            format=format,
            volume=volume,
            rate=rate,
            pitch=pitch,
            word_timestamp_enabled=word_timestamp_enabled,
            phoneme_timestamp_enabled=phoneme_timestamp_enabled
        )

    def synthesize(self, text: str) -> ChunkedStream:
        return ChunkedStream(
            text=text,
            opts=self._opts
        )

class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        text: str,
        opts: _TTSOptions,
    ) -> None:
        super().__init__()
        self._text = text
        self._opts = opts

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()

        class TTSCallback(ResultCallback):
            def __init__(self, stream):
                self.stream = stream
                self.audio_bstream = utils.audio.AudioByteStream(
                    sample_rate=DASHSCOPE_TTS_SAMPLE_RATE,
                    num_channels=DASHSCOPE_TTS_CHANNELS,
                )

            def on_event(self, result: SpeechSynthesisResult):
                if result.get_audio_frame() is not None:
                    audio_data = result.get_audio_frame()
                    for frame in self.audio_bstream.write(audio_data):
                        self.stream._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                segment_id=segment_id,
                                frame=frame,
                            )
                        )

                if result.get_timestamp() is not None and self._opts.word_timestamp_enabled:
                    logger.debug(f"Word timestamps: {result.get_timestamp()}")

            def on_error(self, response: SpeechSynthesisResponse):
                logger.error(f"DashScope TTS error: {response}")
                raise Exception(f"DashScope TTS error: {response}")

        callback = TTSCallback(self)

        # 调用DashScope TTS API
        SpeechSynthesizer.call(
            model=self._opts.model,
            text=self._text,
            sample_rate=self._opts.sample_rate,
            format=self._opts.format,
            volume=self._opts.volume,
            rate=self._opts.rate,
            pitch=self._opts.pitch,
            callback=callback,
            word_timestamp_enabled=self._opts.word_timestamp_enabled,
            phoneme_timestamp_enabled=self._opts.phoneme_timestamp_enabled
        )

        # 处理剩余数据
        for frame in callback.audio_bstream.flush():
            self._event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id=request_id,
                    segment_id=segment_id,
                    frame=frame,
                )
            )
