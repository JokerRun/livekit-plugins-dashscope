from __future__ import annotations

import os
import base64
from dataclasses import dataclass
from typing import AsyncContextManager, Literal
import asyncio

import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback, AudioFormat
from livekit.agents import tts, utils,tokenize
from livekit import rtc
from .log import logger
# https://help.aliyun.com/zh/dashscope/developer-reference/cosyvoice-quick-start
DASHSCOPE_TTS_CHANNELS = 1
BUFFERED_WORDS_COUNT=8
@dataclass 
class _TTSOptions:
    model: str
    voice: str
    format: AudioFormat  # 使用 AudioFormat 枚举
    volume: int  # 音量 0-100
    rate: float  # 语速 0.5-2.0
    pitch: float  # 音调 0.5-2.0
    word_timestamp_enabled: bool  # 是否开启字级别时间戳
    phoneme_timestamp_enabled: bool  # 是否开启音素级别时间戳
    sent_tokenizer: tokenize.SentenceTokenizer
class TTSV2(tts.TTS):
    def __init__(
        self,
        *,
        model: str = "cosyvoice-v1",
        voice: str = "longxiaochun",
        format: AudioFormat = AudioFormat.WAV_16000HZ_MONO_16BIT,
        volume: int = 50,
        rate: float = 1.0,
        pitch: float = 1.0,
        word_timestamp_enabled: bool = False,
        phoneme_timestamp_enabled: bool = False,
        api_key: str | None = None,
        sent_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer()
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=format.sample_rate,
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
            voice=voice,
            format=format,
            volume=volume,
            rate=rate,
            pitch=pitch,
            word_timestamp_enabled=word_timestamp_enabled,
            phoneme_timestamp_enabled=phoneme_timestamp_enabled,
            sent_tokenizer=sent_tokenizer
        )

    def synthesize(self, text: str) -> tts.ChunkedStream:
        """实现非流式合成方法"""
        logger.debug(f"\033[32mDashScope TTS V2 synthesize: {text}\033[0m")
        return ChunkedStream(
            text=text,
            opts=self._opts
        )
        

    def stream(self) -> SynthesizeStream:
        """实现流式合成方法"""
        return SynthesizeStream(self._opts)

class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, opts: _TTSOptions):
        super().__init__()
        self._opts = opts
        self._sent_tokenizer_stream = opts.sent_tokenizer.stream()
        self._callback_queue = utils.aio.Chan[tts.SynthesizedAudio]()
        self._complete_event = asyncio.Event()  # 添加完成事件

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:

        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()

        callback = self.Callback(self._opts,self,request_id,segment_id)
        synthesizer = SpeechSynthesizer(
            model=self._opts.model,
            voice=self._opts.voice,
            format=self._opts.format,
            callback=callback,
            volume=self._opts.volume,
            speech_rate=self._opts.rate,
            pitch_rate=self._opts.pitch
        )

        async def text_input_task():
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()


        async def sentence_stream_task():
            try:
                async for ev in self._sent_tokenizer_stream:
                    logger.info(f"\033[32m[[tts.sentence_stream_task]] detected: {ev.token}\033[0m")
                    synthesizer.streaming_call(ev.token)
                # 使用异步方式处理完成事件
                await asyncio.get_event_loop().run_in_executor(None, synthesizer.streaming_complete)
            finally:
                self._complete_event.set()  # 设置完成事件
            
        async def audio_output_task():
            try:
                async for syn_audio in self._callback_queue:
                    logger.info("\033[32m [[tts.stream]] 开始输出音频数据\033[0m")
                    self._event_ch.send_nowait(syn_audio)
                # 等待所有音频数据处理完成
                await self._complete_event.wait()
            finally:
                self._callback_queue.close()
        # 创建三个独立的任务
        input_task = asyncio.create_task(text_input_task())
        sentence_task = asyncio.create_task(sentence_stream_task()) 
        output_task = asyncio.create_task(audio_output_task())

        try:
            # 等待两个任务都完成
            await input_task
            await sentence_task
            await output_task
        except Exception as e:
            logger.error(f"TTS任务执行出错: {e}")
            raise
        finally:
            # 确保清理任务
            await utils.aio.gracefully_cancel(text_input_task, sentence_task,output_task)
                
    class Callback(ResultCallback):
        def __init__(self, opts: _TTSOptions,stream,request_id,segment_id):
            self._opts = opts
            self.request_id = request_id
            self.segment_id = segment_id
            self.audio_byte_stream = utils.audio.AudioByteStream(
                sample_rate=self._opts.format.sample_rate,
                num_channels=DASHSCOPE_TTS_CHANNELS,
            )
            self.stream=stream

        # def on_open(self) -> None:
        #     logger.info("Synthesis started")

        def on_complete(self) -> None:
            # 输出剩余数据
            for frame in self.audio_byte_stream.flush():
                self.stream._callback_queue.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=self.request_id,
                        segment_id=self.segment_id,
                        frame=frame,
                    )
                )
            logger.info("\033[32m [[tts.stream]] 音频数据已输出完成\033[0m")

        # def on_error(self, message) -> None:
        #     logger.error(f"Synthesis error: {message}")

        def on_close(self) -> None:
            logger.info("Synthesis closed")

        # def on_event(self, message: str) -> None:
        #     logger.debug(f"Synthesis event: {message}")

        def on_data(self, data: bytes) -> None:
            for frame in self.audio_byte_stream.write(data):
                self.stream._callback_queue.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=self.request_id,
                        segment_id=self.segment_id,
                        frame=frame,
                    )
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
        self.audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.format.sample_rate,
            num_channels=DASHSCOPE_TTS_CHANNELS,
        )
        # 创建合成器实例
        synthesizer = SpeechSynthesizer(
            model=self._opts.model,
            voice=self._opts.voice,
            format=self._opts.format,
            volume=self._opts.volume,
            speech_rate=self._opts.rate,
            pitch_rate=self._opts.pitch
        )
        # 调用合成方法并等待完成
        audio = synthesizer.call(self._text)
        for frame in self.audio_bstream.write(audio):
            self._event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id=request_id,
                    segment_id=segment_id,
                    frame=frame,
                )
            )
