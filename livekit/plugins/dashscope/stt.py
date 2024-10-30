from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass
from http import HTTPStatus

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from livekit.agents import stt, utils
from livekit.agents.utils import AudioBuffer, merge_frames

from .log import logger
# https://help.aliyun.com/zh/dashscope/developer-reference/real-time-speech-recognition-api-details

DASHSCOPE_SAMPLE_RATE = 16000

@dataclass
class STTOptions:
    model: str
    format: str
    sample_rate: int
    interim_results: bool
    enable_punctuation: bool
    enable_timestamp: bool
    enable_semantic_sentence: bool
    language: str

class STT(stt.STT):
    def __init__(
        self,
        *,
        model: str = "paraformer-realtime-v2",
        format: str = "wav",
        interim_results: bool = True,
        enable_punctuation: bool = True,
        enable_timestamp: bool = False,
        enable_semantic_sentence: bool = True,
        language: str = "zh",
        api_key: str | None = None,
    ) -> None:
        """
        创建DashScope STT实例
        
        Args:
            model: 使用的模型,默认为paraformer-realtime-v2
            format: 音频格式,默认为wav
            interim_results: 是否返回中间结果
            enable_punctuation: 是否启用标点
            enable_timestamp: 是否启用时间戳
            enable_semantic_sentence: 是否启用语义分句
            language: 语言,默认为zh
            api_key: DashScope API密钥
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False
            )
        )

        # 验证API key
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError("DashScope API key is required")

        dashscope.api_key = api_key

        self._opts = STTOptions(
            model=model,
            format=format,
            sample_rate=DASHSCOPE_SAMPLE_RATE,
            interim_results=interim_results,
            enable_punctuation=enable_punctuation,
            enable_timestamp=enable_timestamp,
            enable_semantic_sentence=enable_semantic_sentence,
            language=language
        )

    async def recognize(self, buffer: AudioBuffer, **kwargs) -> stt.SpeechEvent:
        """实现非流式识别"""
        logger.info(f"\033[32mDashScope STT recognize调用\033[0m")

        # 创建识别器实例并调用API
        recognition = Recognition(
            model=self._opts.model,
            format=self._opts.format, 
            sample_rate=buffer.sample_rate,
            callback=None
        )        
        buffer = merge_frames(buffer)
        # 直接写入临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            with wave.open(temp_file, "wb") as wav:
                wav.setnchannels(buffer.num_channels)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(buffer.sample_rate)
                wav.writeframes(buffer.data)
            temp_file_path = temp_file.name
        result = recognition.call(temp_file_path)
        if result.status_code == HTTPStatus.OK:
            text = "".join(sentence.get("text", "") for sentence in result.get_sentence() or [])
            logger.debug(f"识别结果: {text}")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=text,
                        confidence=1.0,
                        language=self._opts.language,
                    )
                ]
            )
        else:
            logger.error(f"识别请求失败: {result.message}")
            raise Exception(f"识别请求失败: {result.message}")
        
        
    def stream(self) -> "SpeechStream":
        """实现流式识别"""
        logger.info(f"\033[32mDashScope STT stream调用\033[0m")
        config = self._sanitize_options()
        return SpeechStream(config)

class SpeechStream(stt.SpeechStream):
    def __init__(self, opts: STTOptions) -> None:
        super().__init__()
        self._opts = opts
        self._speaking = False
        self._recognition = None

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        class StreamCallback(RecognitionCallback):
            def __init__(self, stream: SpeechStream):
                self.stream = stream
                
            def on_complete(self) -> None:
                if self.stream._speaking:
                    self.stream._speaking = False
                    self.stream._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )
                    
            def on_error(self, result: RecognitionResult) -> None:
                logger.error(f"DashScope STT error: {result}")
                
            def on_event(self, result: RecognitionResult) -> None:
                # 检查是否有文本
                if not result.text:
                    return
                    
                # 处理说话开始
                if not self.stream._speaking:
                    self.stream._speaking = True
                    self.stream._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                
                # 发送识别结果
                event_type = (stt.SpeechEventType.FINAL_TRANSCRIPT 
                            if result.is_final 
                            else stt.SpeechEventType.INTERIM_TRANSCRIPT)
                
                self.stream._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=event_type,
                        alternatives=[
                            stt.SpeechData(
                                text=result.text,
                                confidence=1.0,
                                language="zh"
                            )
                        ]
                    )
                )

        # 创建识别器实例
        self._recognition = Recognition(
            model=self._opts.model,
            format=self._opts.format, 
            sample_rate=self._opts.sample_rate,
            callback=StreamCallback(self)
        )
        
        # 开始识别
        self._recognition.start()
        
        try:
            # 处理输入音频
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    continue
                    
                # 发送音频数据
                self._recognition.send_audio_frame(data.data.tobytes())
                
        finally:
            # 停止识别
            if self._recognition:
                self._recognition.stop()
                self._recognition = None

