from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List

from livekit.agents import tokenize
from livekit.agents.tokenize import WordStream, BufferedWordStream

@dataclass
class _TokenizerOptions:
    """中文分词器的配置选项"""
    min_word_len: int  # 最小词长度
    stream_context_len: int  # 流式处理的上下文长度
    punctuations: list[str]  # 标点符号列表
    ignore_punctuation: bool  # 是否忽略标点

class ChineseWordTokenizer(tokenize.WordTokenizer):
    """专门用于中文文本的分词器
    
    主要特点:
    1. 基于字符级别的分词
    2. 处理中英文混合文本
    3. 保留或忽略标点符号
    4. 支持流式处理
    """
    
    def __init__(
        self,
        *,
        min_word_len: int = 1,
        stream_context_len: int = 2,
        ignore_punctuation: bool = True,
    ) -> None:
        super().__init__()
        self._config = _TokenizerOptions(
            min_word_len=min_word_len,
            stream_context_len=stream_context_len,
            punctuations=['。', '，', '、', '：', '；', '！', '？', '（', '）', 
                        '"', '"', ''', ''', '「', '」', '『', '』', '【', '】',
                        '《', '》', '〈', '〉', '…', '—', '～', '·', '〃',
                        '.', ',', '!', '?', ';', ':', '"', "'", '(', ')',
                        '[', ']', '{', '}', '<', '>', '-', '_', '=', '+',
                        '*', '/', '\\', '|', '@', '#', '$', '%', '^', '&'],
            ignore_punctuation=ignore_punctuation
        )

    def _sanitize_options(self) -> _TokenizerOptions:
        """返回配置的副本"""
        return dataclasses.replace(self._config)

    def _is_chinese_char(self, char: str) -> bool:
        """判断是否为中文字符"""
        return '\u4e00' <= char <= '\u9fff'

    def _is_punctuation(self, char: str) -> bool:
        """判断是否为标点符号"""
        return char in self._config.punctuations

    def _split_words(self, text: str) -> list[tuple[str, int, int]]:
        """核心分词逻辑
        
        返回: List[Tuple[词, 起始位置, 结束位置]]
        """
        if not text:
            return []
            
        words = []
        current_word = ''
        start_pos = 0
        
        for i, char in enumerate(text):
            # 跳过标点符号
            if self._config.ignore_punctuation and self._is_punctuation(char):
                if current_word:
                    words.append((current_word, start_pos, i))
                    current_word = ''
                continue
                
            # 处理标点符号
            if not self._config.ignore_punctuation and self._is_punctuation(char):
                if current_word:
                    words.append((current_word, start_pos, i))
                words.append((char, i, i + 1))
                current_word = ''
                start_pos = i + 1
                continue
                
            # 处理中文字符
            if self._is_chinese_char(char):
                if current_word:
                    words.append((current_word, start_pos, i))
                words.append((char, i, i + 1))
                current_word = ''
                start_pos = i + 1
                continue
                
            # 处理其他字符(英文、数字等)
            if char.isspace():
                if current_word:
                    words.append((current_word, start_pos, i))
                    current_word = ''
                    start_pos = i + 1
            else:
                if not current_word:
                    start_pos = i
                current_word += char
                
        # 处理最后一个词
        if current_word:
            words.append((current_word, start_pos, len(text)))
            
        return words

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """将文本分割成词列表"""
        return [word[0] for word in self._split_words(text)]

    def stream(self, *, language: str | None = None) -> WordStream:
        """返回用于流式处理的分词器"""
        config = self._sanitize_options()
        return BufferedWordStream(
            tokenizer=self._split_words,
            min_token_len=config.min_word_len,
            min_ctx_len=config.stream_context_len,
        ) 
