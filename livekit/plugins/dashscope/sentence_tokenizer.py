from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Callable

from livekit.agents import tokenize
from livekit.agents.tokenize import SentenceStream, BufferedSentenceStream

@dataclass
class _TokenizerOptions:
    """中文分句器的配置选项"""
    min_sentence_len: int  # 最小句子长度
    stream_context_len: int  # 流式处理的上下文长度
    sentence_ends: list[str]  # 句子结束标记
    quote_pairs: dict[str, str]  # 引号对

class ChineseSentenceTokenizer(tokenize.SentenceTokenizer):
    """专门用于中文文本的分句器
    
    主要特点:
    1. 处理常见的中文句末标点符号(。！？…)
    2. 处理带有引号的句子
    3. 处理特殊标点符号(；)
    4. 保留句子中的标点符号
    5. 支持流式处理
    """
    
    def __init__(
        self,
        *,
        min_sentence_len: int = 10,
        stream_context_len: int = 5,
    ) -> None:
        super().__init__()
        self._config = _TokenizerOptions(
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
            sentence_ends=['。', '！', '？', '…', '；', '.', '!', '?', '...', ';'],
            quote_pairs={'"': '"', '"': '"', '「': '」', '『': '』'}
        )

    def _sanitize_options(self) -> _TokenizerOptions:
        """返回配置的副本"""
        return dataclasses.replace(self._config)

    def _split_sentences(self, text: str) -> list[str]:
        """核心分句逻辑"""
        if not text:
            return []
            
        sentences = []
        current_sentence = ''
        quote_stack = []  # 用于追踪引号的栈
        
        for char in text:
            current_sentence += char
            
            # 处理引号
            if char in self._config.quote_pairs.keys():
                quote_stack.append(char)
            elif char in self._config.quote_pairs.values():
                if quote_stack and self._config.quote_pairs[quote_stack[-1]] == char:
                    quote_stack.pop()
                    
            # 当遇到句末标点且不在引号内时,进行分句
            if char in self._config.sentence_ends and not quote_stack:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ''
                
        # 处理最后一个句子
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        return sentences

    def tokenize(self, text: str) -> list[str]:
        """将文本分割成句子列表,并合并过短的句子"""
        raw_sentences = self._split_sentences(text)
        
        # 合并过短的句子
        sentences = []
        buff = ""
        for sentence in raw_sentences:
            buff += sentence + " "
            if len(buff) - 1 >= self._config.min_sentence_len:
                sentences.append(buff.rstrip())
                buff = ""

        if buff:
            sentences.append(buff.rstrip())

        return sentences

    def _is_sentence_complete(self, text: str) -> bool:
        """判断文本是否构成完整的句子"""
        if not text:
            return False
            
        # 检查是否以句末标点结束
        if text[-1] in self._config.sentence_ends:
            # 检查引号是否配对完整
            quote_stack = []
            for char in text:
                if char in self._config.quote_pairs.keys():
                    quote_stack.append(char)
                elif char in self._config.quote_pairs.values():
                    if quote_stack and self._config.quote_pairs[quote_stack[-1]] == char:
                        quote_stack.pop()
            
            # 只有当引号都配对完成时,才认为句子完整
            return len(quote_stack) == 0
            
        return False

    def stream(self) -> SentenceStream:
        """返回用于流式处理的分句器"""
        config = self._sanitize_options()
        return BufferedSentenceStream(
            tokenizer=self._split_sentences,
            min_token_len=config.min_sentence_len,
            min_ctx_len=config.stream_context_len,
        )
