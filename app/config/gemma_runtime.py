from __future__ import annotations

from pathlib import Path

import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler, suppress_stdout_stderr

from app.config.gemma_config import GEMMA_CONFIG, GemmaModelConfig


class Gemma4VisionChatHandler(Llava15ChatHandler):
    DEFAULT_SYSTEM_MESSAGE = None
    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'system' %}"
        "<start_of_turn>user\n{{ message.content }}<end_of_turn>\n"
        "{% endif %}"
        "{% if message.role == 'user' %}"
        "<start_of_turn>user\n"
        "{% if message.content is string %}{{ message.content }}{% endif %}"
        "{% if message.content is iterable %}"
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' and content.image_url is string %}"
        "{{ content.image_url }}\n"
        "{% endif %}"
        "{% if content.type == 'image_url' and content.image_url is mapping %}"
        "{{ content.image_url.url }}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}{{ content.text }}{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "<end_of_turn>\n"
        "{% endif %}"
        "{% if message.role == 'assistant' and message.content is not none %}"
        "<start_of_turn>model\n{{ message.content }}<end_of_turn>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    )

    def __init__(
        self,
        *args,
        mmproj_use_gpu: bool = GEMMA_CONFIG.model.mmproj_use_gpu,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mmproj_use_gpu = mmproj_use_gpu

    def _init_mtmd_context(self, llama_model) -> None:
        if self.mtmd_ctx is not None:
            return

        with suppress_stdout_stderr(disable=self.verbose):
            ctx_params = self._mtmd_cpp.mtmd_context_params_default()
            ctx_params.use_gpu = self.mmproj_use_gpu
            ctx_params.print_timings = self.verbose
            ctx_params.n_threads = llama_model.n_threads
            ctx_params.flash_attn_type = llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED

            self.mtmd_ctx = self._mtmd_cpp.mtmd_init_from_file(
                self.clip_model_path.encode(),
                llama_model.model,
                ctx_params,
            )

            if self.mtmd_ctx is None:
                raise ValueError(f"Failed to load mtmd context from: {self.clip_model_path}")

            if not self._mtmd_cpp.mtmd_support_vision(self.mtmd_ctx):
                raise ValueError("Vision is not supported by this model")


def create_chat_handler(
    mmproj_path: str,
    verbose: bool = True,
    *,
    model_config: GemmaModelConfig = GEMMA_CONFIG.model,
) -> Gemma4VisionChatHandler:
    mmproj_file = Path(mmproj_path)
    if not mmproj_file.exists():
        raise FileNotFoundError(f"mmproj 파일을 찾을 수 없습니다: {mmproj_path}")

    return Gemma4VisionChatHandler(
        clip_model_path=mmproj_path,
        verbose=verbose,
        mmproj_use_gpu=model_config.mmproj_use_gpu,
    )


def create_llm(
    model_path: str,
    chat_handler: Gemma4VisionChatHandler,
    verbose: bool = True,
    *,
    model_config: GemmaModelConfig = GEMMA_CONFIG.model,
) -> Llama:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    return Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=model_config.n_ctx,
        n_gpu_layers=model_config.n_gpu_layers,
        offload_kqv=model_config.offload_kqv,
        main_gpu=model_config.main_gpu,
        verbose=verbose,
    )
