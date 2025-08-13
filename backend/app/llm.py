import os
import logging
from typing import Generator, Optional

logger = logging.getLogger("chatbot.llm")


class LLMGenerationError(Exception):
    pass


class BaseLLMProvider:
    def generate_sync(
        self,
        user_message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        raise NotImplementedError

    def generate_stream(
        self,
        user_message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        raise NotImplementedError


class MockLLMProvider(BaseLLMProvider):
    def generate_sync(self, user_message: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        return f"You said: {user_message}\n(Note: Mock model. Configure a real model to generate intelligent responses.)"

    def generate_stream(self, user_message: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9):
        reply = self.generate_sync(user_message, max_new_tokens, temperature, top_p)
        for token in reply.split(" "):
            yield token + " "


class HFInferenceLLMProvider(BaseLLMProvider):
    def __init__(self, model: str, token: Optional[str] = None) -> None:
        try:
            from huggingface_hub import InferenceClient
        except Exception as e:
            raise LLMGenerationError("huggingface_hub is required for HF Inference backend. Install it and try again.") from e
        self.client = InferenceClient(model=model, token=token)
        self.model = model

    def _format_prompt(self, user_message: str) -> str:
        system = "You are a helpful assistant. Answer clearly and concisely."
        return f"<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{user_message}\n[/USER]\n[ASSISTANT]\n"

    def generate_sync(self, user_message: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        prompt = self._format_prompt(user_message)
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
            )
            return response
        except Exception as e:
            logger.exception("HF Inference generation failed")
            raise LLMGenerationError(str(e))

    def generate_stream(self, user_message: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9):
        prompt = self._format_prompt(user_message)
        try:
            for event in self.client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                details=False,
                return_full_text=False,
            ):
                yield str(event)
        except Exception as e:
            logger.exception("HF Inference streaming failed")
            raise LLMGenerationError(str(e))


class LocalTransformersLLMProvider(BaseLLMProvider):
    def __init__(self, model: str) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
            import torch  # noqa: F401
        except Exception as e:
            raise LLMGenerationError(
                "transformers and torch are required for local backend. Install them and try again."
            ) from e
        self.model_name = model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)

        # Optional: detect device automatically if available
        self.device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                self.model = self.model.to(self.device)
        except Exception:
            self.device = "cpu"

        from transformers import TextIteratorStreamer
        self.TextIteratorStreamer = TextIteratorStreamer

    def _format_prompt(self, user_message: str) -> str:
        system = "You are a helpful assistant. Answer clearly and concisely."
        return f"<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{user_message}\n[/USER]\n[ASSISTANT]\n"

    def generate_sync(self, user_message: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        from transformers import GenerationConfig
        prompt = self._format_prompt(user_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Return only assistant part after the prompt
            if "[ASSISTANT]" in generated:
                return generated.split("[ASSISTANT]")[-1].strip()
            return generated
        except Exception as e:
            raise LLMGenerationError(str(e))

    def generate_stream(self, user_message: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9):
        import torch
        prompt = self._format_prompt(user_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = self.TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        try:
            # Run generation in a background thread to allow streaming
            import threading
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            for new_text in streamer:
                yield new_text
            thread.join()
        except Exception as e:
            raise LLMGenerationError(str(e))


def get_llm_provider() -> BaseLLMProvider:
    backend = os.getenv("MODEL_BACKEND", "mock").lower()
    if backend not in {"mock", "local", "hf_api"}:
        logger.warning("Unknown MODEL_BACKEND '%s', defaulting to 'mock'", backend)
        backend = "mock"

    if backend == "mock":
        logger.info("Using MockLLMProvider (set MODEL_BACKEND=local or hf_api to enable a real model)")
        return MockLLMProvider()

    if backend == "hf_api":
        model = os.getenv("HF_API_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        token = os.getenv("HF_API_TOKEN")
        logger.info("Using HF Inference backend with model=%s", model)
        return HFInferenceLLMProvider(model=model, token=token)

    # backend == local
    model = os.getenv("MODEL_NAME", "sshleifer/tiny-gpt2")
    logger.info("Using local Transformers backend with model=%s", model)
    return LocalTransformersLLMProvider(model=model)