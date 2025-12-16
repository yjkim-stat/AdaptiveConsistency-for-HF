import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import traceback
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from transformers import Gemma3ForConditionalGeneration


@dataclass
class HFBundle:
    model: Any
    tokenizer: Any
    processor: Any  # gemma 등에서 사용, llama는 None
    device: torch.device
    meta: Dict[str, Any]  # family/is_multimodal 등


MODEL_REGISTRY: Dict[str, HFBundle] = {}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================
# 1. 모델 로딩 / task_lm 생성
# ==============================

def normalize_multimodal_context(context: list) -> list:
    """
    이러한 형식은 multimodal을 지원하는 모델에서 나오는 패턴
    """
    new_context = []
    for message in context:
        content = message["content"]
        if isinstance(content, str):
            new_content = [{"type": "text", "text": content}]
        else:
            new_content = content  # 이미 list[dict] 형태면 그대로 유지
        new_context.append({
            "role": message["role"],
            "content": new_content
        })
    return new_context

def build_model_and_tokenizer(model_name: str, cache_dir: str | None) -> tuple[Any, Any]:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = os.getenv('CACHE_DIR')

    quantization_config = {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
    }

    tokenizer = None
    processor = None

    if 'llama' in model_name:

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(**quantization_config),
            cache_dir=cache_dir,
            attn_implementation=os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2"),
            token=os.getenv("HF_TOKEN"),
        )

        step_ids = tokenizer.encode("<|eot_id|>", add_special_tokens=False)

        eos_id = tokenizer.encode(
            "<|end_of_text|>", add_special_tokens=False
        )[0]
        pad_id = 128248
        pad_token = tokenizer.decode(pad_id)
        model.generation_config.pad_token_id = pad_id
        model.generation_config.eos_token_id = eos_id
        meta = {"family": "llama", "is_multimodal": False}  # 또는 gemma면 True
    elif 'gemma' in model_name:

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
            quantization_config=BitsAndBytesConfig(**quantization_config),
            cache_dir=cache_dir,
            attn_implementation=os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2"),
            token=os.getenv("HF_TOKEN"),
        )

        eos_id = tokenizer.encode(
            "<|end_of_turn|>", add_special_tokens=False
        )[0]
        pad_id = 0
        pad_token = tokenizer.decode(pad_id)
        model.generation_config.pad_token_id = pad_id
        # model.generation_config.eos_token_id = [eos_id, model.generation_config.eos_token_id]
        model.generation_config.eos_token_id = eos_id

        processor = AutoProcessor.from_pretrained(model_name)
    
        meta = {"family": "gemma", "is_multimodal": True}  # 또는 gemma면 True

    else:
        raise KeyError
    # model.config.pad_token_id = model.config.eos_token_id
    # tokenizer.pad_token = tokenizer.eos_token

    # model.to(DEVICE)
    
    return model, tokenizer, processor, meta


def normalize_messages(messages):
    normalized = []
    for m in messages:
        if isinstance(m, str):
            normalized.append({"role": "user", "content": m})
        elif isinstance(m, dict):
            normalized.append(m)
        else:
            raise TypeError(f"Unsupported message type: {type(m)}")
    return normalized

def get_hf_model(
    model_name: str,
    *,
    cache_dir: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,   # 호환용으로 남겨둠(실제 로드는 build_model_and_tokenizer가 담당)
    device_map: Optional[Union[str, Dict[str, int]]] = "auto",  # 호환용
) -> HFBundle:
    """
    모델 레지스트리에서 가져오고, 없으면 build_model_and_tokenizer()로 로드해서 넣는다.
    - llama / gemma 모두 지원
    - processor까지 함께 반환
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    device = _pick_device()

    # ✅ 여기서 직접 AutoTokenizer/AutoModel... 로드하지 않고,
    #    네가 만든 build_model_and_tokenizer()를 사용
    model, tokenizer, processor, meta = build_model_and_tokenizer(model_name, cache_dir)

    # device_map="auto"로 로드된 경우 보통 이미 배치됨.
    # 다만 build_model_and_tokenizer가 cpu로 로드했거나 device_map을 안 쓴 경우 대비:
    if device.type != "cuda":
        try:
            model.to(device)
        except Exception:
            # device_map="auto"거나 sharded면 .to()가 안 될 수도 있으니 그냥 pass
            pass

    model.eval()

    bundle = HFBundle(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        device=device,
        meta=meta,
    )
    MODEL_REGISTRY[model_name] = bundle
    return bundle



# =========================
# 2) Helpers (stop/logprobs)
# =========================

def _truncate_at_stop(text: str, stop: Optional[Union[str, List[str]]]) -> str:
    if stop is None:
        return text
    stops = [stop] if isinstance(stop, str) else stop
    cut = None
    for s in stops:
        idx = text.find(s)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    return text if cut is None else text[:cut]


def _build_prompt(tokenizer, prompt: str, chat: bool) -> str:
    """
    chat 모델이면 apply_chat_template을 최대한 활용.
    """
    if not chat:
        return prompt

    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
    # fallback
    return prompt


def _extract_logprobs_from_generate(
    tokenizer,
    gen_out,
    input_len: int,
) -> Dict[str, Any]:
    """
    HF generate(output_scores=True, return_dict_in_generate=True)에서
    생성된 토큰들의 token_logprobs를 계산해 OpenAI logprobs처럼 유사 구조로 반환.
    """
    # scores: List[Tensor] where each is (batch, vocab) for each generated step
    scores = gen_out.scores  # length = gen_tokens
    sequences = gen_out.sequences  # (batch, input+gen)

    # 생성 토큰 ids
    gen_token_ids = sequences[0, input_len:]  # (gen_len,)
    token_strs = [tokenizer.decode([tid], skip_special_tokens=False) for tid in gen_token_ids.tolist()]

    token_logprobs: List[float] = []
    for t, step_scores in enumerate(scores):
        # step_scores[0]: (vocab,)
        logp = torch.log_softmax(step_scores[0], dim=-1)
        tid = gen_token_ids[t].item()
        token_logprobs.append(float(logp[tid].detach().cpu()))

    # OpenAI 스타일에 완전 동일하진 않지만, downstream에서 쓰기 좋게 구성
    return {
        "tokens": token_strs,
        "token_logprobs": token_logprobs,
        "text_offset": None,  # 필요하면 나중에 채워도 됨
    }




def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.inference_mode()
def call_hf(
    prompt: str,
    model: str,
    stop: Optional[Union[str, List[str]]] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    majority_at: Optional[int] = None,
    logprobs: int = 0,
    *,
    chat: bool = True,
    cache_dir: Optional[str] = None,
    num_completions_batch_size: int = 5,
    retry: int = 20,
    sleep_cap: int = 60,
    images: Optional[Union[Any, List[Any]]] = None,  # ✅ 추가: gemma 등 멀티모달 입력
) -> Union[List[str], Tuple[List[str], List[Dict[str, Any]]]]:
    """
    OpenAI call_gpt와 비슷한 I/O:
    - return: completions(list[str])
    - logprobs!=0 이면 (completions, all_data) 리턴

    ✅ processor가 있는 모델(Gemma3 등)에서는:
      - images가 주어지면 processor(text=..., images=...)로 입력 생성
      - images가 없으면 tokenizer로 텍스트만 처리
    """
    bundle = get_hf_model(model, cache_dir=cache_dir)
    tok = bundle.tokenizer
    lm = bundle.model
    proc = getattr(bundle, "processor", None)

    num_completions = majority_at if majority_at is not None else 1
    completions: List[str] = []
    all_data: List[Dict[str, Any]] = []

    rendered = _build_prompt(tok, prompt, chat=chat)

    # pad/eos는 model config 우선 사용 (없으면 tokenizer fallback)
    pad_id = getattr(lm.generation_config, "pad_token_id", None)
    if pad_id is None:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = getattr(lm.generation_config, "eos_token_id", None)
    if eos_id is None:
        eos_id = tok.eos_token_id

    # 모델이 분산(device_map="auto")이면 lm.device가 대표 디바이스를 주는 편
    model_device = getattr(lm, "device", bundle.device)

    for i in range(retry):
        try:
            remaining = num_completions - len(completions)
            if remaining <= 0:
                break
            requested = min(num_completions_batch_size, remaining)

            # ============================
            # ✅ 입력 준비: processor 우선
            # ============================
            if proc is not None and images is not None:
                # processor는 보통 input_ids 뿐 아니라 pixel_values 등을 포함
                inputs = proc(
                    text=rendered,
                    images=images,
                    return_tensors="pt",
                )
            else:
                inputs = tok(rendered, return_tensors="pt")

            # input_len 계산 (logprobs/디코딩 offset용)
            input_ids = inputs.get("input_ids", None)
            input_len = int(input_ids.shape[1]) if input_ids is not None else 0

            # device 이동
            inputs = _move_to_device(inputs, model_device)

            do_sample = temperature > 0.0
            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-8) if do_sample else None,
                top_p=top_p if do_sample else None,
                num_return_sequences=requested,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            if logprobs != 0:
                gen_out = lm.generate(
                    **inputs,
                    **gen_kwargs,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                seqs = gen_out.sequences  # (batch, input+gen)
                for b in range(seqs.shape[0]):
                    class _Tmp: ...
                    tmp = _Tmp()
                    tmp.sequences = seqs[b:b+1]
                    tmp.scores = [s[b:b+1] for s in gen_out.scores]

                    text = tok.decode(tmp.sequences[0, input_len:], skip_special_tokens=True)
                    text = _truncate_at_stop(text, stop)
                    completions.append(text)

                    all_data.append(_extract_logprobs_from_generate(tok, tmp, input_len))
            else:
                seqs = lm.generate(**inputs, **gen_kwargs)  # Tensor
                for b in range(seqs.shape[0]):
                    text = tok.decode(seqs[b, input_len:], skip_special_tokens=True)
                    text = _truncate_at_stop(text, stop)
                    completions.append(text)

        except torch.cuda.OutOfMemoryError:
            max_tokens = max(16, max_tokens // 2)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(e, type(e))
            traceback.print_exc()

            t = min((i + 1) ** 2, sleep_cap)
            print("Sleeping", t)
            time.sleep(t)
            continue

    if len(completions) < num_completions:
        raise RuntimeError("Failed to generate enough completions")

    completions = completions[:num_completions]
    if logprobs != 0:
        all_data = all_data[:num_completions]
        return completions, all_data
    return completions

