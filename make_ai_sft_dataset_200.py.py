#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ai_sft_dataset_200.py

Qwen3-8B 코딩 보조 AI용 SFT 데이터셋(200 샘플)을 생성합니다.
- 딥러닝/파인튜닝 주제 위주 (Transformers/PEFT/AMP/DataLoader/DDP 등)
- PATCH 중심(160개) + PLAN(40개)
- 출력 파일:
  - training/data/train.jsonl  (200)
  - training/data/val.jsonl    (60, 앞부분 서브셋)

실행:
  python make_ai_sft_dataset_200.py
"""
from __future__ import annotations
import os, json, random, pathlib, re
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "training" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = OUT_DIR / "train.jsonl"
VAL_PATH   = OUT_DIR / "val.jsonl"

random.seed(42)

# ------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------

def norm(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

def brace(s: str) -> str:
    return s

@dataclass
class Record:
    instruction: str
    input: str
    output: dict

    def as_json(self) -> str:
        return json.dumps({
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }, ensure_ascii=False)

# SFT 포맷 헬퍼

def make_patch_record(intent: str, files: list[tuple[str,str]], edits: list[dict]) -> Record:
    ctx = []
    for path, text in files:
        ctx.append(f"<<<FILE {path}>>>\n{norm(text)}\n<<<END>>>")
    inp = f"[INTENT]\n{intent}\n\n[CONTEXT]\n" + "\n".join(ctx)
    out = {"version": "1", "edits": edits}
    return Record(
        instruction="Fix or improve ML/finetune code. Output PATCH JSON only.",
        input=inp,
        output=out,
    )

def make_plan_record(intent: str, files: list[tuple[str,str]], files_plan: list[dict], notes: str="") -> Record:
    ctx = []
    for path, text in files:
        ctx.append(f"<<<FILE {path}>>>\n{norm(text)}\n<<<END>>>")
    inp = f"[INTENT]\n{intent}\n\n[CONTEXT]\n" + "\n".join(ctx)
    out = {"files": files_plan, "notes": notes}
    return Record(
        instruction="Propose a minimal JSON plan for code changes. Output PLAN JSON only.",
        input=inp,
        output=out,
    )

# 공통 패턴

def regex_whole_file() -> dict:
    return {"type": "regex", "pattern": r"(?s)\\A[\\s\\S]*\\Z"}

def regex_func_block(func: str) -> dict:
    # 함수 블록 전체를 잡는 안전한 정규식(다음 비공백 시작 전까지)
    return {"type": "regex", "pattern": rf"^def {re.escape(func)}\\([^)]*\\):[\\s\\S]*?(?=^\\S)"}

# ------------------------------------------------------------
# 토픽 제너레이터(일부 파라미터화)
# ------------------------------------------------------------

def g_dtype_choice(ix: int):
    path = f"examples/ai{ix}/train_utils.py"
    before = """import torch\n\n# BUG: dtype가 고정되어 혼합정밀에 불리\ndef get_dtype():\n    return torch.float16\n"""
    after = """import torch\n\n\ndef get_dtype():\n    \"\"\"GPU가 bf16 지원 시 bfloat16, 아니면 float16 반환\"\"\"\n    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():\n        return torch.bfloat16\n    return torch.float16\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["torch.bfloat16"]}, "code": after}
    intent = "GPU bf16 지원 시 dtype 선택 개선"
    return make_patch_record(intent, [(path, before)], [edit])


def g_dataloader_main_guard(ix: int):
    path = f"examples/ai{ix}/dataloader_main.py"
    before = """import torch, torch.utils.data as tud\n\nclass Dummy(torch.utils.data.Dataset):\n    def __len__(self): return 8\n    def __getitem__(self, i): return i\n\n# BUG: 모듈 임포트만으로 워커가 스폰되어 Windows에서 프리즈\nloader = tud.DataLoader(Dummy(), batch_size=2, num_workers=4)\nfor b in loader:\n    pass\n"""
    after = """import torch, torch.utils.data as tud\n\nclass Dummy(torch.utils.data.Dataset):\n    def __len__(self): return 8\n    def __getitem__(self, i): return i\n\n\ndef main():\n    loader = tud.DataLoader(Dummy(), batch_size=2, num_workers=4, pin_memory=True, persistent_workers=True)\n    for _ in loader:\n        pass\n\n\nif __name__ == "__main__":\n    main()\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["if __name__ == \"__main__\":"]}, "code": after}
    intent = "Windows에서 DataLoader 프리즈 방지 + pin_memory/persistent_workers"
    return make_patch_record(intent, [(path, before)], [edit])


def g_training_args(ix: int):
    path = f"examples/ai{ix}/training_args.py"
    before = """from transformers import TrainingArguments\n\nargs = TrainingArguments(\n    output_dir=\"out\",\n    per_device_train_batch_size=1,\n    num_train_epochs=1,\n)\n"""
    after = """from transformers import TrainingArguments\n\nargs = TrainingArguments(\n    output_dir=\"out\",\n    per_device_train_batch_size=1,\n    gradient_accumulation_steps=8,\n    num_train_epochs=1,\n    logging_steps=10,\n    eval_strategy=\"steps\",\n    eval_steps=50,\n    save_steps=50,\n    lr_scheduler_type=\"cosine\",\n    warmup_ratio=0.03,\n    fp16=False, bf16=True,\n    gradient_checkpointing=True,\n    max_grad_norm=1.0,\n)\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["gradient_accumulation_steps"]}, "code": after}
    intent = "훈련 안정화: GA steps/로깅/평가/체크포인팅/스케줄러"
    return make_patch_record(intent, [(path, before)], [edit])


def g_lora_targets(ix: int):
    path = f"examples/ai{ix}/lora_cfg.py"
    before = """from peft import LoraConfig\n\nLC = LoraConfig(r=8, target_modules=[\n    \"q_proj\", \"v_proj\"  # BUG: 부족\n])\n"""
    after = """from peft import LoraConfig\n\nLC = LoraConfig(r=8, target_modules=[\n    \"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"\n])\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["k_proj", "o_proj"]}, "code": after}
    intent = "LoRA target_modules 확장"
    return make_patch_record(intent, [(path, before)], [edit])


def g_sdpa_fallback(ix: int):
    path = f"examples/ai{ix}/model_load.py"
    before = """from transformers import AutoModelForCausalLM\nmodel = AutoModelForCausalLM.from_pretrained(\n    \"Qwen/Qwen3-8B\",\n)\n"""
    after = """from transformers import AutoModelForCausalLM\nmodel = AutoModelForCausalLM.from_pretrained(\n    \"Qwen/Qwen3-8B\",\n    attn_implementation=\"sdpa\",\n)\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["attn_implementation"]}, "code": after}
    intent = "flash 미지원 시 sdpa 안전 경로"
    return make_patch_record(intent, [(path, before)], [edit])


def g_pad_token(ix: int):
    path = f"examples/ai{ix}/label_cfg.py"
    before = """from transformers import AutoTokenizer\nfrom dataclasses import dataclass\n\n@dataclass\nclass Cfg:\n    pad_id: int = -100  # BUG\n\n"""
    after = """from transformers import AutoTokenizer\nfrom dataclasses import dataclass\n\n@dataclass\nclass Cfg:\n    pad_id: int | None = None\n\n# tokenizer 로드 후 설정\ntok = AutoTokenizer.from_pretrained(\"Qwen/Qwen3-8B\", use_fast=True)\nCFG = Cfg(pad_id=tok.pad_token_id)\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["CFG = Cfg" ]}, "code": after}
    intent = "pad 토큰 ID 정확히 설정"
    return make_patch_record(intent, [(path, before)], [edit])


def g_seed_utils(ix: int):
    path = f"examples/ai{ix}/seed_utils.py"
    before = """# BUG: 시드 미설정\n"""
    after = """import os, random, numpy as np, torch\n\ndef set_seed(seed: int = 42):\n    random.seed(seed)\n    os.environ[\"PYTHONHASHSEED\"] = str(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(seed)\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["set_seed("]}, "code": after}
    intent = "재현성: 시드 함수 추가"
    return make_patch_record(intent, [(path, before)], [edit])


def g_save_adapter(ix: int):
    path = f"examples/ai{ix}/save_after_train.py"
    before = """from peft import PeftModel\n# ... 학습 후\n# BUG: 저장 누락\n"""
    after  = """from peft import PeftModel\n# ... 학습 후\nmodel.save_pretrained(\"training/qlora-out/export\")\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["save_pretrained"]}, "code": after}
    intent = "학습 후 어댑터 저장 추가"
    return make_patch_record(intent, [(path, before)], [edit])


def g_amp_autocast(ix: int):
    path = f"examples/ai{ix}/train_step.py"
    before = """import torch\n\n# BUG: autocast 미사용\ndef train_step(model, batch):\n    x, y = batch\n    out = model(x)\n    loss = out.loss if hasattr(out, 'loss') else out.mean()\n    loss.backward()\n    return loss.item()\n"""
    after = """import torch\n\n@torch.no_grad()\ndef _bf16_supported():\n    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()\n\n\ndef train_step(model, batch):\n    x, y = batch\n    amp_dtype = torch.bfloat16 if _bf16_supported() else torch.float16\n    scaler = torch.cuda.amp.GradScaler(enabled=True)\n    with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):\n        out = model(x)\n        loss = out.loss if hasattr(out, 'loss') else out.mean()\n    scaler.scale(loss).backward()\n    scaler.step(torch.optim.Adam(model.parameters(), lr=1e-4))\n    scaler.update()\n    model.zero_grad(set_to_none=True)\n    return float(loss.detach().item())\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["autocast"]}, "code": after}
    intent = "AMP 적용 + bf16 우선"
    return make_patch_record(intent, [(path, before)], [edit])


def g_trainer_resume(ix: int):
    path = f"examples/ai{ix}/trainer_resume.py"
    before = """from transformers import Trainer\ntrainer = Trainer(... )\n# BUG: 재시작 미지원\ntrainer.train()\n"""
    after = """from transformers import Trainer\ntrainer = Trainer(... )\ntrainer.train(resume_from_checkpoint=True)\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["resume_from_checkpoint"]}, "code": after}
    intent = "체크포인트 재개 지원"
    return make_patch_record(intent, [(path, before)], [edit])


def g_import_insert(ix: int):
    path = f"examples/ai{ix}/imports.py"
    before = """# BUG: numpy 미사용이지만 추후 사용 예정\n"""
    after_line = "import numpy as np\n"
    edit = {"path": path, "loc": {"type":"anchor","before": r"\A"}, "action":"insert_after", "once": True,
            "pre": {"must_not_contain": ["import numpy as np"]}, "code": after_line}
    intent = "필수 import 추가"
    return make_patch_record(intent, [(path, before)], [edit])


def g_eval_callback(ix: int):
    path = f"examples/ai{ix}/callbacks.py"
    before = """from transformers import TrainerCallback\n\nclass MyCB(TrainerCallback):\n    pass\n"""
    after = """from transformers import TrainerCallback, EarlyStoppingCallback\n\nclass MyCB(TrainerCallback):\n    pass\n\nCALLBACKS = [EarlyStoppingCallback(early_stopping_patience=2)]\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["EarlyStoppingCallback"]}, "code": after}
    intent = "조기 종료 콜백 추가"
    return make_patch_record(intent, [(path, before)], [edit])


def g_ddp_main_guard(ix: int):
    path = f"examples/ai{ix}/ddp_main.py"
    before = """import torch\n\n# BUG: torchrun에서 main 가드 없음\ntrainer()\n"""
    after = """import torch\n\n\ndef main():\n    trainer()\n\n\nif __name__ == "__main__":\n    main()\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["__main__"]}, "code": after}
    intent = "DDP/torchrun 안전 가드"
    return make_patch_record(intent, [(path, before)], [edit])


def g_optimizer_paged(ix: int):
    path = f"examples/ai{ix}/optim.py"
    before = """import torch\nopt = torch.optim.Adam([], lr=3e-4)\n"""
    after = """from transformers import get_scheduler\nfrom bitsandbytes.optim import PagedAdamW8bit\nopt = PagedAdamW8bit([], lr=2e-4)\nsched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=100, num_training_steps=1000)\n"""
    edit = {"path": path, "loc": regex_whole_file(), "action": "replace_range", "once": True,
            "pre": {"must_not_contain": ["PagedAdamW8bit"]}, "code": after}
    intent = "8bit 옵티마이저 + 코사인 스케줄러"
    return make_patch_record(intent, [(path, before)], [edit])

# PLAN 생성기 예시들 --------------------------------------------------

def p_dtype_and_args(ix:int):
    f1 = (f"examples/ai{ix}/train_utils.py", "def get_dtype():\n    return torch.float16\n")
    f2 = (f"examples/ai{ix}/training_args.py", "args = TrainingArguments(output_dir='out', per_device_train_batch_size=1, num_train_epochs=1)\n")
    plan = [
        {"path": f1[0], "reason": "dtype가 고정 float16", "strategy": "regex", "tests": ["bf16 또는 fp16 반환"]},
        {"path": f2[0], "reason": "로깅/평가/GA 설정 누락", "strategy": "regex", "tests": ["eval_strategy=='steps'"]},
    ]
    return make_plan_record("학습 안정화(AMP+훈련 설정)",[f1,f2], plan, "bf16 지원 시 우선")


def p_dataloader_and_ddp(ix:int):
    f1 = (f"examples/ai{ix}/dataloader_main.py", "loader = DataLoader(...); for b in loader: pass\n")
    f2 = (f"examples/ai{ix}/ddp_main.py", "trainer()\n")
    plan = [
        {"path": f1[0], "reason": "Windows freeze 가능", "strategy": "regex", "tests": ["__main__ 가드"]},
        {"path": f2[0], "reason": "DDP 가드 없음", "strategy": "regex", "tests": ["__main__"]},
    ]
    return make_plan_record("입출력/런타임 안전성 개선", [f1,f2], plan, "spawn 보호")

# ------------------------------------------------------------
# 데이터 조립 (200 샘플)
# ------------------------------------------------------------
PATCH_GENERATORS = [
    g_dtype_choice,
    g_dataloader_main_guard,
    g_training_args,
    g_lora_targets,
    g_sdpa_fallback,
    g_pad_token,
    g_seed_utils,
    g_save_adapter,
    g_amp_autocast,
    g_trainer_resume,
    g_import_insert,
    g_eval_callback,
    g_ddp_main_guard,
    g_optimizer_paged,
]
PLAN_GENERATORS = [
    p_dtype_and_args,
    p_dataloader_and_ddp,
]

TOTAL = 200
N_PATCH = 160
N_PLAN  = TOTAL - N_PATCH

records: list[Record] = []
ix = 1

# PATCH
for _ in range(N_PATCH):
    g = random.choice(PATCH_GENERATORS)
    records.append(g(ix))
    ix += 1

# PLAN
for _ in range(N_PLAN):
    g = random.choice(PLAN_GENERATORS)
    records.append(g(ix))
    ix += 1

# 셔플
random.shuffle(records)

# 저장
with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(r.as_json() + "\n")

with open(VAL_PATH, "w", encoding="utf-8") as f:
    for r in records[:60]:
        f.write(r.as_json() + "\n")

print(f"[ok] wrote {len(records)} records → {TRAIN_PATH}")
print(f"[ok] wrote {60} records → {VAL_PATH}")
