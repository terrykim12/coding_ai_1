#!/usr/bin/env python3
"""
Qwen3-8B QLoRA SFT 학습 스크립트

사용법:
    python train_sft_qwen3.py [config_file]

예시:
    python train_sft_qwen3.py
    python train_sft_qwen3.py training_config.json
    python train_sft_qwen3.py training/configs/qlora_qwen3_8b.yaml
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 추가: 공용 헬퍼 (키 매핑) ---
def _merge_yaml_into_config(cfg: dict, base: dict) -> dict:
    # base_model
    base["base_model"] = cfg.get("base_model") or cfg.get("base") or base["base_model"]

    # 데이터 경로
    train_file = cfg.get("train_file") or cfg.get("dataset_path")
    val_file   = cfg.get("val_file")
    if train_file: base["dataset_path"] = train_file
    if val_file:   base["val_file"] = val_file  # prepare_dataset에서 처리

    # 길이
    base["max_seq_length"] = cfg.get("max_seq_length") or cfg.get("max_seq_len") or base["max_seq_length"]

    # LoRA
    lora = cfg.get("lora", {})
    base["lora_r"]       = lora.get("r",       base["lora_r"])
    base["lora_alpha"]   = lora.get("alpha",   base["lora_alpha"])
    base["lora_dropout"] = lora.get("dropout", base["lora_dropout"])
    if "target_modules" in lora:
        base["target_modules"] = lora["target_modules"]

    # 정밀도/체크포인트
    prec = cfg.get("precision", {})
    if "bf16" in prec: base["bf16"] = bool(prec["bf16"])
    if "fp16" in prec and prec["fp16"]: base["bf16"] = False
    if "gradient_checkpointing" in cfg:
        base["gradient_checkpointing"] = bool(cfg["gradient_checkpointing"])

    # 옵티마이저
    opt = cfg.get("optimizer", {})
    if "name" in opt: base["optim"] = opt["name"]
    if "lr" in opt: base["learning_rate"] = float(opt["lr"])
    if "warmup_ratio" in opt: base["warmup_ratio"] = float(opt["warmup_ratio"])  # warmup_steps 대신

    # 학습 블록
    tr = cfg.get("training", {})
    if "epochs" in tr: base["num_train_epochs"] = int(tr["epochs"])
    if "batch_size" in tr: base["per_device_train_batch_size"] = int(tr["batch_size"])
    if "grad_accum" in tr: base["gradient_accumulation_steps"] = int(tr["grad_accum"])
    for k_yaml, k_base in [
        ("logging_steps","logging_steps"), ("eval_steps","eval_steps"),
        ("save_steps","save_steps"), ("output_dir","output_dir"),
    ]:
        if k_yaml in tr: base[k_base] = tr[k_yaml]

    # 그 외 편의 옵션
    if "save_total_limit" in cfg: base["save_total_limit"] = int(cfg["save_total_limit"])
    if "dataloader_pin_memory" in cfg: base["dataloader_pin_memory"] = bool(cfg["dataloader_pin_memory"])
    if "dataloader_num_workers" in cfg: base["dataloader_num_workers"] = int(cfg["dataloader_num_workers"])
    if "save_strategy" in cfg: base["save_strategy"] = cfg["save_strategy"]
    if "resume_from_checkpoint" in cfg: base["resume_from_checkpoint"] = cfg["resume_from_checkpoint"]

    return base

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """설정 파일 로딩 (JSON/YAML 모두 지원)"""
    default_config = {
        "base_model": "Qwen/Qwen3-8B",
        "dataset_path": "training/data/train.jsonl",
        "output_dir": "training/qlora-out",
        "validation_split": 0.1,
        "max_seq_length": 2048,
        "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 200, "eval_steps": 200, "save_total_limit": 3,
        "load_in_4bit": True, "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bf16": True, "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit", "lr_scheduler_type": "cosine",
        "remove_unused_columns": False, "dataloader_pin_memory": False,
        "group_by_length": True, "ddp_find_unused_parameters": False,
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        # 확장 필드
        "warmup_ratio": None,
        "dataloader_num_workers": 0,
        "save_strategy": "steps",
        "resume_from_checkpoint": False,
    }

    if config_path and os.path.exists(config_path):
        try:
            if config_path.lower().endswith((".yaml",".yml")):
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    y = yaml.safe_load(f) or {}
                default_config = _merge_yaml_into_config(y, default_config)
                logger.info(f"Loaded YAML config from {config_path}")
            else:
                with open(config_path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded JSON config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    return default_config

def prepare_dataset(dataset_path: str, validation_split: float = 0.1, val_file: Optional[str]=None):
    """데이터셋 준비 (val_file 있으면 그대로 사용)"""
    try:
        from datasets import load_dataset
        if val_file:
            ds_train = load_dataset("json", data_files=dataset_path)["train"]
            ds_val   = load_dataset("json", data_files=val_file)["train"]
            logger.info(f"Dataset loaded: {len(ds_train)} training samples")
            logger.info(f"Validation samples: {len(ds_val)}")
            return ds_train, ds_val

        dataset = load_dataset("json", data_files=dataset_path)
        if validation_split and validation_split > 0:
            dataset = dataset["train"].train_test_split(test_size=validation_split)
            train_dataset = dataset["train"]; eval_dataset = dataset["test"]
        else:
            train_dataset = dataset["train"]; eval_dataset = None

        logger.info(f"Dataset loaded: {len(train_dataset)} training samples")
        if eval_dataset: logger.info(f"Validation samples: {len(eval_dataset)}")
        return train_dataset, eval_dataset
        
    except ImportError:
        logger.error("datasets library not available. Install with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

def format_training_data(examples, tokenizer, max_seq_length: int):
    """학습 데이터 포맷팅"""
    SENTINEL_PRE = "<<<PATCH_JSON>>>"
    SENTINEL_POST = "<<<END>>>"
    
    input_ids = []
    attention_masks = []
    
    for instruction, input_text, output in zip(
        examples["instruction"], 
        examples["input"], 
        examples["output"]
    ):
        # 입력 구성
        if isinstance(output, dict):
            output_text = json.dumps(output, ensure_ascii=False)
        else:
            output_text = str(output)
        
        # 전체 텍스트 구성
        full_text = f"{instruction}\n{input_text}\n{output_text}"
        
        # 토큰화
        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None
        )
        
        input_ids.append(tokens["input_ids"])
        attention_masks.append(tokens["attention_mask"])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": input_ids  # Causal LM에서는 labels = input_ids
    }

def train_model(config: Dict[str, Any]):
    """모델 학습 실행"""
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
            BitsAndBytesConfig, DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from peft.utils import get_peft_model_state_dict
        
        logger.info("Starting model training...")
        
        # 1. 데이터셋 준비
        train_dataset, eval_dataset = prepare_dataset(
            config["dataset_path"], 
            config["validation_split"],
            val_file=config.get("val_file")
        )
        
        # 2. 토크나이저 로딩
        logger.info(f"Loading tokenizer from {config['base_model']}")
        tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"],
            use_fast=True,
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 3. 양자화 설정
        if config["load_in_4bit"] and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
                bnb_4bit_compute_dtype=torch.bfloat16 if config["bf16"] else torch.float16
            )
            logger.info("Using 4-bit quantization")
        else:
            bnb_config = None
            logger.info("Using full precision")
        
        # 4. 베이스 모델 로딩
        logger.info(f"Loading base model: {config['base_model']}")
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config["bf16"] else torch.float16
        )
        
        # 5. LoRA 설정
        logger.info("Setting up LoRA configuration")
        peft_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["target_modules"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 6. 모델 준비
        if config["load_in_4bit"]:
            model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # 7. 데이터 포맷팅
        logger.info("Formatting training data...")
        train_dataset = train_dataset.map(
            lambda x: format_training_data(x, tokenizer, config["max_seq_length"]),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                lambda x: format_training_data(x, tokenizer, config["max_seq_length"]),
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        # 8. 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # 9. 학습 인수 설정
        # warmup_ratio가 None이면 인자를 전달하지 않도록 처리
        _warmup_ratio = config.get("warmup_ratio", None)
        _ta_kwargs = dict(
            output_dir=config["output_dir"],
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            lr_scheduler_type=config["lr_scheduler_type"],
            warmup_steps=config["warmup_steps"],
            # warmup_ratio는 아래에서 조건부로 세팅
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            eval_steps=config["eval_steps"] if eval_dataset is not None else None,
            save_total_limit=config["save_total_limit"],
            save_strategy=config.get("save_strategy", "steps"),
            bf16=config["bf16"],
            gradient_checkpointing=config["gradient_checkpointing"],
            optim=config["optim"],
            remove_unused_columns=config["remove_unused_columns"],
            dataloader_pin_memory=config["dataloader_pin_memory"],
            dataloader_num_workers=config.get("dataloader_num_workers", 0),
            group_by_length=config["group_by_length"],
            ddp_find_unused_parameters=config["ddp_find_unused_parameters"],
            report_to=None,  # wandb 등 비활성화
            run_name="qwen3-qlora-training"
        )
        if _warmup_ratio is not None:
            _ta_kwargs["warmup_ratio"] = float(_warmup_ratio)

        training_args = TrainingArguments(**_ta_kwargs)
        
        # 10. 트레이너 생성 및 학습
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint", False))
        
        # 11. 모델 저장 (안전 모드)
        logger.info("Saving trained model...")
        output_path = os.path.join(config["output_dir"], "final")
        adapter_path = os.path.join(config["output_dir"], "adapter")

        # 전량 저장이 불안하면 아래 한 줄만 유지
        # trainer.save_model(output_path)  # <- 필요시 주석 처리

        # 어댑터(LoRA)만 안전 저장
        try:
            model.save_pretrained(adapter_path, safe_serialization=True)
            tokenizer.save_pretrained(adapter_path)
            logger.info(f"LoRA adapter saved to {adapter_path}")
            logger.info(f"Training completed! (full model save skipped for safety)")
        except Exception as e:
            logger.error(f"Failed to save adapter: {e}")
            # 최후 수단: 어댑터만이라도 저장
            try:
                model.save_pretrained(adapter_path, safe_serialization=True)
                logger.info(f"LoRA adapter saved to {adapter_path} (fallback)")
            except Exception as e2:
                logger.error(f"Failed to save adapter (fallback): {e2}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Required library not available: {e}")
        logger.error("Install required packages: pip install transformers peft accelerate bitsandbytes")
        return False
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-8B QLoRA SFT 학습"
    )
    parser.add_argument(
        "config",
        nargs='?',
        help="설정 파일 경로 (JSON/YAML)"
    )
    
    args = parser.parse_args()
    
    # 설정 로딩
    config = load_config(args.config)
    
    # 출력 디렉토리 생성
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 설정 저장
    config_save_path = os.path.join(config["output_dir"], "training_config.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Configuration saved to {config_save_path}")
    
    # 학습 실행
    success = train_model(config)
    
    if success:
        logger.info("Training completed successfully!")
        print(f"\n🎉 Training completed!")
        print(f"📁 Output directory: {config['output_dir']}")
        print(f"🔧 LoRA adapter: {config['output_dir']}/adapter")
        print(f"📊 Training config: {config_save_path}")
    else:
        logger.error("Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

