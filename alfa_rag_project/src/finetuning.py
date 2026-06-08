"""
Fine-tuning module for RAG pipeline.
Supports LoRA and QLoRA for efficient model adaptation.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from config import DATA_DIR, MAX_RESPONSE_CHARS, MAX_SENTENCES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model aliases (same as kaggle_main.py)
# ─────────────────────────────────────────────

FINETUNING_MODELS = {
    "vikhr-1b-finetuned": "lirex111/vikhrllama1B_AlfaBank",  # Fine-tuned Vikhr-1B for Alfa-Bank (RECOMMENDED)
    "vikhr-1b": "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct",  # Base 1B model
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class FinetuningConfig:
    """Configuration for LoRA/QLoRA fine-tuning."""
    
    # Model settings
    model_name: str = "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct"
    
    # LoRA settings
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Alpha scaling
    lora_dropout: float = 0.1
    
    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    
    # QLoRA settings
    use_qlora: bool = True  # 4-bit quantization
    use_lora: bool = True   # LoRA adaptation
    
    # Output settings
    output_dir: Path = DATA_DIR / "finetuned_models"
    dataset_path: Path = DATA_DIR / "finetuning_dataset.json"


# ─────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────

def create_finetuning_dataset(
    questions_path: Path = DATA_DIR / "questions.csv",
    answers_path: Path = DATA_DIR / "sample_submission.csv",
    output_path: Path = DATA_DIR / "finetuning_dataset.json",
) -> int:
    """
    Create a question-answer dataset for fine-tuning.
    
    Merges questions.csv and sample_submission.csv into a JSON format
    suitable for supervised fine-tuning.
    
    Args:
        questions_path: Path to questions.csv (q_id, query)
        answers_path: Path to sample_submission.csv (q_id, answer_new)
        output_path: Path to save the dataset
        
    Returns:
        Number of examples created
    """
    # Load questions
    questions_df = pd.read_csv(questions_path)
    
    # Load answers
    answers_df = pd.read_csv(answers_path)
    
    # Merge on q_id
    merged = questions_df.merge(answers_df, on="q_id", how="inner")
    
    # Create dataset in chat format
    dataset = []
    for _, row in merged.iterrows():
        query = str(row["query"]).strip()
        answer = str(row["answer_new"]).strip()
        
        # Skip empty entries
        if not query or not answer:
            continue
        
        # Format as chat messages
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": """Ты суровый банковский AI-аналитик. Отвечай на вопрос строго на основе предоставленного текста.

ПРАВИЛА:
1. Выдавай ТОЛЬКО факты из контекста. Никаких приветствий и лишних слов.
2. Если в контексте нет прямого ответа, выбери самое релевантное предложение из контекста.
3. Отвечай максимально емко. Объединяй длинные списки в одно-два предложения через запятую.
4. Твой ответ не должен превышать 3 предложений."""
                },
                {"role": "user", "content": f"Вопрос: {query}"},
                {"role": "assistant", "content": answer},
            ]
        }
        dataset.append(example)
    
    # Save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created finetuning dataset: {len(dataset)} examples at {output_path}")
    return len(dataset)


# ─────────────────────────────────────────────
# Model loading with LoRA/QLoRA
# ─────────────────────────────────────────────

def load_model_for_finetuning(
    model_name: str,
    use_qlora: bool = True,
    lora_config: Optional["LoraConfig"] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model with LoRA/QLoRA configuration.
    
    Args:
        model_name: Hugging Face model identifier
        use_qlora: Use 4-bit quantization (QLoRA)
        lora_config: LoRA configuration (created if not provided)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "use_cache": False,  # Required for training
    }
    
    if use_qlora:
        # QLoRA: 4-bit quantization
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
        
        # Prepare model for k-bit training
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model = prepare_model_for_kbit_training(model)
    else:
        # Regular LoRA: full precision
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Apply LoRA
    if lora_config is None:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# ─────────────────────────────────────────────
# Formatting functions
# ─────────────────────────────────────────────

def format_chat_for_training(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
) -> dict:
    """
    Format chat messages into a single string for training.
    
    Args:
        example: Dataset example with "messages" key
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        
    Returns:
        Tokenized example
    """
    messages = example["messages"]
    
    # Try to use chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback to manual formatting
            text = ""
            for msg in messages:
                text += f"{msg['role']}: {msg['content']}\n\n"
    else:
        # Manual formatting
        text = ""
        for msg in messages:
            text += f"{msg['role']}: {msg['content']}\n\n"
    
    # Tokenize with padding to max_length
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].copy(),  # For causal LM, labels = input_ids
    }


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(
    config: FinetuningConfig,
) -> Path:
    """
    Run LoRA/QLoRA fine-tuning.
    
    Args:
        config: Fine-tuning configuration
        
    Returns:
        Path to the saved model
    """
    from datasets import load_dataset
    
    # Create dataset if not exists
    if not config.dataset_path.exists():
        create_finetuning_dataset(output_path=config.dataset_path)
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(config.dataset_path), split="train")
    
    # Load model
    model, tokenizer = load_model_for_finetuning(
        config.model_name,
        use_qlora=config.use_qlora,
    )
    
    # Format dataset
    formatted_dataset = dataset.map(
        lambda x: format_chat_for_training(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=10,  # Fixed: use warmup_steps instead of deprecated warmup_ratio
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="none",
        fp16=not config.use_qlora,  # Use fp16 if not QLoRA
        bf16=config.use_qlora,  # Use bf16 for QLoRA
    )
    
    # Data collator - use padding_side="right" which is already set
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    
    logger.info(f"Model saved to {config.output_dir}")
    return config.output_dir


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA/QLoRA")
    
    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Create finetuning dataset from questions.csv and sample_submission.csv",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run fine-tuning",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vikhr-1b",
        choices=list(FINETUNING_MODELS.keys()),
        help="Model alias for fine-tuning (vikhr-1b, qwen2.5-7b, etc.)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        default=True,
        help="Use QLoRA (4-bit quantization)",
    )
    parser.add_argument(
        "--no-qlora",
        action="store_true",
        help="Disable QLoRA, use regular LoRA",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "finetuned_models",
        help="Output directory for the model",
    )
    
    args = parser.parse_args()
    
    if args.create_dataset:
        create_finetuning_dataset()
    
    if args.train:
        # Resolve model alias to full HuggingFace ID
        hf_model_name = FINETUNING_MODELS.get(args.model, args.model)
        
        config = FinetuningConfig(
            model_name=hf_model_name,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            use_qlora=not args.no_qlora,
            output_dir=args.output_dir,
        )
        train(config)