import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Paths
input_path = "/scratch/sshriva4/nlp/causal_lm_dataset_hAI.jsonl"  # <<--- CHANGE THIS
output_dir = "/scratch/sshriva4/nlp/phi3_finetuned_hAI"

# Load dataset
dataset = load_dataset("json", data_files=input_path, split="train")

# Model and tokenizer
model_name = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True
)

model.gradient_checkpointing_enable() 

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# Training Arguments (adjusted for Transformers 4.51.3)
training_args = TrainingArguments(
    output_dir=output_dir,          # use your output_dir here
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,
    save_total_limit=2,
    prediction_loss_only=True,      # disables evaluation by default
    report_to=[]                    # disables WandB, tensorboard, etc.
)

# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save final model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine-tuning completed. Model saved to {output_dir}")
