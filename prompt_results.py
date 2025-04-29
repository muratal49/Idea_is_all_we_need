# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Path to your fine-tuned model
# model_dir = "/scratch/sshriva4/nlp/phi3_finetuned_hAI"  # <--- Change if needed

# # Load fine-tuned model
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir,
#     device_map="auto",
#     torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
#     trust_remote_code=True,
#     use_cache=False
# )

# model.eval()

# # 🧠 Your minimal abstract input
# user_abstract = """Recent advances in generative models have enabled the creation of realistic synthetic medical images, which could be useful for training data augmentation and rare disease analysis."""

# # 🔥 Build the correct prompt
# prompt = f"""Abstract:
# {user_abstract}

# Conclusions:

# Limitations:

# Future Work:

# ### Suggest two research ideas:"""

# # Tokenize
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# # Generate
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=250,
#         do_sample=True,
#         top_p=0.95,
#         temperature=0.7,
#         repetition_penalty=1.1,
#         use_cache=False
#     )

# # Decode
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ✨ Print ONLY generated part after "Suggest two research ideas:"
# print("\n----- GENERATED RESEARCH IDEAS -----")
# if "### Suggest two research ideas:" in generated_text:
#     print(generated_text.split("### Suggest two research ideas:")[1].strip())
# else:
#     print(generated_text.strip())












from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
model_path = "/scratch/sshriva4/nlp/phi3_finetuned_hAI"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    use_cache=False
)

# Prompt
prompt = (
    "Suggest two novel, impactful, and slightly underexplored research ideas in healthcare AI, "
    "each explained clearly in 1-2 sentences.\n\n"
    "Research Ideas:"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=250,        
    temperature=0.6,           
    top_p=0.9,
    use_cache=False,
    do_sample=True
)

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Post-process nicely
if "Research Ideas:" in generated_text:
    generated_text = generated_text.split("Research Ideas:")[-1].strip()

print("\n----- SUGGESTED RESEARCH IDEAS -----")
print(generated_text)




