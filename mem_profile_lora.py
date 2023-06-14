import torch
# from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import ProfilerActivity
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
import torch.autograd.profiler as profiler


torch.distributed.init_process_group("nccl")

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = "cpu"
model_name_or_path = "bigscience/mt0-base"
tokenizer_name_or_path = "bigscience/mt0-base"
checkpoint_name = "financial_sentiment_analysis_lora_v1.pt"
text_column = "sentence"
label_column = "text_label"
max_length = 128
lr = 1e-3
num_epochs = 1
batch_size = 1


def main():
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset["validation"] = dataset["test"]
    del dataset["test"]

    classes = dataset["train"].features["label"].names
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["label"]]},
        batched=True,
        num_proc=2,
    )

    print(dataset["train"][0])

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=2,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    profiling_test_done = False

    model = model.to(device)
    if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
        print("1. model init")
        print(torch.cuda.memory_allocated(device))

    for epoch in range(num_epochs):
        if profiling_test_done:
            break
        model.train()
        total_loss = 0.
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            # CUDA Warm-up for accurate profiling
            if step == 0:
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            else:
                # with torch.autograd.profiler.record_function("forward"):
                # with profiler.profile(with_stack=True, profile_memory=True) as prof:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA
                    ],
                    profile_memory=True,
                    record_shapes=True,
                    use_cuda=torch.cuda.is_available(),
                ) as p:

                    outputs = model(**batch)
                    if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
                        print("2. after forward pass")
                        print(torch.cuda.memory_allocated(device))
                        # print(torch.cuda.memory_stats(device))
                        # print(torch.cuda.memory_snapshot())

                    profiling_test_done = True
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    loss.backward()
                    if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
                        print("3. after backward pass")
                        print(torch.cuda.memory_allocated(device))

                    optimizer.step()
                    lr_scheduler.step()
                    if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
                        print("4. after optimizer pass")
                        print(torch.cuda.memory_allocated(device))

                    optimizer.zero_grad()
                    if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
                        print("5. after zero_grad pass")
                        print(torch.cuda.memory_allocated(device))

                    if profiling_test_done:
                        break

        if not profiling_test_done:
            model.eval()
            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                )

            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

    # print(p.key_averages().table(sort_by="self_cpu_memory_usage"))


if __name__ == "__main__":
    main()



