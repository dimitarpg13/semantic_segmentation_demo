from datasets import load_dataset
import json
from huggingface_hub import notebook_login
from huggingface_hub import cached_download, hf_hub_url
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter
import evaluate
import torch
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer


def main():
    notebook_login()
    ds = load_dataset("scene_parse_150", split="train[:50]", trust_remote_code=True)
    ds = ds.train_test_split(test_size=0.2)

    train_ds = ds["train"]
    test_ds = ds["test"]

    repo_id = "huggingface/label-files"
    #filename = "ade20k-id2label.json"
    filename = "maskformer-ade20k-full-id2label.json"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)
    print(num_labels)

    feature_extractor = AutoImageProcessor.from_pretrained("nvidia/mit-b0", reduce_labels=True)

    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    def train_transforms(example_batch):
        images = [jitter(x) for x in example_batch["image"]]
        labels = [x for x in example_batch["annotation"]]
        inputs = feature_extractor(images, labels)
        return inputs

    def val_transforms(example_batch):
        images = [x for x in example_batch["image"]]
        labels = [x for x in example_batch["annotation"]]
        inputs = feature_extractor(images, labels)
        return inputs

    train_ds.set_transform(train_transforms)
    test_ds.set_transform(val_transforms)

    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            logits_tensor = torch.nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = metric.compute(
                predictions=pred_labels,
                references=labels,
                num_labels=num_labels,
                ignore_index=255,
                reduce_labels=False,
            )
            for key, value in metrics.items():
                if type(value) is np.ndarray:
                    metrics[key] = value.tolist()
            return metrics

    pretrained_model_name = "nvidia/mit-b0"
    model = AutoModelForSemanticSegmentation.from_pretrained(
        pretrained_model_name, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="segformer-b0-scene-parse-150",
        learning_rate=6e-5,
        num_train_epochs=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        logging_steps=1,
        eval_accumulation_steps=5,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()

