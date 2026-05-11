import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
    BertForMaskedLM,
    BertConfig,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha_ce=0.5, alpha_mlm=0.2, alpha_cos=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.to(self.args.device)
        self.teacher.eval()
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_mlm = alpha_mlm
        self.alpha_cos = alpha_cos
        self.cosine_loss = nn.CosineEmbeddingLoss()
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_outputs = model(**inputs, output_hidden_states=True)
        student_logits = student_outputs.logits
        student_hidden = student_outputs.hidden_states[-1]                            
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits
            teacher_hidden = teacher_outputs.hidden_states[-1]
        loss_mlm = student_outputs.loss
        loss_ce = F.kl_div(
            input=F.log_softmax(student_logits / self.temperature, dim=-1),
            target=F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        target = torch.ones(student_hidden.size(0) * student_hidden.size(1)).to(self.args.device)
        loss_cos = self.cosine_loss(
            student_hidden.view(-1, student_hidden.size(-1)),
            teacher_hidden.view(-1, teacher_hidden.size(-1)),
            target
        )
        loss = (self.alpha_ce * loss_ce) + (self.alpha_mlm * loss_mlm) + (self.alpha_cos * loss_cos)
        return (loss, student_outputs) if return_outputs else loss
def setup_and_train(is_ablation=False):
    print("Loading tokenizer and datasets...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_from_disk("./data/bookcorpus_10pct")
    print("Initializing Teacher and Student for Masked LM...")
    teacher_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    student_config = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers=6)
    student_model = BertForMaskedLM(student_config)
    student_model.bert.embeddings.load_state_dict(teacher_model.bert.embeddings.state_dict())
    for student_idx in range(6):
        teacher_idx = student_idx * 2
        student_model.bert.encoder.layer[student_idx].load_state_dict(
            teacher_model.bert.encoder.layer[teacher_idx].state_dict()
        )
    student_model.cls.load_state_dict(teacher_model.cls.state_dict())
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    output_dir = "./distilbert-pretraining-ablation" if is_ablation else "./distilbert-pretraining"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        save_steps=10_000,
        logging_steps=100,
        fp16=torch.cuda.is_available(),                                
        prediction_loss_only=True,
    )
    print("Initializing Trainer...")
    alpha_ce = 0.0 if is_ablation else 0.5
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=2.0,
        alpha_ce=alpha_ce,
        alpha_mlm=0.2,
        alpha_cos=0.3,
        model=student_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    print("Starting pre-training...")
    trainer.train()
    print("Saving fully pre-trained student model...")
    save_path = "./distilbert-student-pretrained-ablation" if is_ablation else "./distilbert-student-pretrained"
    trainer.save_model(save_path)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true", help="Run ablation study without distillation loss")
    args = parser.parse_args()
    setup_and_train(is_ablation=args.ablation)
