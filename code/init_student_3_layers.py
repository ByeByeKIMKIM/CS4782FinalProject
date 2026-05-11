import torch
from transformers import BertModel, BertConfig
def initialize_student_3_layer():
    print("Loading teacher model...")
    teacher_model_name = "bert-base-uncased"
    teacher_model = BertModel.from_pretrained(teacher_model_name)
    print("Configuring 3-layer student model architecture...")
    student_config = BertConfig.from_pretrained(
        teacher_model_name,
        num_hidden_layers=3
    )
    student_model = BertModel(student_config)
    print("Transferring weights from teacher to student...")
    student_model.embeddings.load_state_dict(teacher_model.embeddings.state_dict())
    for student_layer_idx in range(student_config.num_hidden_layers):
        teacher_layer_idx = student_layer_idx * 4                                  
        print(f"Mapping Teacher Layer {teacher_layer_idx} -> Student Layer {student_layer_idx}")
        student_model.encoder.layer[student_layer_idx].load_state_dict(
            teacher_model.encoder.layer[teacher_layer_idx].state_dict()
        )
    student_model.pooler.load_state_dict(teacher_model.pooler.state_dict())
    return teacher_model, student_model
def verify_parameters(teacher_model, student_model):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    reduction_percentage = (1 - (student_params / teacher_params)) * 100
    print("\n--- Parameter Verification ---")
    print(f"Teacher Parameters (BERT-base): {teacher_params:,}")
    print(f"Student Parameters (3-layer):   {student_params:,}")
    print(f"Parameter Reduction:            {reduction_percentage:.2f}%\n")
    assert student_params < teacher_params, "Student should have fewer parameters."
    assert 58 <= reduction_percentage <= 62, f"Reduction is {reduction_percentage:.2f}%, expected ~60%."
    print("Verification passed. 3-layer student model is ready for Phase 2.")
if __name__ == "__main__":
    teacher, student = initialize_student_3_layer()
    verify_parameters(teacher, student)
