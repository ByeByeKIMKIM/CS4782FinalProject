import torch
from transformers import BertModel, BertConfig

def initialize_student():
    # 1. Load the pre-trained teacher model (bert-base-uncased)
    print("Loading teacher model...")
    teacher_model_name = "bert-base-uncased"
    teacher_model = BertModel.from_pretrained(teacher_model_name)

    # 2. Configure the student model
    # We copy the teacher's configuration but reduce the number of hidden layers to 6
    print("Configuring student model architecture...")
    student_config = BertConfig.from_pretrained(
        teacher_model_name,
        num_hidden_layers=6
    )
    student_model = BertModel(student_config)

    # 3. Initialize student weights from the teacher
    print("Transferring weights from teacher to student...")
    
    # A. Copy embedding layers directly
    student_model.embeddings.load_state_dict(teacher_model.embeddings.state_dict())

    # B. Copy every other transformer encoder layer (Teacher layers 0, 2, 4, 6, 8, 10)
    for student_layer_idx in range(student_config.num_hidden_layers):
        teacher_layer_idx = student_layer_idx * 2
        student_model.encoder.layer[student_layer_idx].load_state_dict(
            teacher_model.encoder.layer[teacher_layer_idx].state_dict()
        )

    # C. Copy the pooler layer
    student_model.pooler.load_state_dict(teacher_model.pooler.state_dict())

    return teacher_model, student_model

def verify_parameters(teacher_model, student_model):
    # 4. Count and verify the parameter reduction
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    reduction_percentage = (1 - (student_params / teacher_params)) * 100

    print("\n--- Parameter Verification ---")
    print(f"Teacher Parameters (BERT-base): {teacher_params:,}")
    print(f"Student Parameters (6-layer):   {student_params:,}")
    print(f"Parameter Reduction:            {reduction_percentage:.2f}%\n")
    
    # Basic assertions to ensure the reduction matches the target
    assert student_params < teacher_params, "Student should have fewer parameters."
    assert 38 <= reduction_percentage <= 42, f"Reduction is {reduction_percentage:.2f}%, expected ~40%."
    print("Verification passed. Student model is ready for Phase 2.")

if __name__ == "__main__":
    teacher, student = initialize_student()
    verify_parameters(teacher, student)
    
    # Save the initialized student model locally for the next phase
    # student.save_pretrained("./distilbert-student-init")