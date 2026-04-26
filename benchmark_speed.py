import time
import torch
import argparse
from transformers import BertModel

def benchmark(model, inputs, num_runs=1000, device="cpu"):
    model.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(**inputs)

    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(**inputs)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs * 1000 # in ms
    return avg_latency

def main(student_model_path):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading teacher model...")
    # NOTE: It is best practice to not compute gradients in the benchmark loop
    teacher = BertModel.from_pretrained("bert-base-uncased")
    
    print(f"Loading student model from {student_model_path}...")
    student = BertModel.from_pretrained(student_model_path)
    
    # Dummy inputs: batch_size=1, sequence_length=128
    inputs = {
        "input_ids": torch.randint(0, 30000, (1, 128)).to(device),
        "attention_mask": torch.ones(1, 128).to(device)
    }
    
    num_runs = 1000
    print(f"Measuring teacher latency ({num_runs} runs)...")
    teacher_latency = benchmark(teacher, inputs, num_runs=num_runs, device=device)
    
    print(f"Measuring student latency ({num_runs} runs)...")
    student_latency = benchmark(student, inputs, num_runs=num_runs, device=device)
    
    speedup = teacher_latency / student_latency if student_latency > 0 else float("inf")
    reduction_time = (teacher_latency - student_latency) / teacher_latency * 100 if teacher_latency > 0 else 0
    
    print("\n--- Benchmark Results ---")
    print(f"Teacher (12-layer) Latency:    {teacher_latency:.2f} ms")
    print(f"Student (6-layer) Latency:     {student_latency:.2f} ms")
    print(f"Speedup:                       {speedup:.2f}x faster ({reduction_time:.2f}% reduction in latency)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./distilbert-student-pretrained", help="Path to student model")
    args = parser.parse_args()
    main(args.model_path)
