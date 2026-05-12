# DistilBERT Re-implementation

## 1. Introduction
This repository contains the code and resources for a final project re-implementing **DistilBERT**, a distilled version of the BERT language model. The goal of this project is to explore knowledge distillation in Natural Language Processing, demonstrating how a smaller, faster student model can retain the majority of a massive teacher model's performance.

**Original Paper:** *Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.*

**Main Contribution:** DistilBERT applies knowledge distillation during the pre-training phase, reducing the parameter count of BERT by 40% and increasing speed by 60% while retaining 97% of its language understanding capabilities.

## 2. Chosen Result
Our re-implementation specifically targets the core results from the paper regarding **Model Size, Inference Latency, and Downstream Task Accuracy**.
* **Significance:** In the era of massive LLMs, efficient deployment on edge devices requires rigorous compression techniques without devastating accuracy loss.
* **Reference:** We target the results corresponding to *Table 1 (Retained performance on GLUE)* and the *Model Architecture* sections of the original paper, specifically comparing our fine-tuned student against the `bert-base-uncased` teacher on the IMDb sentiment dataset.

## 3. GitHub Contents
The repository is structured according to the CS 4782 final deliverables guidelines:
- `code/`: Contains all PyTorch/Hugging Face Python scripts and Jupyter notebooks.
- `data/`: Directory for caching datasets (with instructions for retrieval).
- `poster/`: Contains the PDF of our academic poster presentation.
- `results/`: Contains generated figures, tables, and inference benchmarking logs.
- `report/`: Contains the final 2-page project summary report.

## 4. Re-implementation Details
**Approach:** We re-implemented the distillation pipeline using PyTorch and the Hugging Face Transformers library.
- **Model Architecture:** We initialized a 6-layer student model following the paper by extracting every alternate layer (0, 2, 4, 6, 8, 10) from the 12-layer `bert-base-uncased` teacher model. We also created an even smaller 3-layer student model using layers 0, 4, and 8 from the teacher to further explore the limits of knowledge distillation.
- **Datasets:** We used the IMDb dataset for downstream evaluation. For pre-training, due to computational constraints, we trained on a scaled-down 10% subset of the BookCorpus dataset instead of the full Wikipedia + BookCorpus pipeline.
- **Training Objective:** We implemented a custom Trainer utilizing the paper's specified Triple Loss: Masked Language Modeling Loss + Soft-label Cross-Entropy (KL Divergence) + Cosine Embedding Loss.
- **Modifications:** We introduced an ablation flag to independently isolate and verify the effects of the distillation loss term.

## 5. Reproduction Steps
**How someone can use this GitHub to re-implement our work:**

1. **Environment Setup:** Ensure you have access to a CUDA-enabled GPU (e.g., Google Colab with an A100/V100).
2. **Install Dependencies:**
   ```bash
   pip install torch transformers datasets evaluate accelerate numpy
   ```
3. **Data Preparation & Student Initialization:**
   ```bash
   cd code
   python init_student.py       # Initializes the 6-layer student and verifies param reduction
   python init_student_3_layers.py       # Initializes the 3-layer student and verifies param reduction
   python prepare_data.py       # Automatically fetches and tokenizes IMDb and the BookCorpus subset
   ```
4. **Pre-training (Requires ~20 GPU Hours Total):**
   ```bash
   python pretrain_student.py   # Runs the full distillation training loop for the 6-layer student
   python pretrain_student_3.py   # Runs the full distillation training loop for the 3-layer student
   ```
   *(For the ablation study without distillation loss, run `python pretrain_student.py --ablation` and `python pretrain_student_3.py --ablation` respectively)*
5. **Downstream Evaluation & Inference Benchmarking:**
   ```bash
   python finetune_imdb.py --model_path ./distilbert-student-pretrained
   python finetune_imdb.py --model_path ./distilbert-student-pretrained-3-layer
   python benchmark_speed.py --model_path ./distilbert-student-pretrained
   python benchmark_speed.py --model_path ./distilbert-student-pretrained-3-layer
   ```

## 6. Results/Insights
Our re-implementation successfully reproduced the central claims of the DistilBERT paper:
- **Parameter Reduction:** We verified a ~40% reduction in parameters (from ~110M to ~66M) for the 6-layer student model, and a ~60% reduction for the 3-layer student model (from ~110M to ~45M).
- **Inference Speed:** Our benchmarks confirmed that the 6-layer student model executes forward passes ~44% faster than the 12-layer teacher. The 3-layer student model executes forward passes ~70% faster than the 12-layer teacher.
- **Downstream Task:** Despite training on only 10% of the original BookCorpus, the distilled model successfully fine-tuned on the IMDb dataset, maintaining a highly competitive accuracy relative to the teacher model.

## 7. Conclusion
This project validates that knowledge distillation is an exceptionally powerful technique for compressing deep Transformer networks. The Triple Loss function successfully transfers the generalizations learned by the teacher into the student, proving that a model's architectural depth can be halved with minimal penalty to its classification accuracy.

## 8. References
1. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
2. Hugging Face Documentation: [Transformers](https://huggingface.co/docs/transformers/index) and [Datasets](https://huggingface.co/docs/datasets/index)
3. Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis.* Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics.

## 9. Acknowledgements
This project was completed as part of the CS 4782 coursework. We thank the teaching staff for their guidance and for providing the framework and compute resources necessary to explore large-scale model distillation.
