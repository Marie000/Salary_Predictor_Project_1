from pathlib import Path
import os
import predictor_model
import torch

PACKAGE_ROOT = Path(predictor_model)
DATA_PATH = os.path.join(PACKAGE_ROOT, "datasets")
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")
SAVE_MODEL_NAME = "model_0.pt"
SAVE_VOCAB_NAME = "vocab.pt"
TOKENIZER = "basic_english"

TARGET = "salary"
FEATURE = "description"

DROP_FEATURES = [
    "job_id",
    "company_id",
    "closed_time",
    "posting_domain",
    "sponsored",
    "formatted_work_type",
    "formatted_experience_level",
    "applies",
    "application_type",
    "expiry",
    "skills_desc",
    "scraped",
    "original_listed_time",
    "application_url",
    "job_posting_url",
    "views",
    "currency",
    "compensation_type",
    "max_salary",
    "med_salary",
    "min_salary",
    "pay_period",
    "avg_salary",
]

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model
EMBEDDING_DIM = 64
HIDDEN_DIM = 32
NUM_LAYERS = 2
DROPOUT_RATE = 0.5

# training
EPOCHS = 50
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.05

