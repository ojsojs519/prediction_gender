from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# FastAPI 앱 생성
app = FastAPI()

# 모델과 토크나이저 로드 (저장했던 파일을 같은 폴더에 업로드)
MODEL_DIR = "./my_gender_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cpu") # Render 무료 버전은 CPU 사용
model.to(device)

# 입력 데이터 형식을 정의
class Item(BaseModel):
    text: str

# 예측 API 엔드포인트 생성
@app.post("/predict")
def predict_gender(item: Item):
    model.eval()
    text = item.text
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    prediction = torch.argmax(outputs.logits, dim=1).flatten().item()
    gender = "남자" if prediction == 0 else "여자"
    
    return {"gender": gender}