import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Hugging Face 저장소 이름
MODEL_NAME = "ojs595/gen_predict" 
# Render에 등록한 환경 변수에서 토큰 값을 가져옵니다.
HF_TOKEN = os.getenv("HF_TOKEN")

# ⭐️ from_pretrained 함수에 token 인자를 추가합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN)
device = torch.device("cpu")
model.to(device)

# --- 이하 코드는 이전과 동일 ---

class Item(BaseModel):
    text: str

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
