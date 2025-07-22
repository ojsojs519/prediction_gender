from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# ⭐️ 이 부분을 수정했습니다!
# 로컬 폴더 대신 Hugging Face Hub의 저장소 이름을 적어줍니다.
# "본인HuggingFace아이디/모델저장소이름" 형식입니다.
MODEL_NAME = "ojs595/gen_predict" 

# 아래 코드는 그대로 유지됩니다.
# from_pretrained 함수가 알아서 Hugging Face Hub에서 모델을 다운로드합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
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