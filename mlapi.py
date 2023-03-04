from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

app = FastAPI()

class Item(BaseModel): 
    gender: int
    university: int
    age: int

with open('model.pkl', 'rb') as f: 
    model = pickle.load(f)

@app.post("/")
async def scoring_endpoint(item:Item):
     
   df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
   yhat = model.predict(df)
   le = preprocessing.LabelEncoder()
   return {"price":str(yhat[0][0]), "pricetype":str(yhat[0][1]), "city":str(yhat[0][2]), "gender":str(yhat[0][3]), "squareft":str(yhat[0][4]), "parking":str(yhat[0][5])}
#    return {"prediction":str(yhat[0][0])}
