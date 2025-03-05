import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv(r'c:\Users\sanya\Downloads\medical_insurance.csv')
print(df)
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())
X = df.iloc[:, :-1]  
y = df['charges']   
LE = LabelEncoder()
X_encoded = X.apply(lambda col: LE.fit_transform(col) if col.dtype == 'object' else col)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=6)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
err = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {err}')
plt.bar(np.arange(len(y_test)), y_test, label='Actual')
plt.bar(np.arange(len(y_pred)), y_pred, alpha=0.5, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Charges')
plt.title('Charges Distribution')
plt.legend()
plt.show()
def predict_insurance(age, sex, bmi, children, smoker, region):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    input_data_encoded = input_data.apply(lambda col: LE.fit_transform(col) if col.dtype == 'object' else col)
    prediction = model.predict(input_data_encoded)
    return prediction[0]
iface = gr.Interface(
    fn=predict_insurance,
    inputs=[
        gr.Number(label="Age"),
        gr.Text(label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Children"),
        gr.Text(label="Smoker"),
        gr.Text(label="Region")
    ],
    outputs="text",
    title="Insurance cost Prediction with Linear Regression",
    description="Medical insurance cost prediction using Linear Regression"
)
iface.launch(share=True)
