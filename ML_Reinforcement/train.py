# train_model.py
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/heart.csv")
data.head()
data.info()
data.describe()
data.drop_duplicates()
data.isnull().sum()
label_encoder_column = ['Sex', 'ExerciseAngina','RestingECG','ST_Slope','ChestPainType']
label_encoder = LabelEncoder()
for column in label_encoder_column:
    data[column] = label_encoder.fit_transform(data[column])
data.head()

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


rclf=RandomForestClassifier(n_estimators=100, random_state=42)
rclf.fit(X_train,y_train)

y_prediction_RF=rclf.predict(X_test)
testing_data_accuracy=accuracy_score(y_test,y_prediction_RF)
print('Accuracy score of the testing data: ',testing_data_accuracy)

# Save model and scaler
joblib.dump(rclf, 'model/model.pkl')


print("✅ Model and Scaler saved successfully in 'model/' folder!")
