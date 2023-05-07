from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import XCom
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 5, 7),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='my_dag',
    default_args=default_args,
    description='My DAG',
    schedule=timedelta(hours=1),
)


def preprocess_data(ti, **kwargs):
    data = pd.read_csv('fraud_email_dataset.csv')
    data.dropna(inplace=True)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['content'])
    y = data['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ti.xcom_push(key='X_train', value=X_train)
    ti.xcom_push(key='X_test', value=X_test)
    ti.xcom_push(key='y_train', value=y_train)
    ti.xcom_push(key='y_test', value=y_test)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

def train_model(ti, **kwargs):
    X_train = ti.xcom_pull(key='X_train')
    y_train = ti.xcom_pull(key='y_train')

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

def evaluate_model(ti, **kwargs):
    X_test = ti.xcom_pull(key='X_test')
    y_test = ti.xcom_pull(key='y_test')

    model = joblib.load('model.pkl')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    with open('evaluation.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Confusion Matrix: {conf_matrix}\n')

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

def deploy_model():
    model = joblib.load('model.pkl')

    # Deploy the model using a REST API or save it as a serialized file
    # For example, save the model to an S3 bucket or deploy it using a cloud service like AWS SageMaker or Google AI Platform
    # Add the necessary deployment code here

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)


preprocess_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task
