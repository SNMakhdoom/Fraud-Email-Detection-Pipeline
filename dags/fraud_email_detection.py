from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz, load_npz
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import os

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
    schedule_interval=timedelta(hours=1),
)

dag_folder = os.path.dirname(__file__)
csv_file_path = os.path.join(dag_folder, 'fraud_email_dataset.csv')

def preprocess_data(ti, **kwargs):
    data = pd.read_csv(csv_file_path)
    data.dropna(inplace=True)
    
    print("Column names in the dataset:", data.columns)
    
    # Check if 'Text' and 'Class' columns exist before trying to access them
    if 'Text' in data.columns and 'Class' in data.columns:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data['Text'])
        y = data['Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        save_npz(os.path.join(dag_folder, 'X_train.npz'), X_train)
        save_npz(os.path.join(dag_folder, 'X_test.npz'), X_test)
        
        ti.xcom_push(key='y_train', value=y_train.tolist())  # Convert Series to list
        ti.xcom_push(key='y_test', value=y_test.tolist())  # Convert Series to list
    else:
        print("Columns 'Text' and 'Class' are not found in the dataset!")

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

def train_model(ti, **kwargs):
    dag_folder = os.path.dirname(__file__)
    X_train = load_npz(os.path.join(dag_folder, 'X_train.npz'))
    y_train = ti.xcom_pull(key='y_train', task_ids='preprocess_data')

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(dag_folder, 'model.pkl'))

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

def evaluate_model(ti, **kwargs):
    X_test = load_npz(os.path.join(dag_folder, 'X_test.npz'))
    y_test = ti.xcom_pull(key='y_test', task_ids='preprocess_data')

    model = joblib.load(os.path.join(dag_folder, 'model.pkl'))

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    with open(os.path.join(dag_folder, 'evaluation.txt'), 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Confusion Matrix: {conf_matrix}\n')

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

def deploy_model():
    model = joblib.load(os.path.join(dag_folder, 'model.pkl'))

    # Deploy the model using a REST API or save it as a serialized file
    # For example, save the model to disk
    joblib.dump(model, os.path.join(dag_folder, 'model_deployment.pkl'))

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

preprocess_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task
