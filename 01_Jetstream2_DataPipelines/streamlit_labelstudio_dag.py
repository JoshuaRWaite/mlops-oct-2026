import os
from datetime import datetime, timedelta

import docker
from docker.errors import NotFound

from airflow import DAG
from airflow.operators.python import PythonOperator

# Folder that holds the scripts and Dockerfile (on the Jetstream2 VM)
PROJECT_DIR = '/home/exouser/01_Jetstream2_DataPipelines'

# Host subfolders mounted into the containers (same layout as the manual docker commands)
FRAMES_DIR = os.path.join(PROJECT_DIR, 'frames')
LABEL_STUDIO_DATA_DIR = os.path.join(PROJECT_DIR, 'ls-data')

# Ports on the VM. Airflow's UI already uses 8080, so Label Studio is exposed on 8081.
STREAMLIT_HOST_PORT = 8501
LABEL_STUDIO_HOST_PORT = 8081

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 10, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'streamlit_labelstudio_pipeline',
    default_args=default_args,
    description='Streamlit and Label Studio MLOps Pipeline',
    schedule=None,  # trigger manually from the Airflow UI
    catchup=False,
)


def _remove_if_exists(client, name):
    """Remove a leftover container with this name so re-triggering the DAG works."""
    try:
        client.containers.get(name).remove(force=True)
    except NotFound:
        pass


def launch_streamlit():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    client = docker.from_env()
    _remove_if_exists(client, 'streamlit')
    container = client.containers.run(
        'streamlit-app:latest',
        name='streamlit',
        detach=True,
        ports={'8501/tcp': STREAMLIT_HOST_PORT},
        volumes={FRAMES_DIR: {'bind': '/app/frames', 'mode': 'rw'}},
    )
    return container.id


def launch_label_studio():
    # Label Studio runs as uid 1001, so ls-data must be writable by it
    # (sudo chown -R 1001:1001 ls-data, or chmod -R a+rwX ls-data).
    os.makedirs(LABEL_STUDIO_DATA_DIR, exist_ok=True)
    client = docker.from_env()
    _remove_if_exists(client, 'label-studio')
    container = client.containers.run(
        'heartexlabs/label-studio:latest',
        name='label-studio',
        detach=True,
        ports={'8080/tcp': LABEL_STUDIO_HOST_PORT},
        volumes={LABEL_STUDIO_DATA_DIR: {'bind': '/label-studio/data', 'mode': 'rw'}},
        environment={
            'LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED': 'true',
            'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT': '/label-studio/data',
        },
    )
    return container.id


launch_streamlit_task = PythonOperator(
    task_id='launch_streamlit',
    python_callable=launch_streamlit,
    dag=dag,
)

launch_label_studio_task = PythonOperator(
    task_id='launch_label_studio',
    python_callable=launch_label_studio,
    dag=dag,
)

launch_streamlit_task >> launch_label_studio_task
