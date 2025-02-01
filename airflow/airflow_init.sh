#!/bin/bash

""" Airflow base setting
처음 airflow를 사용할 때 다음을 실행합니다.
1. airflow users create에서 example로 된 부분을 변경합니다.
2. airflow 디렉토리로 이동하여 'bash airflow_init.sh' 명령어를 실행합니다.
"""

export AIRFLOW_HOME=`pwd`
export AIRFLOW__CORE__DEFAULT_TIMEZONE="Asia/Seoul"
export AIRFLOW__WEBSERVER__DEFAULT_UI_TIMEZONE="Asia/Seoul"

airflow db init
airflow users create \
    --username example \
    --password 'example' \
    --firstname example \
    --lastname example \
    --role Admin \
    --email example@example.com \

echo "export AIRFLOW_HOME=$AIRFLOW_HOME" >> ~/.bashrc
source ~/.bashrc