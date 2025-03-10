## Airflow base setting
처음 airflow를 사용할 때 다음을 실행합니다.

1. airflow 디렉토리로 이동
    ```bash
    cd airflow
    ```
2. Airflow 데이터베이스 초기화
    ```bash
    airflow db init
    ```
3. Airflow 관리자 계정 생성 (example 부분을 변경)
    ```bash
    airflow users create \
        --username example \
        --password 'example' \
        --firstname example \
        --lastname example \
        --role Admin \
        --email example@example.com
    ```
4. airflow.cfg.example 파일을 airflow.cfg로 복사
    ```bash
    cp airflow.cfg.example airflow.cfg
    ```
    - 복사 후 현재 경로가 '/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/airflow'와 다를 경우 dags_folder와 plugins_folder 경로를 수정합니다. ({airflow 폴더 경로}/dags, {airflow 폴더 경로}/plugins로 변경)

## Airflow 실행
1. 환경 변수 실행 (새로운 터미널을 열 때마다 실행)
    ```bash
    export AIRFLOW_HOME=$(pwd)
    ```
2. Airflow webserver 실행
    ```bash
    airflow webserver --port 8080
    ```
3. Airflow scheduler 실행
    ```bash
    airflow scheduler
    ```