# 서울 관광지 혼잡도 예측 대시보드 설치 및 실행 안내

## 1. Git Clone
먼저, GitHub에서 코드를 클론합니다:

```
git clone https://github.com/사용자명/저장소명.git
cd 저장소명
```

---

## 2. 가상환경 설정 및 라이브러리 설치
Anaconda를 사용하는 경우, `environment.yml` 파일을 사용하여 환경을 생성합니다:

```
conda env create -f environment.yml
conda activate seoul-tourist-dashboard
```

---

## 3. 데이터셋 준비
`dataset_filtered_csv` 폴더에 관광지별로 이름이 지정된 CSV 파일을 넣습니다.
- CSV 파일의 인코딩은 `cp949`입니다.

예시:
```
프로젝트 루트/
│
├── crowd_predict_xgboost.py               # Streamlit 코드가 있는 메인 파일
├── requirements.txt
├── environment.yml
└── dataset_filtered_csv/
    ├── 경복궁_filtered2.csv
    ├── 덕수궁_filtered2.csv
    └── ...
```

---

## 4. Streamlit 실행

환경 설정과 데이터 준비가 완료되면 다음 명령어로 Streamlit 앱을 실행합니다:

```
streamlit run crowd_predict_xgboost.py
```

- 브라우저가 자동으로 열리며 `localhost:8501`에서 앱을 확인할 수 있습니다.

