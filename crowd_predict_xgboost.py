import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 1. 관광지 목록 및 최대 수용 인원 설정
tourist_sites = {
    "경복궁": 400000,
    "덕수궁": 100000,
    "서울대공원": 300000,
    "예술의전당": 50000,
    "종묘": 80000,
    "창경궁": 30000,
    "창덕궁": 30000
}

# 2. 요일 매핑
weekday_map = {
    "월요일": "weekday_0",
    "화요일": "weekday_1",
    "수요일": "weekday_2",
    "목요일": "weekday_3",
    "금요일": "weekday_4",
    "토요일": "weekday_5",
    "일요일": "weekday_6"
}

# 3. 요일 선택에 따른 데이터 필터링 및 예측 함수
def load_and_predict(site_name, max_capacity, selected_weekday):
    # 데이터 불러오기
    file_path = f"dataset_filtered_csv/{site_name}_filtered2.csv"
    df = pd.read_csv(file_path, encoding='cp949')

    # Feature와 Label 분리
    X = df.drop(columns=[site_name], errors='ignore')  # Feature
    y = df[site_name]                                # Label

    # Date 컬럼 제거
    X = X.drop(columns=['Date'], errors='ignore')

    # NaN 값을 0으로 대체 (모든 칼럼에 적용)
    X = X.fillna(0)

    # True/False → 1/0 변환
    bool_cols = [
        'weekday_0','weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6',
        'season_0','season_1','season_2','season_3'
    ]
    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].astype(int)

    # 선택한 요일에 해당하는 데이터 필터링
    weekday_column = weekday_map[selected_weekday]
    target_data = X[X[weekday_column] == 1]
    target_label = y[X[weekday_column] == 1]

    if target_data.empty:
        return None

    # 학습 데이터
    X_train = X[X[weekday_column] == 0]
    y_train = y[X[weekday_column] == 0]

    # NaN 또는 Inf 값 확인 및 처리
    y_train = y_train.replace([np.inf, -np.inf], np.nan)  # inf, -inf → NaN
    y_train = y_train.fillna(0)  # NaN → 0

    # XGBoost 회귀 모델 설정 및 학습
    model = xgb.XGBRegressor(
        objective="reg:squarederror",  # 회귀
        n_estimators=369,
        learning_rate=0.087,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(target_data)
    predicted_visitors = int(np.mean(y_pred))  # 선택된 요일의 평균 방문자 수

    # 혼잡도 계산 함수
    def get_congestion_level(visitors):
        ratio = visitors / max_capacity * 100  # 백분율
        if ratio < 30:
            return "여유"
        elif ratio < 70:
            return "보통"
        else:
            return "혼잡"

    congestion_level = get_congestion_level(predicted_visitors)

    # 예측 결과 반환
    return {
        "predicted_visitors": predicted_visitors,
        "congestion_level": congestion_level
    }




# 4. Streamlit 웹 대시보드 시작
st.title("서울 관광지 혼잡도 예측 대시보드")
st.markdown("---")

# 5. 요일 선택
st.subheader("무슨 요일에 갈래요?")
selected_weekday = st.selectbox("요일 선택", list(weekday_map.keys()))

# 6. 가장 가고 싶은 관광지 선택
st.subheader("가장 원하시는 관광지가 어디에요?")
selected_site = st.selectbox("관광지 선택", list(tourist_sites.keys()))

# 7. 선택된 관광지의 예측 결과 상단 표시
st.subheader(f"선택한 관광지: {selected_site}")
selected_result = load_and_predict(selected_site, tourist_sites[selected_site], selected_weekday)
if selected_result:
    st.write(f"**예상 방문자수**: {selected_result['predicted_visitors']}명")
    st.write(f"**혼잡도**: {selected_result['congestion_level']}")
else:
    st.write("해당 요일에 대한 데이터가 없습니다.")

st.markdown("---")

# 8. 모든 관광지에 대한 혼잡도 예측 및 시각화
st.subheader(f"{selected_weekday} 요일 관광지 혼잡도 시각화")
all_predictions = []

for site, capacity in tourist_sites.items():
    prediction = load_and_predict(site, capacity, selected_weekday)
    if prediction:
        all_predictions.append({
            "site": site,
            "predicted_visitors": prediction['predicted_visitors'],
            "congestion_level": prediction['congestion_level']
        })

# 막대 그래프 시각화
if all_predictions:
    sites = [x['site'] for x in all_predictions]
    visitors = [x['predicted_visitors'] for x in all_predictions]

    fig, ax = plt.subplots()
    ax.bar(sites, visitors, color='skyblue')
    ax.set_xlabel('관광지')
    ax.set_ylabel('예상 방문자 수')
    ax.set_title(f'{selected_weekday} 요일 혼잡도')
    st.pyplot(fig)

st.markdown("---")

# 9. 혼잡도 랭킹
st.subheader("혼잡도 랭킹")
all_predictions = sorted(all_predictions, key=lambda x: x['predicted_visitors'], reverse=True)
for idx, pred in enumerate(all_predictions):
    st.write(f"{idx+1}. **{pred['site']}** - 방문자수: {pred['predicted_visitors']}명, 혼잡도: {pred['congestion_level']}")

st.markdown("---")

# 10. 대안 추천지
st.subheader("대안 추천지")
least_crowded = sorted(all_predictions, key=lambda x: x['predicted_visitors'])
best_site = least_crowded[0]['site']
best_visitors = least_crowded[0]['predicted_visitors']
best_congestion = least_crowded[0]['congestion_level']

st.write(f"**가장 여유로운 대안 관광지:** {best_site}")
st.write(f"예상 방문자수: {best_visitors}명 → 혼잡도: **{best_congestion}**")
