import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

tourist_sites = {
    "경복궁": 400000,
    "덕수궁": 100000,
    "서울대공원": 300000,
    "예술의전당": 50000,
    "종묘": 80000,
    "창경궁": 30000,
    "창덕궁": 30000
}

weekday_map = {
    "월요일": "weekday_0",
    "화요일": "weekday_1",
    "수요일": "weekday_2",
    "목요일": "weekday_3",
    "금요일": "weekday_4",
    "토요일": "weekday_5",
    "일요일": "weekday_6"
}

def load_and_predict(site_name, max_capacity, selected_weekday):
    file_path = f"dataset_filtered_csv/{site_name}_filtered2.csv"
    df = pd.read_csv(file_path, encoding='cp949')

    X = df.drop(columns=[site_name], errors='ignore')
    y = df[site_name]

    X = X.drop(columns=['Date'], errors='ignore')
    X = X.fillna(0)

    bool_cols = [
        'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
        'season_0', 'season_1', 'season_2', 'season_3'
    ]
    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].astype(int)

    weekday_column = weekday_map[selected_weekday]
    target_data = X[X[weekday_column] == 1]

    if target_data.empty:
        return None

    X_train = X[X[weekday_column] == 0]
    y_train = y[X[weekday_column] == 0]
    y_train = y_train.replace([np.inf, -np.inf], np.nan)
    y_train = y_train.fillna(0)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=369,
        learning_rate=0.087,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(target_data)
    predicted_visitors = int(np.mean(y_pred))

    # 혼잡도 계산
    congestion_level = (predicted_visitors / max_capacity) * 100

    return {
        "predicted_visitors": predicted_visitors,
        "congestion_level": congestion_level
    }

st.title("서울 관광지 혼잡도 예측 대시보드")

# 선택란
selected_site = st.selectbox("관광지 선택", list(tourist_sites.keys()))
selected_weekday = st.selectbox("요일 선택", list(weekday_map.keys()))

# 🔴 간격 추가
st.markdown("<br><hr><br>", unsafe_allow_html=True)

# 예측 결과
selected_result = load_and_predict(selected_site, tourist_sites[selected_site], selected_weekday)
if selected_result:
    st.markdown(f"### 예상 방문자수: {selected_result['predicted_visitors']}명")
    st.markdown(f"### 혼잡도: **{selected_result['congestion_level']:.2f}%**")
else:
    st.warning("해당 요일에 대한 데이터가 없습니다.")

# 🔴 간격 추가
st.markdown("<br><hr><br>", unsafe_allow_html=True)

# 혼잡도 랭킹 및 대안 추천지
all_predictions = []
for site, capacity in tourist_sites.items():
    prediction = load_and_predict(site, capacity, selected_weekday)
    if prediction:
        all_predictions.append({
            "site": site,
            "predicted_visitors": prediction['predicted_visitors'],
            "congestion_level": prediction['congestion_level']
        })

# 그래프 시각화 (Plotly)
if all_predictions:
    sites = [x['site'] for x in all_predictions]
    visitors = [x['predicted_visitors'] for x in all_predictions]
    congestion_values = [x['congestion_level'] for x in all_predictions]

    # Plotly Subplots - 왼쪽: 예상 방문자수 / 오른쪽: 혼잡도
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"{selected_weekday} 예상 방문자수", 
            f"{selected_weekday} 혼잡도"
        )
    )

    # 왼쪽 그래프 - 예상 방문자수
    fig.add_trace(
        go.Bar(x=sites, y=visitors, name='방문자수', marker_color='blue'),
        row=1, col=1
    )

    # 오른쪽 그래프 - 혼잡도
    fig.add_trace(
        go.Bar(x=sites, y=congestion_values, name='혼잡도', marker_color='red'),
        row=1, col=2
    )

    fig.update_layout(height=600, showlegend=False, template="plotly_white")
    st.plotly_chart(fig)

    # 🔴 간격 추가
    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # 혼잡도 상태에 따라 색상 결정
    def get_congestion_status(congestion_level):
        if congestion_level < 30:
            return "여유", "green"
        elif congestion_level < 70:
            return "보통", "orange"
        else:
            return "혼잡", "red"

    # 혼잡도 랭킹
    st.markdown("### 혼잡도 랭킹 (낮은 순)")
    congestion_rank = sorted(all_predictions, key=lambda x: x['congestion_level'])
    for idx, pred in enumerate(congestion_rank):
        status, color = get_congestion_status(pred['congestion_level'])
        st.markdown(
            f"""
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <span style='color: {color}; font-size: 1.5em;'>●</span>
                <span style='margin-left: 10px; font-size: 1.2em;'>
                    {idx+1}. <strong>{pred['site']}</strong> - 혼잡도: {pred['congestion_level']:.2f}% ({status})
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 🔴 간격 추가
    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # 대안 추천지
    st.markdown("## 대안 추천 관광지 🔥")
    alternative_site = congestion_rank[0]  # 가장 낮은 곳 1개만 선택
    st.markdown(
        f"""
        <div style='
            padding: 20px;
            border-radius: 10px;
            background-color: #f9fafb;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-size: 2em;
            color: green;
        '>
            🎉 여기는 어때요? <strong style='color: black;'>{alternative_site['site']}</strong>🎉
            <br>
            예상 방문자수: <strong>{alternative_site['predicted_visitors']}명</strong> 
            <br>
            혼잡도: <strong>{alternative_site['congestion_level']:.2f}%</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    
