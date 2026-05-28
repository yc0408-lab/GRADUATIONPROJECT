"""
공급망 리스크 모니터링 대시보드 - 전체 6영역 완성본
=========================================================

실행 방법 (PyCharm):
  1. PyCharm 하단 Terminal 열기 (View > Tool Windows > Terminal)
  2. 라이브러리 설치 (한 번만):
     pip install streamlit pandas plotly numpy
  3. 실행:
     streamlit run app.py
  4. 브라우저가 자동으로 열림 (http://localhost:8501)

실제 데이터 연결 방법:
  아래 "## 데이터 영역" 부분의 변수들을 모델 출력값/CSV 데이터로 교체
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timezone, timedelta

# 한국 시간대 (KST = UTC+9). Streamlit Cloud 서버가 UTC라서 필요
KST = timezone(timedelta(hours=9))

# =====================================
# 페이지 기본 설정
# =====================================
st.set_page_config(
    page_title="공급망 리스크 모니터",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
    .stPlotlyChart { font-family: 'Malgun Gothic', sans-serif !important; }
</style>
""", unsafe_allow_html=True)


# =====================================
# ## 데이터 영역 (← 실제 데이터로 교체)
# =====================================

# --- 모델 예측 결과 ---
signal = "상승"             # "상승" / "안정" / "하락"
phase = "급변기"            # "급변기" / "회복기" / "조정기" / "안정기"
change_rate = 6.2           # 예측 변화율 (%)
confidence = 72.4           # 앙상블 모델 확신도 (%)
current_ccfi = 1247.3       # 현재 CCFI
predicted_ccfi = 1324.6     # 3주 후 예측 CCFI
current_week = "2026년 21주차 (5/18~5/24)"
forecast_week = "2026년 24주차 (6/8~6/14)"

# --- Top 3 기여 변수 (SHAP 상위) ---
drivers = [
    {"name": "환경 강도3 비율", "ratio": 0.78, "multiplier": "2.3배", "direction": "up"},
    {"name": "환경 기사 수",    "ratio": 0.62, "multiplier": "주당 42건", "direction": "up"},
    {"name": "정치 강도3 비율", "ratio": 0.45, "multiplier": "1.6배", "direction": "up"},
]

# --- 자연어 요약 ---
summary_text = ("최근 4주간 **환경 리스크가 평소 대비 2.3배 증가**했으며 "
                "(홍해 항로 변경·동남아 태풍 영향), 미중 관세 협상 결렬로 "
                "**정치 강도3 기사가 누적**되어 운임 상승 압력이 강합니다.")

# --- 학습 데이터 분포 (카테고리별) ---
categories = [
    {"name": "정치",   "ratio": 57.7, "color": "#E24B4A"},
    {"name": "금융",   "ratio": 24.0, "color": "#EF9F27"},
    {"name": "환경",   "ratio": 8.6,  "color": "#0F6E56"},
    {"name": "수급",   "ratio": 4.6,  "color": "#7F77DD"},
    {"name": "비정형", "ratio": 2.8,  "color": "#B4B2A9"},
    {"name": "물류",   "ratio": 2.3,  "color": "#5F5E5A"},
]

# --- 감성 강도 분포 ---
intensities = [
    {"name": "강도0 긍정",   "ratio": 31, "color": "#5A8C2B"},
    {"name": "강도1 중립",   "ratio": 52, "color": "#B4B2A9"},
    {"name": "강도2 강부정", "ratio": 15, "color": "#EF9F27"},
    {"name": "강도3 심각",   "ratio": 2,  "color": "#E24B4A"},
]

# --- 근거 뉴스 (실제 기사 링크 · 2026.5.11~5.24) ---
news_items = [
    {"category": "환경", "date": "05.13", "intensity": 3,
     "title": "글로벌 공급망 위기, 한국 경제 핵심 변수로… 해상 운임 급등·원자재 조달 불안 경고",
     "source": "전국인력신문 · 국제금융센터·KDI 4대 부문 동시 교란",
     "url": "https://www.kjob.news/news/487498"},
    {"category": "금융", "date": "05.14", "intensity": 2,
     "title": "HMM '운임 30% 올랐는데'… SCFI 5월 둘째 주 1954까지 급등",
     "source": "한국신용신문 · 벙커유 50% 폭등에 수익성 악화",
     "url": "https://www.creditnews.kr/news/articleView.html?idxno=2842"},
    {"category": "정치", "date": "05.15", "intensity": 2,
     "title": "2026 미중 정상회담 종합… 반도체 수출통제 별도 트랙 관리, 무역위원회 설립 합의",
     "source": "법률신문 · 우리 기업 시사점 분석",
     "url": "https://www.lawtimes.co.kr/news/articleView.html?idxno=220864"},
    {"category": "환경", "date": "05.22", "intensity": 2,
     "title": "해상운임지수(CCFI·SCFI·BDI) 통합 모니터링 최신 데이터 갱신",
     "source": "INDEXerGO · 발틱·상하이 해운거래소 기준",
     "url": "https://www.indexergo.com/series/?frq=D&codeId=246"},
]

# --- CCFI 시계열 (CSV에서 불러올 부분) ---
np.random.seed(42)
dates_past = pd.date_range(end='2026-05-24', periods=52, freq='W')
values_past = 1000 + np.cumsum(np.random.randn(52) * 10) + np.linspace(0, 247, 52)
dates_future = pd.date_range(start='2026-05-24', periods=4, freq='W')
values_future = [values_past[-1], values_past[-1]+25, values_past[-1]+55, values_past[-1]+77]


# =====================================
# 신호별 스타일 (자동 전환)
# =====================================
SIGNAL_STYLE = {
    "상승": {"bg": "#FCEBEB", "text": "#501313", "sub": "#791F1F", "line": "#E24B4A"},
    "안정": {"bg": "#F1EFE8", "text": "#2C2C2A", "sub": "#444441", "line": "#888780"},
    "하락": {"bg": "#EAF3DE", "text": "#173404", "sub": "#27500A", "line": "#5A8C2B"},
}
style = SIGNAL_STYLE[signal]

PHASE_INFO = {
    "급변기": {"icon": "🔥", "color": "#501313", "note": "뉴스 기여도 높은 시기"},
    "회복기": {"icon": "📈", "color": "#854F0B", "note": "구조적 변화 진행 중"},
    "조정기": {"icon": "⚖️", "color": "#444441", "note": "AR 신호 충분히 작동"},
    "안정기": {"icon": "✓",  "color": "#173404", "note": "안정적인 흐름 유지"},
}
phase_info = PHASE_INFO[phase]


# =====================================
# ① 헤더
# =====================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("### 공급망 리스크 모니터")
    st.caption(f"CCFI 3주 후 예측 · {current_week} 기준")
with col_h2:
    access_time = datetime.now(KST).strftime('%Y.%m.%d %H:%M:%S')
    st.markdown(
        f"<div style='text-align:right; padding-top:18px; color:#5F5E5A; font-size:12px;'>"
        f"🔄 접속 시각 {access_time}</div>",
        unsafe_allow_html=True
    )

st.markdown(
    "<hr style='margin:8px 0 16px 0; border:none; border-top:0.5px solid #E8E6DE;'/>",
    unsafe_allow_html=True
)


# =====================================
# ② 메인 예측 카드
# =====================================
st.markdown(f"""
<div style="background:{style['bg']}; padding:20px 24px; border-radius:12px; margin-bottom:16px;">
    <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:16px;">
        <div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="font-size:12px; color:{style['sub']}; font-weight:500;">
                    3주 후 전망 · {forecast_week}
                </span>
                <span style="font-size:10px; padding:2px 8px; background:{phase_info['color']}; color:#fff; border-radius:10px; font-weight:500;">
                    {phase_info['icon']} {phase}
                </span>
            </div>
            <div style="display:flex; align-items:baseline; gap:10px;">
                <span style="font-size:28px; font-weight:500; color:{style['text']};">{signal}</span>
                <span style="font-size:14px; color:{style['sub']};">예상 {change_rate:+.1f}%</span>
            </div>
            <div style="font-size:12px; color:{style['sub']}; margin-top:4px;">
                앙상블 모델 확신도 {confidence:.1f}% · {phase_info['note']}
            </div>
        </div>
        <div style="display:flex; gap:24px;">
            <div style="text-align:right;">
                <div style="font-size:11px; color:{style['sub']}; margin-bottom:4px;">현재 CCFI</div>
                <div style="font-size:20px; font-weight:500; color:{style['text']};">{current_ccfi:,.1f}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px; color:{style['sub']}; margin-bottom:4px;">3주 후 예측</div>
                <div style="font-size:20px; font-weight:500; color:{style['text']};">{predicted_ccfi:,.1f}</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================
# ③ 학습 데이터 분포 + 감성 강도 분포 (2단)
# =====================================
col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        st.markdown("**학습 데이터 분포** :gray[총 54,379건]")
        fig_donut = go.Figure(data=[go.Pie(
            labels=[c['name'] for c in categories],
            values=[c['ratio'] for c in categories],
            hole=0.55,
            marker=dict(colors=[c['color'] for c in categories], line=dict(color='white', width=2)),
            textinfo='none',
            hovertemplate='%{label}: %{value}%<extra></extra>'
        )])
        fig_donut.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.0, font=dict(size=11)),
            font=dict(family="Malgun Gothic, sans-serif")
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

with col_b:
    with st.container(border=True):
        st.markdown("**감성 강도 분포** :gray[학습 데이터]")
        for it in intensities:
            bar_html = f"""
            <div style="display:flex; align-items:center; gap:8px; margin:6px 0;">
                <span style="width:80px; font-size:11px; color:#5F5E5A;">{it['name']}</span>
                <div style="flex:1; height:7px; background:#F1EFE8; border-radius:3px; overflow:hidden;">
                    <div style="width:{it['ratio']}%; height:100%; background:{it['color']};"></div>
                </div>
                <span style="width:30px; text-align:right; font-size:11px; color:#5F5E5A; font-weight:500;">{it['ratio']}%</span>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px; color:#888780; margin-top:8px; padding-top:8px; border-top:0.5px solid #E8E6DE;'>"
            "ℹ️ 강도3은 전체 2%로 희소 → 발생 시 강한 신호</div>",
            unsafe_allow_html=True
        )


# =====================================
# ④ CCFI 추이 차트
# =====================================
with st.container(border=True):
    st.markdown(
        "**CCFI 추이 및 예측** :gray[최근 12개월] &nbsp;&nbsp;"
        "<a href='https://www.indexergo.com/series/?frq=D&codeId=246' target='_blank' "
        "style='font-size:11px; color:#3C3489; text-decoration:none;'>실시간 운임지수 보기 ↗</a>",
        unsafe_allow_html=True
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_past, y=values_past,
        name='실측', line=dict(color='#5F5E5A', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates_future, y=values_future,
        name='예측', line=dict(color=style['line'], width=2, dash='dash'),
        mode='lines+markers'
    ))
    fig.add_shape(
        type="line",
        x0=dates_past[-1], x1=dates_past[-1],
        y0=0, y1=1, yref="paper",
        line=dict(dash="dot", color="gray", width=1)
    )
    fig.add_annotation(
        x=dates_past[-1], y=1.02, yref="paper",
        text="현재", showarrow=False,
        font=dict(size=10, color="#5F5E5A")
    )
    # 원래 add_vline은 plotly+pandas 호환성 이슈로 add_shape로 대체
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#F1EFE8'),
        font=dict(family="Malgun Gothic, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# =====================================
# ⑤ 신호 근거 (Top 3 변수 + 자연어 요약)
# =====================================
with st.container(border=True):
    col_d1, col_d2 = st.columns([3, 1])
    with col_d1:
        st.markdown(f"**{signal} 신호의 주요 근거**")
    with col_d2:
        st.markdown(
            "<div style='text-align:right; color:#888780; font-size:11px; padding-top:4px;'>"
            "최근 4주 기준</div>", unsafe_allow_html=True
        )

    for d in drivers:
        bar_color = style['line'] if d['ratio'] >= 0.6 else "#EF9F27"
        text_color = style['sub'] if d['ratio'] >= 0.6 else "#854F0B"
        arrow = "↑" if d['direction'] == "up" else "↓"

        driver_html = f"""
        <div style="display:flex; align-items:center; gap:12px; margin:8px 0;">
            <div style="width:140px; font-size:12px; color:#5F5E5A;">{d['name']}</div>
            <div style="flex:1; height:8px; background:#F1EFE8; border-radius:4px; overflow:hidden;">
                <div style="width:{d['ratio']*100}%; height:100%; background:{bar_color};"></div>
            </div>
            <div style="width:120px; text-align:right; font-size:11px; color:#888780;">
                평소 대비 <span style="font-weight:500; color:{text_color};">{d['multiplier']} {arrow}</span>
            </div>
        </div>
        """
        st.markdown(driver_html, unsafe_allow_html=True)

    st.markdown(
        "<hr style='margin:14px 0 12px 0; border:none; border-top:0.5px solid #E8E6DE;'/>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='font-size:12px; color:#5F5E5A; line-height:1.6;'>📌 {summary_text}</div>",
        unsafe_allow_html=True
    )


# =====================================
# ⑥ 근거 뉴스
# =====================================
with st.container(border=True):
    col_n1, col_n2 = st.columns([3, 1])
    with col_n1:
        st.markdown("**근거 뉴스** :gray[강도3 기사 우선 표시]")
    with col_n2:
        st.markdown(
            "<div style='text-align:right; color:#888780; font-size:11px; padding-top:4px;'>"
            "최근 2주 (5/11~5/24)</div>", unsafe_allow_html=True
        )

    CAT_STYLE = {
        "환경":   {"bg": "#E7F4EF", "text": "#04342C"},
        "정치":   {"bg": "#FCEBEB", "text": "#791F1F"},
        "금융":   {"bg": "#FAEEDA", "text": "#854F0B"},
        "수급":   {"bg": "#EEEDFE", "text": "#3C3489"},
        "물류":   {"bg": "#F1EFE8", "text": "#444441"},
        "비정형": {"bg": "#F1EFE8", "text": "#444441"},
    }
    INTENSITY_STYLE = {
        3: {"label": "강도3", "bg": "#501313"},
        2: {"label": "강도2", "bg": "#854F0B"},
    }

    for n in news_items:
        cat = CAT_STYLE.get(n['category'], CAT_STYLE["비정형"])
        ints = INTENSITY_STYLE.get(n['intensity'], INTENSITY_STYLE[2])

        news_html = f"""
        <a href="{n['url']}" target="_blank"
           style="text-decoration:none; display:flex; align-items:center; gap:12px;
                  padding:12px 6px; border-bottom:1px solid #E8E6DE;">
            <div style="display:flex; flex-direction:column; gap:4px; align-items:flex-start; min-width:75px;">
                <span style="font-size:10px; padding:2px 8px; background:{cat['bg']};
                             color:{cat['text']}; border-radius:10px; font-weight:500;">{n['category']}</span>
                <span style="font-size:10px; color:#888780;">{n['date']}</span>
            </div>
            <div style="flex:1;">
                <div style="font-size:13px; color:#1F1F1F; line-height:1.4;">{n['title']}</div>
                <div style="font-size:11px; color:#888780; margin-top:3px;">{n['source']}</div>
            </div>
            <span style="font-size:10px; padding:2px 8px; background:{ints['bg']};
                         color:#fff; border-radius:8px; font-weight:500;">{ints['label']}</span>
        </a>
        """
        st.markdown(news_html, unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center; margin-top:14px;'>"
        "<a href='https://search.naver.com/search.naver?where=news&query=공급망 운임 리스크' "
        "target='_blank' style='font-size:12px; color:#3C3489; text-decoration:none;'>"
        "전체 기사 보기 (최근 4주 28건) →</a></div>",
        unsafe_allow_html=True
    )


# =====================================
# 푸터
# =====================================
st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "시스템종합설계 · 뉴스 기반 공급망 리스크 모니터링 시스템  |  "
    "Walk-Forward 160회 검증 · R² 0.886 · MAPE 5.30% · 방향 정확도 64.5%"
)
