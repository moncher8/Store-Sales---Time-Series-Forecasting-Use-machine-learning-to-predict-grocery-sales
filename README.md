# Store Sales - Time Series Forecasting

머신러닝을 사용한 식료품 매출 예측 프로젝트

## 프로젝트 개요

이 프로젝트는 Kaggle의 [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) 대회를 위한 시계열 예측 모델입니다. XGBoost와 Nested Cross-Validation을 활용하여 식료품 매장의 일일 매출을 예측합니다.

## 주요 특징

- **Nested Time-Series Cross-Validation**: 최근 데이터와 1년 전 계절성 데이터를 모두 고려한 교차 검증
- **Feature Engineering**: 
  - 휴일, 급여일, 프로모션 등 다양한 시간적 특성
  - Fourier 변환을 통한 주기성 캡처
  - 휴일 거리 기반 특성
  - 지진 영향 기간 고려
- **XGBoost 모델**: Optuna를 사용한 하이퍼파라미터 최적화
- **로그 변환**: log1p 변환을 통한 안정적인 학습

## 데이터셋

- `train.csv`: 훈련 데이터 (2013-01-01 ~ 2017-08-15)
- `test.csv`: 테스트 데이터 (2017-08-16 ~ 2017-08-31)
- `stores.csv`: 매장 정보
- `oil.csv`: 유가 데이터
- `holidays_events.csv`: 공휴일 정보
- `transactions.csv`: 거래량 데이터

## 주요 결과

- 최종 RMSE(log1p): **0.40996** (2017-08-01~2017-08-15 검증 구간)
- 최적 파라미터는 Optuna를 통해 자동으로 선택됨

## 사용 방법

1. 필요한 라이브러리 설치:
```bash
pip install pandas numpy xgboost optuna matplotlib seaborn scikit-learn
```

2. Jupyter Notebook 실행:
```bash
jupyter notebook fe-xgm-nested-final.ipynb
```

3. 데이터 경로 설정:
   - Kaggle 환경: `/kaggle/input/store-sales-time-series-forecasting/`
   - 로컬 환경: 데이터 파일 경로를 수정하세요

## 모델 구조

### Feature Engineering
- **시간 특성**: 요일, 월, 연도, 주차, 주말 여부, 급여일, 월말
- **Fourier 특성**: 연간/주간 주기성 (sin/cos 변환)
- **프로모션**: 원본 및 log1p 변환
- **휴일**: 휴일 여부, 휴일-프로모션 상호작용, 휴일까지/이후 거리
- **기타**: 유가, 거래량 프록시, 지진 영향 기간

### Nested CV 전략
- **Outer Folds**: 2개의 15일 검증 구간 (2017-08, 2017-01)
- **Inner CV**: 
  - 최근 3개 rolling window (30일 간격)
  - 1년 전 2개 anchor window (30일 간격)
  - 가중 평균 (최근 0.6, 1년 전 0.4)

## 파일 구조

```
.
├── README.md
└── fe-xgm-nested-final.ipynb  # 메인 분석 노트북
```

## 참고 자료

- [Kaggle Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.
