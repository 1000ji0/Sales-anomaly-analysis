# 이상치는 오류인가, 전략인가
## 글로벌 완구 판매 데이터의 이상 거래 패턴 분석 (2003–2005)

> **핵심 메시지:** 이상치는 제거할 오류가 아니라 관리해야 할 VIP 신호다.  
> 규칙 기반 탐지를 넘어 다변량 모델로 고도화할 때 비로소 숨겨진 패턴이 드러난다.

---

## 📌 프로젝트 개요

매출 데이터에서 통계적으로 이상치로 분류된 거래를 무조건 제거하면, 고액 VIP 거래가 모델에서 누락되는 정보 손실이 발생한다. 이 프로젝트는 이상치를 단순 제거하지 않고 비즈니스 맥락으로 재분류하는 의사결정 프레임을 구축하고, 이상 거래 자동 탐지 체계를 단계적으로 고도화한다.

**분석 도구:** Python 3.12 · pandas · seaborn · statsmodels · scikit-learn · imbalanced-learn · SHAP  
**데이터:** [Kaggle - Sample Sales Data](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)

---

## 📁 프로젝트 구조

```
sales-anomaly-analysis/
├── data/
│   ├── sales_data_sample.csv     # 원본 데이터
│   └── df_final.csv              # 모델링용 전처리 데이터
├── src/
│   ├── eda_preprocessing.ipynb   # EDA · 전처리 · 피처 엔지니어링
│   └── modeling.ipynb            # Isolation Forest · Random Forest · 비교
├── figures/                      # 시각화 결과물
└── README.md
```

---

## 📊 데이터 개요

| 항목 | 값 |
|------|-----|
| 총 레코드 수 | 2,823건 |
| 분석 기간 | 2003-01-06 ~ 2005-05-31 |
| 고유 고객사 | 92개 |
| 고유 국가 | 19개 |
| 제품군 | 7개 (Classic Cars, Vintage Cars, Motorcycles 등) |
| 주문 상태 | 6종 (Shipped / Cancelled / Disputed / On Hold 등) |

한 행(row)이 하나의 주문 라인을 의미하며, 어떤 고객사가 어떤 제품을 얼마에 몇 개 주문했는지, 그리고 그 주문이 최종적으로 어떻게 처리됐는지를 담고 있다.

---

## 🔬 분석 가설

| 가설 | 예상 | 검증 방법 |
|------|------|---------|
| H1. 이상치는 데이터 오류가 아니다 | 이상치 거래의 대부분은 정상 처리(Shipped)된 고액 거래일 것이다 | STATUS 분포 확인, 고객 비교 분석 |
| H2. 이상치 고객은 VIP다 | 이상치를 발생시킨 고객은 일반 고객보다 주문 빈도·매출 기여가 높을 것이다 | 고객 행동 비교 (주문 횟수, 총 매출) |
| H3. IQR은 일부 이상치를 놓친다 | 매출 금액 외 다른 변수 조합으로도 이상치가 존재할 것이다 | Isolation Forest 다변량 탐지 결과와 비교 |

---

## 🔄 분석 프레임워크

```
[1단계] 데이터 이해        EDA · 상관관계 · VIF
           ↓
[2단계] 이상치 탐지        국가별 IQR → 매출 기준 75건 식별
           ↓
[3단계] 이상치 해석        제품군 · 고객 · 리스크 거래 분류 (H1, H2 검증)
           ↓
[4단계] 파생변수 설계      PRICE_RATIO · BULK_ORDER_RATIO · REVENUE_CONTRIBUTION
           ↓
[5단계] 모델링             Isolation Forest (비지도) · Random Forest (지도) (H3 검증)
```

---

## 📈 EDA 핵심 발견

| 분석 축 | 발견 | 의미 |
|--------|------|------|
| 매출 추이 | 2004년 최고 매출, 2005년은 5월까지만 집계 | 연도 간 단순 비교 불가, 맥락 고려 필요 |
| 제품군 | Classic Cars 967건·매출 압도적 1위, Ships·Planes는 단가 분산 큼 | 핵심 매출원 vs 가격 변동성 높은 군 구분 |
| 국가·지역 | USA 1,004건(35.6%) 압도적 1위, EMEA에 이상치 집중 | 지역별 탐지 기준 분리 필요성 확인 |
| 상관관계 | SALES ↔ PRICEEACH 0.66 / MSRP 0.64 / QUANTITYORDERED 0.55 | 매출은 가격에 가장 민감, 시간 변수 무관 |
| 다중공선성 | QUANTITYORDERED VIF 12,734 / BULK_ORDER_RATIO VIF 12,712 | 사실상 동일 정보 — 피처 해석 시 주의 |

---

## ⚙️ 파생변수 설계

| 변수 | 계산식 | 비즈니스 의미 |
|------|-------|------------|
| `PRICE_RATIO` | PRICEEACH / MSRP | 1.0 초과 = 정가 이상 거래, 미만 = 할인 |
| `BULK_ORDER_RATIO` | QUANTITYORDERED / 제품군별 평균 주문량 | 동일 제품군 평균 대비 대량 구매 여부 |
| `REVENUE_CONTRIBUTION` | SALES / 고객별 평균 매출 | 해당 거래의 고객 내 기여 수준 |
| `Customer_Priority` | Total_Revenue > 전체 평균 → High-Priority | 우량 고객 여부 레이블 |
| `PRICE_CATEGORY` | PRICE_RATIO 기준 3구간 | Discounted / Standard / Premium |
| `Is_Outlier` | IQR 결과 boolean 플래그 | 모델링 타깃 변수 |

---

## 🤖 모델링

### Phase 1. Isolation Forest (비지도)

라벨 없이 14개 피처를 동시에 고려하는 다변량 이상치 탐지.  
contamination은 IQR 이상치 비율(75/2823 = 2.66%)을 기준으로 설정.

**IQR vs Isolation Forest 교차 분석 결과**

| 구분 | 건수 | 의미 |
|------|------|------|
| 두 방법 모두 이상치 | 17건 | 고신뢰 이상치 |
| IQR만 이상치 | 58건 | 매출 고액, 다변량으론 정상 → VIP 대량 구매 |
| **IF만 이상치** | **58건** | **매출 정상, 변수 조합 비정상 → IQR이 놓친 패턴** |
| 둘 다 정상 | 2,690건 | 정상 거래 |

**IF가 새로 발견한 58건의 공통점**

- PRICE_RATIO 평균 1.33 → 정가 초과 거래
- REVENUE_CONTRIBUTION 평균 7.33 → 고객 기여도 집중
- 매출 금액만 보면 정상 범위 → IQR이 놓친 이유

**SHAP 핵심 발견:** Isolation Forest는 매출 금액이 아니라 고객 기여도(REVENUE_CONTRIBUTION)와 정가 초과(PRICE_RATIO) 조합을 핵심 이상 신호로 판단한다.

---

### Phase 2. Random Forest + SMOTE (지도)

**클래스 불균형 처리:** 정상 2,748건 vs 이상치 75건 (36:1) → SMOTE 적용  
**파이프라인:** SMOTE를 pipeline으로 묶어 각 CV fold에서 train에만 적용, 데이터 누수 방지  
**검증:** StratifiedKFold (n_splits=5)

**Cross Validation 결과 (5-Fold)**

| 지표 | 점수 |
|------|------|
| Precision | 0.749 ± 0.058 |
| **Recall** | **0.883 ± 0.085** |
| F1 | 0.809 ± 0.056 |

**테스트셋 Confusion Matrix**

|  | 예측 Normal | 예측 Outlier |
|--|------------|-------------|
| **실제 Normal** | 547 (TN) | 3 (FP) |
| **실제 Outlier** | 2 (FN) | 13 (TP) |

실제 이상치 15건 중 13건 탐지, 오탐율 0.5%.

**Feature Importance 상위 5개**

| 순위 | 변수 | 중요도 |
|------|------|--------|
| 1 | SALES | 0.53 |
| 2 | BULK_ORDER_RATIO | 0.13 |
| 3 | QUANTITYORDERED | 0.11 |
| 4 | PRICE_RATIO | 0.09 |
| 5 | PL_Classic Cars | 0.05 |

---

### Phase 3. 두 모델 최종 비교

| 항목 | IQR | Isolation Forest | Random Forest |
|------|-----|-----------------|---------------|
| 학습 방식 | 규칙 기반 | 비지도 | 지도 |
| 라벨 필요 | 불필요 | 불필요 | 필요 |
| 핵심 변수 | SALES 단일 | REVENUE_CONTRIBUTION, PRICE_RATIO | SALES, BULK_ORDER_RATIO |
| 총 탐지 | 75건 | 75건 | 76건 |
| IQR 일치 | — | 17건 (23%) | 73건 (97%) |
| 새로 발견 | — | **58건** | 3건 |
| Recall | — | — | 0.883 |

---

## ✅ 가설 검증 결과

| 가설 | 결과 | 근거 |
|------|------|------|
| H1. 이상치는 오류가 아니다 | ✅ 채택 | 이상치 75건 중 93.3%가 정상 Shipped 처리. 리스크 거래는 2건뿐 |
| H2. 이상치 고객은 VIP다 | ✅ 채택 | 이상치 고객 총 매출 기여 +78.6%, 주문 빈도 +44.3% |
| H3. IQR은 일부 이상치를 놓친다 | ✅ 채택 | Isolation Forest가 IQR 미탐지 58건 추가 발견 |

---

## 💡 비즈니스 제언

| 시점 | 제언 |
|------|------|
| 즉시 | Euro Shopping Channel · UK Collectables 계약·결제 조건 재검토 |
| 단기 | IQR 기반 월별 자동 이상 거래 스크리닝 프로세스 도입 |
| 중기 | Isolation Forest 연동으로 신규 시장 진입 시 이상 패턴 조기 탐지 |
| 장기 | Random Forest 기반 실시간 이상 거래 알림 시스템 구축 |

**모델 선택 기준**

| 상황 | 추천 |
|------|------|
| 라벨 없는 신규 데이터 | Isolation Forest |
| 이상치 기준 확립 후 운영 자동화 | Random Forest |
| 빠른 1차 스크리닝 | IQR |

---

## ⚠️ 분석의 한계 & 확장 가능성

**한계**
- 데이터가 2003~2005년으로 현재 시장에 직접 적용 불가
- `Is_Outlier` 라벨이 IQR 기반 — 지도 학습 품질이 라벨 품질에 종속
- 외부 요인(계절성, 프로모션, 경쟁사 동향) 미반영

**확장 가능성**
- 시계열 분석 추가 시 계절성 이상 거래 탐지 가능
- 고객 세그먼트별 이상치 임계값 개인화 적용 검토
- CRM 연동으로 VIP 고객 조기 식별 자동화
