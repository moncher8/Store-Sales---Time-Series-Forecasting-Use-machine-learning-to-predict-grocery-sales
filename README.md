[fe-xgm-nested-final.md](https://github.com/user-attachments/files/25199906/fe-xgm-nested-final.md)```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import optuna
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/store-sales-time-series-forecasting/oil.csv
    /kaggle/input/store-sales-time-series-forecasting/sample_submission.csv
    /kaggle/input/store-sales-time-series-forecasting/holidays_events.csv
    /kaggle/input/store-sales-time-series-forecasting/stores.csv
    /kaggle/input/store-sales-time-series-forecasting/train.csv
    /kaggle/input/store-sales-time-series-forecasting/test.csv
    /kaggle/input/store-sales-time-series-forecasting/transactions.csv



```python
train = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv', parse_dates=['date'])
stores = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv')
oil = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv', parse_dates=['date'])
transactions = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/transactions.csv')
```


```python
datasets = {
    'train': train,
    'stores': stores,
    'oil': oil,
    'holidays': holidays,
    'transactions': transactions
}

for name, df in datasets.items():
    print(f"\n===== {name} =====")
    print(df.info())
    #print(df.describe())
    print(df.isnull().sum())
    print("\n")
```

    
    ===== train =====
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3000888 entries, 0 to 3000887
    Data columns (total 6 columns):
     #   Column       Dtype         
    ---  ------       -----         
     0   id           int64         
     1   date         datetime64[ns]
     2   store_nbr    int64         
     3   family       object        
     4   sales        float64       
     5   onpromotion  int64         
    dtypes: datetime64[ns](1), float64(1), int64(3), object(1)
    memory usage: 137.4+ MB
    None
    id             0
    date           0
    store_nbr      0
    family         0
    sales          0
    onpromotion    0
    dtype: int64
    
    
    
    ===== stores =====
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 54 entries, 0 to 53
    Data columns (total 5 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   store_nbr  54 non-null     int64 
     1   city       54 non-null     object
     2   state      54 non-null     object
     3   type       54 non-null     object
     4   cluster    54 non-null     int64 
    dtypes: int64(2), object(3)
    memory usage: 2.2+ KB
    None
    store_nbr    0
    city         0
    state        0
    type         0
    cluster      0
    dtype: int64
    
    
    
    ===== oil =====
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1218 entries, 0 to 1217
    Data columns (total 2 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   date        1218 non-null   datetime64[ns]
     1   dcoilwtico  1175 non-null   float64       
    dtypes: datetime64[ns](1), float64(1)
    memory usage: 19.2 KB
    None
    date           0
    dcoilwtico    43
    dtype: int64
    
    
    
    ===== holidays =====
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 350 entries, 0 to 349
    Data columns (total 6 columns):
     #   Column       Non-Null Count  Dtype         
    ---  ------       --------------  -----         
     0   date         350 non-null    datetime64[ns]
     1   type         350 non-null    object        
     2   locale       350 non-null    object        
     3   locale_name  350 non-null    object        
     4   description  350 non-null    object        
     5   transferred  350 non-null    bool          
    dtypes: bool(1), datetime64[ns](1), object(4)
    memory usage: 14.1+ KB
    None
    date           0
    type           0
    locale         0
    locale_name    0
    description    0
    transferred    0
    dtype: int64
    
    
    
    ===== transactions =====
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 83488 entries, 0 to 83487
    Data columns (total 3 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   date          83488 non-null  object
     1   store_nbr     83488 non-null  int64 
     2   transactions  83488 non-null  int64 
    dtypes: int64(2), object(1)
    memory usage: 1.9+ MB
    None
    date            0
    store_nbr       0
    transactions    0
    dtype: int64
    
    


# Data Preprocessing


```python

# holidays_events 전처리 (National + Regional + Local; 실제 쉬는 날만)
hol = holidays.copy()
hol['date'] = pd.to_datetime(hol['date'], errors='coerce')

# transferred -> bool 정리
if hol['transferred'].dtype != bool:
    hol['transferred'] = (
        hol['transferred'].astype(str).str.strip().str.lower()
        .map({'true': True, 'false': False, '1': True, '0': False})
        .fillna(False)
    )

# 실제 쉬는 날만 남김:
#  - Holiday/Additional/Event & transferred==False
#  - 또는 type=='Transfer' (실제 기념일 날짜)
#  - Work Day / Bridge 제외
keep = ((hol['type'].isin(['Holiday','Additional','Event']) & (~hol['transferred']))
        | (hol['type'] == 'Transfer'))
hol = hol[keep & (~hol['type'].isin(['Work Day','Bridge']))].copy()

# locale 매핑으로 매장별로 확장
stores_map = stores[['store_nbr','city','state']].copy()

# National: 모든 매장에 해당
h_nat = (hol[hol['locale']=='National']
         .assign(_k=1).merge(stores_map.assign(_k=1), on='_k').drop(columns='_k'))

# Regional: state 매칭
h_reg = hol[hol['locale']=='Regional'] \
        .merge(stores_map, left_on='locale_name', right_on='state', how='inner')

# Local: city 매칭
h_loc = hol[hol['locale']=='Local'] \
        .merge(stores_map, left_on='locale_name', right_on='city',  how='inner')

# 조인용 키(중복 제거)
holidays_sd = (pd.concat([h_nat, h_reg, h_loc], ignore_index=True)
               [['store_nbr','date']]
               .drop_duplicates())

# train/test에 플래그 머지 (행 복제 없음)
train = train.merge(holidays_sd.assign(is_holiday=1), on=['store_nbr','date'], how='left')
test  = test.merge(holidays_sd.assign(is_holiday=1),  on=['store_nbr','date'], how='left')
train['is_holiday'] = train['is_holiday'].fillna(0).astype('int8')
test['is_holiday']  = test['is_holiday'].fillna(0).astype('int8')

print("train is_holiday mean:", float(train['is_holiday'].mean()))
print("test is_holiday mean:", float(test['is_holiday'].mean()))

```

    train is_holiday mean: 0.08076009501187649
    test is_holiday mean: 0.0023148148148148147



```python
# oil 
#주말, 공휴일 처리 
oil2 = oil[['date','dcoilwtico']].sort_values('date').copy()
oil2['dcoilwtico'] = oil2['dcoilwtico'].ffill()
train = pd.merge_asof(train.sort_values('date'), oil2, on='date', direction='backward').sort_index()
test  = pd.merge_asof(test.sort_values('date'),  oil2, on='date', direction='backward').sort_index()

for df in (train, test):
    df['dcoilwtico'] = df['dcoilwtico'].bfill()

print("train dcoilwtico NaN:", train['dcoilwtico'].isna().sum())
print("test dcoilwtico NaN:", test['dcoilwtico'].isna().sum())
```

    train dcoilwtico NaN: 0
    test dcoilwtico NaN: 0



```python
#add wage feature, 15th/last day of the month
for df in (train, test):
    dom = df['date'].dt.day
    dim = df['date'].dt.days_in_month  
    df['is_payday15']   = (dom == 15).astype(int)
    df['is_month_end']  = (dom == dim).astype(int)

print("[train]\n",train[['is_payday15','is_month_end']].mean()) 
print("[test]\n",test[['is_payday15','is_month_end']].mean()) 
```

    [train]
     is_payday15     0.033254
    is_month_end    0.032660
    dtype: float64
    [test]
     is_payday15     0.0000
    is_month_end    0.0625
    dtype: float64



```python
#transcations
transactions['date'] = pd.to_datetime(transactions['date'])

```

# EDA



```python
#total Sales in Train.csv
import matplotlib.pyplot as plt
sales_by_date = train.groupby('date', as_index=False)['sales'].sum()
sales_by_date['mean7'] = sales_by_date['sales'].rolling(7, min_periods=1).mean()

plt.figure(figsize=(12,4))
plt.plot(sales_by_date['date'], sales_by_date['sales'], alpha=0.35, label='Daily')
plt.plot(sales_by_date['date'], sales_by_date['mean7'], linewidth=2, label='7-day Mean')

plt.title('Total Sales over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.tight_layout()
plt.show()
```


    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_9_0.png)
    


seems to flutuacte daily but in general, there seems to be an increasing trend over the years and hitting lower sales in the beginning of the year-> possibly due to store closing during end of the year and for the New years-> holiday 인지 확인 필요


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 0) 전처리: cluster/holiday flags 머지 + onpromotion flag
#    - 이미 train, stores
# ─────────────────────────────────────────────────────────────
# cluster 머지 (없으면 자동 병합)
if "cluster" not in train.columns:
    train = train.merge(stores[["store_nbr", "cluster"]], on="store_nbr", how="left")

# onpromotion flag
train['promo_flag'] = (train['onpromotion'] > 0).astype(int)


def boxplot_by_category_pretty(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    top_k: int | None = None,
    rotate_labels: bool = True,
    sample_cap: int = 10000,
    log_y: bool = True,
    log_offset: float = 1e-3,
    palette: str = "tab20"
):
   
    if category_col not in df.columns:
        raise KeyError(f"'{category_col}' not in DataFrame. 현재 컬럼들: {list(df.columns)[:20]} ...")

    data = df[[category_col, "sales"]].copy()

    # Top-K 필터
    '''if top_k:
        keep = (data.groupby(category_col)["sales"]
                .sum()
                .sort_values(ascending=False)
                .head(top_k)
                .index.tolist())
        data = data[data[category_col].isin(keep)]'''

    groups, labels = [], []
    for key, grp in data.groupby(category_col):
        s = grp["sales"]
        # 시각화용 로그 오프셋 적용
        y = s.values
        if log_y:
            y = y + log_offset
        # 샘플링(속도)
        if len(y) > sample_cap:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(y), size=sample_cap, replace=False)
            y = y[idx]
        groups.append(y)
        labels.append(str(key))

    # 색상 팔레트
    cmap = plt.get_cmap(palette)
    colors = cmap(np.linspace(0, 1, len(groups)))

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(groups, labels=labels, showfliers=True, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.5)
    for w in bp["whiskers"]:
        w.set_color("black"); w.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color("black"); cap.set_linewidth(1.0)
    for fl in bp["fliers"]:
        fl.set_markerfacecolor("white")
        fl.set_markeredgecolor("black")
        fl.set_alpha(0.5)
        fl.set_markersize(3)

    if rotate_labels:
        plt.xticks(rotation=45, ha="right")

    if log_y:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel("Sales" + (" (log scale)" if log_y else ""))
    plt.tight_layout()
    plt.show()

# whisker Plot 호출
# (a) store_nbr — 상위 12개 매장
boxplot_by_category_pretty(train, "store_nbr", "Sales by Store") #, top_k=12

# (b) family — 상위 12개 상품군
boxplot_by_category_pretty(train, "family", "Sales by Family")#, top_k=12)

# (c) onpromotion flag — 0/1
boxplot_by_category_pretty(train, "promo_flag", "Sales by On-Promotion Flag (0/1)", rotate_labels=False)

# (d) cluster — 전체
boxplot_by_category_pretty(train, "cluster", "Sales by Store Cluster")


```


    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_11_0.png)
    



    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_11_1.png)
    



    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_11_2.png)
    



    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_11_3.png)
    



total sales 에 영향이 많은 변수 1)on promotion, 2)sales by family and possibly 3) sales by store

특히 family 별로(아이템 품목) 별로 sales 큰 차이를 보임



```python
# 날짜별 총 판매량
sales_daily = train.groupby('date')['sales'].sum().reset_index()

# 병합
merged = pd.merge(sales_daily, oil, on='date', how='left')

# 플롯
fig, ax1 = plt.subplots(figsize=(14,6))

# Sales plot (primary y-axis)
ax1.plot(merged['date'], merged['sales'], color='blue', label='Total Sales')
ax1.set_ylabel('Total Sales', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Oil price plot (secondary y-axis)
ax2 = ax1.twinx()
ax2.plot(merged['date'], merged['dcoilwtico'], color='orange', label='Oil Price (WTI)')
ax2.set_ylabel('Oil Price (USD/barrel)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# 급여일 표시 (15일, 말일)
paydays = merged[merged['date'].dt.day.isin([15, merged['date'].dt.daysinmonth])]
for day in paydays['date']:
    ax1.axvline(day, color='green', alpha=0.1)  # 연한 초록색 vertical line

# 지진일 표시
quake_date = pd.Timestamp('2016-04-16')
ax1.axvline(quake_date, color='red', linestyle='--', linewidth=2, label='Earthquake 2016-04-16')

# 제목 & 범례
fig.suptitle('Oil Price vs Total Sales with Paydays & Earthquake', fontsize=14)
ax1.grid(alpha=0.3)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()
```


    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_13_0.png)
    


by looking at the linear grpah, 직관적으로 보이는 oil price 와 total sales 의 상관관계는 보이지 않음, maybe a slight effect
earthquake 발생 시점 직후 sales 에 단기 급락. 하지만 금방 회복하는 경향성을 보임


```python
#heatmap, matrix graph
import seaborn as sns


eq_date = pd.to_datetime("2016-04-16")
train['is_earthquake'] = ((train['date'] >= eq_date) & 
                          (train['date'] <= eq_date + pd.Timedelta(days=30))).astype(int)


# correlation
base = train[['date','store_nbr','family','sales','onpromotion',
              'is_holiday','is_payday15','is_month_end','dcoilwtico','is_earthquake']].copy()

base = base.merge(transactions, on=['date','store_nbr'], how='left')

#RAW
cols_raw = ['sales','onpromotion','transactions','dcoilwtico',
            'is_holiday','is_payday15','is_month_end']
corr_raw = base[cols_raw].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr_raw, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation (Date–Store–Family, Raw)')
plt.tight_layout(); plt.show()

#LOG (극단치 완화)
base_log = base.copy()
base_log['log_sales']   = np.log1p(base_log['sales'])
base_log['log_onpromo'] = np.log1p(base_log['onpromotion'])
base_log['log_trans']   = np.log1p(base_log['transactions'].fillna(0))

cols_log = ['log_sales','log_onpromo','log_trans','dcoilwtico',
            'is_holiday','is_payday15','is_month_end','is_earthquake']
corr_log = base_log[cols_log].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr_log, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation (Date–Store–Family, Log-Scaled)')
plt.tight_layout(); plt.show()
```


    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_15_0.png)
    



    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_15_1.png)
    


onpromotion ↗ sales: 가장 강한 양의 상관. 로그 스케일에서 더 선명해짐(선형성↑).
transactions ↗ sales: 중간 정도 양의 상관. 안정적인 보조 지표.
dcoilwtico ↘ sales**: 약한 음의 상관. 신호 약해서 파생(이동평균/수익률)만 얕게 쓰는 게 무난.
is_holiday: 전체 상관은 낮지만, 특정 날 스파이크를 잘 설명 → 리드/래그/감쇠 피처로 활용 가치 큼.
log target: log1p로 변환하면 전반적 상관이 올라가고 학습이 안정적.



```python
#interaction term -> on promotion-sale and family
tmp = base.copy()
tmp['transactions'] = tmp['transactions'].fillna(0)

corr_by_family = (tmp.groupby('family')
                    .apply(lambda g: pd.Series({
                        'corr_sales_onpromo': g['sales'].corr(g['onpromotion']),
                    }))
                  ).sort_values('corr_sales_onpromo', ascending=False)

corr_by_family.head(10)
```

    /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:2897: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:2898: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    /tmp/ipykernel_36/1057378199.py:6: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      .apply(lambda g: pd.Series({





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corr_sales_onpromo</th>
    </tr>
    <tr>
      <th>family</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SCHOOL AND OFFICE SUPPLIES</th>
      <td>0.669793</td>
    </tr>
    <tr>
      <th>BEVERAGES</th>
      <td>0.372682</td>
    </tr>
    <tr>
      <th>HOME CARE</th>
      <td>0.369643</td>
    </tr>
    <tr>
      <th>HOME AND KITCHEN II</th>
      <td>0.362344</td>
    </tr>
    <tr>
      <th>PRODUCE</th>
      <td>0.356896</td>
    </tr>
    <tr>
      <th>BEAUTY</th>
      <td>0.342411</td>
    </tr>
    <tr>
      <th>SEAFOOD</th>
      <td>0.289401</td>
    </tr>
    <tr>
      <th>GROCERY I</th>
      <td>0.277722</td>
    </tr>
    <tr>
      <th>HOME AND KITCHEN I</th>
      <td>0.275865</td>
    </tr>
    <tr>
      <th>PET SUPPLIES</th>
      <td>0.270316</td>
    </tr>
  </tbody>
</table>
</div>



sales ↔ onpromotion
School & Office Supplies(0.67)가 압도적으로 높고, Beverages/Home Care/Produce도 꽤 반응적.
= 카테고리별 프로모션 탄력성이 크게 다름. 
일상 구매 품목일수록 onpromotion 영향이 고르게 있지만, 특정 카테고리는 행사 시에만 크게 뜀.


```python
corr_by_family = (tmp.groupby('family')
                    .apply(lambda g: pd.Series({
                        'corr_sales_trans'  : g['sales'].corr(g['transactions'])
                    }))
                  ).sort_values('corr_sales_trans', ascending=False)

corr_by_family
```

    /tmp/ipykernel_36/1472815095.py:2: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      .apply(lambda g: pd.Series({





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corr_sales_trans</th>
    </tr>
    <tr>
      <th>family</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GROCERY I</th>
      <td>0.854843</td>
    </tr>
    <tr>
      <th>CLEANING</th>
      <td>0.847347</td>
    </tr>
    <tr>
      <th>DAIRY</th>
      <td>0.844086</td>
    </tr>
    <tr>
      <th>BREAD/BAKERY</th>
      <td>0.829004</td>
    </tr>
    <tr>
      <th>PREPARED FOODS</th>
      <td>0.819452</td>
    </tr>
    <tr>
      <th>POULTRY</th>
      <td>0.817598</td>
    </tr>
    <tr>
      <th>DELI</th>
      <td>0.802662</td>
    </tr>
    <tr>
      <th>PERSONAL CARE</th>
      <td>0.780476</td>
    </tr>
    <tr>
      <th>SEAFOOD</th>
      <td>0.765203</td>
    </tr>
    <tr>
      <th>BEVERAGES</th>
      <td>0.764050</td>
    </tr>
    <tr>
      <th>BEAUTY</th>
      <td>0.691216</td>
    </tr>
    <tr>
      <th>EGGS</th>
      <td>0.649263</td>
    </tr>
    <tr>
      <th>GROCERY II</th>
      <td>0.645475</td>
    </tr>
    <tr>
      <th>AUTOMOTIVE</th>
      <td>0.623862</td>
    </tr>
    <tr>
      <th>MEATS</th>
      <td>0.601522</td>
    </tr>
    <tr>
      <th>PLAYERS AND ELECTRONICS</th>
      <td>0.558262</td>
    </tr>
    <tr>
      <th>PET SUPPLIES</th>
      <td>0.548831</td>
    </tr>
    <tr>
      <th>PRODUCE</th>
      <td>0.536328</td>
    </tr>
    <tr>
      <th>HOME CARE</th>
      <td>0.516016</td>
    </tr>
    <tr>
      <th>FROZEN FOODS</th>
      <td>0.511028</td>
    </tr>
    <tr>
      <th>LADIESWEAR</th>
      <td>0.472166</td>
    </tr>
    <tr>
      <th>LIQUOR,WINE,BEER</th>
      <td>0.470864</td>
    </tr>
    <tr>
      <th>HOME APPLIANCES</th>
      <td>0.455041</td>
    </tr>
    <tr>
      <th>MAGAZINES</th>
      <td>0.420329</td>
    </tr>
    <tr>
      <th>LAWN AND GARDEN</th>
      <td>0.395984</td>
    </tr>
    <tr>
      <th>HOME AND KITCHEN II</th>
      <td>0.394980</td>
    </tr>
    <tr>
      <th>HOME AND KITCHEN I</th>
      <td>0.372037</td>
    </tr>
    <tr>
      <th>CELEBRATION</th>
      <td>0.367621</td>
    </tr>
    <tr>
      <th>HARDWARE</th>
      <td>0.361383</td>
    </tr>
    <tr>
      <th>LINGERIE</th>
      <td>0.357617</td>
    </tr>
    <tr>
      <th>SCHOOL AND OFFICE SUPPLIES</th>
      <td>0.169533</td>
    </tr>
    <tr>
      <th>BOOKS</th>
      <td>0.147033</td>
    </tr>
    <tr>
      <th>BABY CARE</th>
      <td>-0.004081</td>
    </tr>
  </tbody>
</table>
</div>



sales ↔ transactions_proxy
대다수 식품/소모재( Grocery I, Cleaning, Dairy … )에서 0.8로 매우 높아. 즉 transaction 이 곧 매출을 강하게 설명.
반면 Books, School & Office, Baby Care 등은 낮거나 거의 0 → 트래픽보다 행사/시즌성/재고의 영향이 큼.

# seasonaility/ time feature catpure


```python
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def seasonal_plot_weekday(df, date_col='date', y_col='sales'):
    z = df[[date_col, y_col]].copy()
    z[date_col] = pd.to_datetime(z[date_col])
    z['week'] = z[date_col].dt.isocalendar().week.astype(int)
    z['dow'] = z[date_col].dt.dayofweek
    mat = z.pivot_table(index='dow', columns='week', values=y_col, aggfunc='mean')

    plt.figure(figsize=(9,3))
    for w in mat.columns:
        plt.plot(mat.index, mat[w].values, alpha=0.8)
    plt.title('Seasonal Plot (week/day)')
    plt.xlabel('day'); plt.ylabel(y_col)
    plt.show()
seasonal_plot_weekday(train, 'date', 'sales')

```


    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_22_0.png)
    


주간 패턴이 뚜렷: 대부분의 주에서 월–목 저점 → 금·토·일 상승. 일요일(6)이 가장 높게 끝나는 라인이 많음 → dayofweek, is_weekend은 강신호.
주별 편차: 라인 간 간격이 꽤 있어 프로모션/휴일/급여일이 주간 패턴을 증폭/약화. ⇒ onpromotion, is_payday15, is_month_end, promo×holiday feature 중요하게 보임


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# 0) 주간 시즌성 제거 컬럼 생성
# ---------------------------
def build_weekly_deseason(df, date_col='date', y_col='sales', out_col='sales_deseason'):
    """
    요일 패턴(weekly)을 제거한 값을 out_col로 추가해 반환.
    - 일단 전체 일매출 합계 기준으로 요일 인덱스를 추정(간단/안전)
    - 그룹별로 더 정교하게 하고 싶으면 확장 가능
    """
    d = df[[date_col, y_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])

    # 날짜별 합계 → 요일 패턴 인덱스 계산
    daily = d.groupby(date_col, as_index=False)[y_col].sum()
    daily['dow'] = pd.to_datetime(daily[date_col]).dt.dayofweek  # 0=Mon ... 6=Sun
    dow_mean = daily.groupby('dow')[y_col].mean()
    dow_index = (dow_mean / dow_mean.mean()).to_dict()  # ex) Fri>1, Sun<1

    # 각 날짜의 요일 인덱스 매핑
    daily['weekly_factor'] = daily['dow'].map(dow_index)
    factor_map = daily.set_index(date_col)['weekly_factor']

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out['weekly_factor'] = out[date_col].map(factor_map).fillna(1.0)
    out[out_col] = out[y_col].astype(float) / np.clip(out['weekly_factor'].to_numpy(), 1e-12, None)
    return out

# train에 sales_deseason 없으면 생성
if 'sales_deseason' not in train.columns:
    train = build_weekly_deseason(train, 'date', 'sales', 'sales_deseason')

# -----------------------------------------
# 1) 주기 스펙트럼(연간 사이클 축) 플로팅 함수 (원문 유지)
# -----------------------------------------
def periodogram_cycles_per_year(df, date_col='date', y_col='sales',
                                agg='mean', detrend=True, use_window=True,
                                smooth_win=5, epsilon=1e-12):
    """
    Plot periodogram with x-axis in cycles/year (1=annual, 52=weekly, 104=semiweekly).
    Smoothing + normalization for a 'Kaggle/Medium' look.
    """
    s = df[[date_col, y_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    # daily aggregate
    if agg == 'sum':
        daily = s.groupby(date_col, as_index=False)[y_col].sum()
    else:
        daily = s.groupby(date_col, as_index=False)[y_col].mean()
    daily = daily.sort_values(date_col)

    x = daily[y_col].astype(float).to_numpy()
    if detrend:
        x = x - np.nanmean(x)
    # fill NaN if any
    x = np.where(np.isnan(x), np.nanmean(x), x)

    # optional window to reduce leakage
    if use_window:
        w = np.hanning(len(x))
        xw = x * w
        norm = (w**2).sum()
    else:
        xw = x
        norm = len(x)

    # one-sided FFT
    fft = np.fft.rfft(xw)
    power = (fft.real**2 + fft.imag**2) / (norm + epsilon)  # normalized power

    # frequencies in cycles/day → cycles/year
    n = len(xw)
    freqs_per_day = np.fft.rfftfreq(n, d=1.0)  # 1/day
    freqs_cy = freqs_per_day * 365.25          # cycles/year
    # drop DC
    freqs_cy = freqs_cy[1:]
    power = power[1:]

    # simple smoothing (moving average on power)
    if smooth_win and smooth_win > 1:
        k = smooth_win
        kernel = np.ones(k) / k
        power_sm = np.convolve(power, kernel, mode='same')
    else:
        power_sm = power

    # plot
    plt.figure(figsize=(10,3.2))
    plt.plot(freqs_cy, power_sm)
    # reference lines (cycles/year)
    refs = {
        'Annual (1)': 1,
        'Semiannual (2)': 2,
        'Quarterly (~4)': 4,
        'Bimonthly (~6)': 6,
        'Monthly (~12)': 12,
        'Biweekly (~26)': 26,
        'Weekly (52)': 52,
        'Semiweekly (104)': 104
    }
    ymax = power_sm.max() if np.isfinite(power_sm.max()) else 1.0
    for lbl, f in refs.items():
        if freqs_cy.min() <= f <= freqs_cy.max():
            plt.axvline(f, ls='--', alpha=0.4)
            plt.text(f, ymax*0.9, lbl, rotation=90, va='top', ha='right', fontsize=8)

    plt.xlim(0, min(120, freqs_cy.max()))
    plt.xlabel('Cycles per year'); plt.ylabel('Variance (normalized)')
    plt.title('Periodogram')
    plt.tight_layout()
    plt.show()

# ---------------------------
# 2) 플롯: 원본 vs 디시즈널
# ---------------------------
# 합계 스펙트럼을 보고 싶다면 agg='sum' 추천
periodogram_cycles_per_year(train, 'date', 'sales', agg='sum')
periodogram_cycles_per_year(train, 'date', 'sales_deseason', agg='sum')


```


    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_24_0.png)
    



    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_24_1.png)
    


By looking at the periodgram, the weekly effect is dominant. 
There is a huge spike at ~52 cycles/year ⇒ period ≈ 7 days-> Day-of-week effects are the main driver.

The clear spike at ~104 cycles/year ⇒ second harmonic of the week (≈3.5 days). That usually means the weekly pattern is asymmetric (e.g., weekdays vs weekend).

weaker = biweekly/monthly


```python

def plot_year_deseason_with_holidays(df_ds, holidays_df, year=2017,
                                     date_col='date', y_col='sales_deseason',
                                     holiday_locales=('National',),
                                     line_color='black', marker_color='red'):
    """
    Plot deseasonalized daily series for a single year with holiday markers.
    df_ds: DataFrame with date_col and y_col (deseasonalized values)
    holidays_df: holidays_events.csv loaded
    holiday_locales: tuple of locales to mark ('National','Regional','Local')
    """
    df = df_ds.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    # filter for the year
    df = df[df[date_col].dt.year == year]
    # daily mean
    daily = df.groupby(date_col, as_index=False)[y_col].mean()

    # filter holidays
    h = holidays_df.copy()
    h[date_col] = pd.to_datetime(h[date_col])
    h = h[h['locale'].isin(holiday_locales)]
    h_year = h[h[date_col].dt.year == year][date_col].drop_duplicates()

    # plot
    plt.figure(figsize=(10,3))
    plt.plot(daily[date_col], daily[y_col], color=line_color, alpha=0.7, marker='o', markersize=3)
    # mark holidays as red dots
    holiday_points = daily[daily[date_col].isin(h_year)]
    plt.scatter(holiday_points[date_col], holiday_points[y_col],
                color=marker_color, label='holidays', zorder=3)
    plt.title(f"{year} Average Daily Sales - Deseasonalized")
    plt.xlabel('date'); plt.ylabel(y_col)
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_year_deseason_with_holidays(train, holidays,
                                 year=2017,
                                 holiday_locales=('National','Regional'))

```


    
![png](fe-xgm-nested-final_files/fe-xgm-nested-final_26_0.png)
    


season(e.g,day of the week)을 제외하고 큰 피크(low/high) is likely to be explained by holidays. The effect of those peaks seem to last for short amount of time- approxately 1-3 days However, some peaks that are not explained by holidays can be due to 급여일 or promotion. 

Keep holiday, 급여일/ 프로모션 as well as its interaction effect. 

# Feature Engineering



```python
from sklearn.preprocessing import LabelEncoder
import numpy as np, pandas as pd

for df in (train, test, transactions):
    df['date'] = pd.to_datetime(df['date'])
holidays['date'] = pd.to_datetime(holidays['date'])   # 전처리 끝난 clean holidays
stores['date'] = pd.to_datetime(stores.get('date', pd.NaT)) if 'date' in stores.columns else pd.NaT

#  캘린더 파생 + 전역 time_idx

GLOBAL_MIN_DATE = min(train['date'].min(), test['date'].min())
#earthquake affect period
EQ_START = pd.Timestamp('2016-04-16')
EQ_END   = pd.Timestamp('2016-05-31')
for df in (train, test):
    df['dow']         = df['date'].dt.dayofweek
    df['month']       = df['date'].dt.month
    df['year']        = df['date'].dt.year
    df['day']         = df['date'].dt.day
    df['weekofyear']  = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend']  = (df['dow'] >= 5).astype(int)
    df['is_payday15'] = (df['day'] == 15).astype(int)
    df['is_month_end']= (df['date'] == df['date'] + pd.offsets.MonthEnd(0)).astype(int)
    df['time_idx']    = (df['date'] - pd.to_datetime(GLOBAL_MIN_DATE)).dt.days
    df['is_quake_window'] = ((df['date'] >= EQ_START) & (df['date'] <= EQ_END)).astype(np.int8)

# Fourier (연/주기) - 점(.) 없는 이름

for k in (1,2,3):   # yearly order=3
    for df in (train, test):
        df[f'sin_year_{k}'] = np.sin(2*np.pi*k*df['time_idx']/365.25)
        df[f'cos_year_{k}'] = np.cos(2*np.pi*k*df['time_idx']/365.25)
for k in (1,2):     # weekly order=2
    for df in (train, test):
        df[f'sin_week_{k}'] = np.sin(2*np.pi*k*df['time_idx']/7.0)
        df[f'cos_week_{k}'] = np.cos(2*np.pi*k*df['time_idx']/7.0)

# Transactions Proxy 

trans_dow = (transactions
             .assign(dow=transactions['date'].dt.dayofweek)
             .groupby(['store_nbr','dow'])['transactions']
             .mean()
             .reset_index(name='trans_dow_mean'))

def add_trans_proxy(df):
    out = df.merge(trans_dow, on=['store_nbr','dow'], how='left')
    store_mean = out.groupby('store_nbr')['trans_dow_mean'].transform('mean')
    out['transactions_proxy'] = out['trans_dow_mean'].fillna(store_mean) \
                                                   .fillna(out['trans_dow_mean'].mean())
    return out.drop(columns=['trans_dow_mean'])

train = add_trans_proxy(train)
test  = add_trans_proxy(test)

#  프로모션 스케일 보정
train['promo_log1p'] = np.log1p(train['onpromotion'])
test['promo_log1p']  = np.log1p(test['onpromotion'])

# 라벨 인코딩
le_store  = LabelEncoder().fit(pd.concat([train['store_nbr'], test['store_nbr']]))
le_family = LabelEncoder().fit(pd.concat([train['family'].astype(str), test['family'].astype(str)]))

train['store_le']  = le_store.transform(train['store_nbr'])
test['store_le']   = le_store.transform(test['store_nbr'])
train['family_le'] = le_family.transform(train['family'].astype(str))
test['family_le']  = le_family.transform(test['family'].astype(str))

# Lag & Rolling (누수 방지: train+test 합쳐 과거만 사용)-> short-term memory
'''TRAIN_END = pd.Timestamp('2017-08-15')  # 테스트 직전
train = train[train['date'] <= TRAIN_END].copy()     # 라벨 NaN/누수 차단

def rebuild_lag_roll(train_df, test_df):
    tmp = pd.concat([
        train_df.assign(_is_train=1),
        test_df.assign(_is_train=0)
    ], ignore_index=True, sort=False)

    tmp = tmp.sort_values(['store_nbr','family','date']).reset_index(drop=True)

    # lags
    for lag in (7, 14, 28):
        tmp[f'lag_{lag}'] = tmp.groupby(['store_nbr','family'], sort=False)['sales'].shift(lag)

    # rollings (과거만 사용)
    for w in (7, 30):
        tmp[f'rolling_mean_{w}'] = (
            tmp.groupby(['store_nbr','family'], sort=False)['sales']
               .shift(1).rolling(w, min_periods=1).mean()
        )

    train_new = tmp[tmp['_is_train'] == 1].drop(columns=['_is_train']).copy()
    test_new  = tmp[tmp['_is_train'] == 0].drop(columns=['_is_train']).copy()
    return train_new, test_new

train, test = rebuild_lag_roll(train, test)

# 라벨 안전 확인(더 이상 NaN/음수/Inf 없어야 함)
_bad = (~np.isfinite(train['sales'])) | (train['sales'].isna()) | (train['sales'] < 0)
assert int(_bad.sum()) == 0, f"still bad labels: {int(_bad.sum())}"
'''

#Holidays — 매장별 확장 → (유니크 store-date로) 조인 → 상호작용/거리
#    - holidays: 전처리 끝난 clean 테이블

stores_map = stores[['store_nbr','city','state']].copy()

# 7-1) 매장별로 휴일 확장
h_nat = (holidays[holidays['locale']=='National']
         .assign(_k=1).merge(stores_map.assign(_k=1), on='_k').drop(columns='_k'))
h_reg = holidays[holidays['locale']=='Regional'] \
        .merge(stores_map, left_on='locale_name', right_on='state', how='inner')
h_loc = holidays[holidays['locale']=='Local'] \
        .merge(stores_map, left_on='locale_name', right_on='city',  how='inner')

holidays_sd = (pd.concat([h_nat, h_reg, h_loc], ignore_index=True)
               [['store_nbr','date']]
               .drop_duplicates()
               .sort_values(['store_nbr','date'])
               .reset_index(drop=True))

#조인 → is_holiday (indicator 방식: 절대 누락 안 됨) ===
n_before_tr, n_before_te = len(train), len(test)

holidays_sd = holidays_sd[['store_nbr','date']].drop_duplicates()

# dtype/타입 맞추기
train['date'] = pd.to_datetime(train['date'])
test['date']  = pd.to_datetime(test['date'])
holidays_sd['date'] = pd.to_datetime(holidays_sd['date'])
train['store_nbr'] = train['store_nbr'].astype(holidays_sd['store_nbr'].dtype, copy=False)
test['store_nbr']  = test['store_nbr'].astype(holidays_sd['store_nbr'].dtype,  copy=False)

# indicator merge → BOTH이면 휴일
train = train.merge(holidays_sd, on=['store_nbr','date'], how='left', indicator='_holhit')
test  = test.merge(holidays_sd, on=['store_nbr','date'], how='left', indicator='_holhit')

train['is_holiday'] = (train['_holhit'] == 'both').astype(np.int8)
test['is_holiday']  = (test['_holhit'] == 'both').astype(np.int8)

# indicator 및 우측 키 제거(이미 on으로 합쳐져 있음)
train.drop(columns=['_holhit'], inplace=True)
test.drop(columns=['_holhit'], inplace=True)

assert len(train) == n_before_tr, "Holiday merge duplicated train rows"
assert len(test)  == n_before_te, "Holiday merge duplicated test rows"

# 휴일 × 프로모션
train['is_holiday_and_promo'] = (train['is_holiday'] * train['onpromotion']).astype(np.int32)
test['is_holiday_and_promo']  = (test['is_holiday']  * test['onpromotion']).astype(np.int32)

# 휴일까지/이후 거리 (매장별 searchsorted; )
def add_holiday_distances_per_store(df, holidays_sd):
    df = df[['store_nbr','date']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store_nbr','date'])
    out = pd.DataFrame(index=df.index, columns=['days_since_holiday','days_until_holiday'], dtype='float32')

    # 매장별 휴일 배열 준비(날짜만)
    holidays_sd = holidays_sd[['store_nbr','date']].drop_duplicates().sort_values(['store_nbr','date'])
    for s, g in df.groupby('store_nbr', sort=False):
        hdates = holidays_sd.loc[holidays_sd['store_nbr']==s, 'date'].values.astype('datetime64[D]')
        if hdates.size == 0:
            continue
        dates = g['date'].values.astype('datetime64[D]')

        idx_prev = np.searchsorted(hdates, dates, side='right') - 1
        prev = np.where(idx_prev >= 0, hdates[idx_prev], np.datetime64('NaT','D'))

        idx_next = np.searchsorted(hdates, dates, side='left')
        nxt = np.where(idx_next < hdates.size, hdates[idx_next], np.datetime64('NaT','D'))

        out.loc[g.index, 'days_since_holiday'] = (dates - prev).astype('timedelta64[D]').astype('float32')
        out.loc[g.index, 'days_until_holiday'] = (nxt - dates).astype('timedelta64[D]').astype('float32')
    return out

dist_tr = add_holiday_distances_per_store(train, holidays_sd)
dist_te = add_holiday_distances_per_store(test,  holidays_sd)

train['days_since_holiday'] = dist_tr['days_since_holiday'].values
train['days_until_holiday'] = dist_tr['days_until_holiday'].values
test['days_since_holiday']  = dist_te['days_since_holiday'].values
test['days_until_holiday']  = dist_te['days_until_holiday'].values



```


```python
# final features
FEATS = [
    # IDs
    'store_le','family_le',
    # calendar
    'dow','month','year','day','weekofyear','is_weekend','is_payday15','is_month_end','time_idx',
    # fourier
    'sin_year_1','cos_year_1','sin_year_2','cos_year_2','sin_year_3','cos_year_3',
    'sin_week_1','cos_week_1','sin_week_2','cos_week_2',
    # promo
    'onpromotion','promo_log1p',
    # oil
    'dcoilwtico',
    # transactions proxy
    'transactions_proxy',
    # lags/rollings
    #'lag_7','lag_14','lag_28','rolling_mean_7','rolling_mean_30',
    # holidays
    'is_holiday','is_holiday_and_promo','days_since_holiday','days_until_holiday',
    #earthquake
    'is_quake_window',
]
print(f"#FEATS={len(FEATS)}")
print(f"#FEATS={len(FEATS)}")

```

    #FEATS=30
    #FEATS=30


#  Nested XGBoost Inner CV: 최근 rolling 창 + 1년 전(anchor) rolling 창


 Nested Time-series CV summary
 - Outer folds (2x15d):
   2017/08/01 to 08/15 and 2017/01/01 to 01/16
    as outer validation windows.
    (15 days like the test sample) 
 - Inner CV (per outer fold):
   Build multiple validation windows BEFORE the outer window:
     * Recent windows (rolling, step=30d)
     * Seasonal windows around ~1y before (anchor), step=30d
   For each trial (Optuna TPE):
     - Train on data strictly before each inner window, validate on that window
     - Early stopping (rmse) -> get best_iteration
     - Aggregate scores with weights (recent/year) -> trial score
 - Select best params per outer fold, then evaluate on the outer window (no leakage).
 - Pick the outer fold with better RMSE(log1p) -> FINAL_PARAMS & BEST_ROUND.
 - Final: train on ALL train with FINAL_PARAMS for BEST_ROUND, predict test.


```python
# Nested XGBoost (xgb.train) + Optuna 
#  - Inner CV: 최근 rolling 창 + 1년 전(anchor) rolling 창 (가중 평균)
#  - Outer folds: 고정 2창 (2017-08 / 2017-01)

USE_GPU    = True          # GPU 없으면 False
N_TRIALS   = 5            # Optuna trial 수
INNER_SPLITS_RECENT = 3    # 최근창 개수
INNER_SPLITS_YEAR   = 2    # 1년 전(anchor) 창 개수
EARLY_STOP = 200           # early stopping rounds
MAX_ROUNDS = 6000          # 상한 boosting rounds
SEED       = 73

import numpy as np, pandas as pd
import xgboost as xgb
import optuna

# 
train[FEATS] = train[FEATS].replace([np.inf, -np.inf], np.nan)
test[FEATS]  = test[FEATS].replace([np.inf, -np.inf], np.nan)
for c in FEATS:
    med = train[c].median()
    if pd.isna(med): med = 0.0
    train[c] = train[c].fillna(med).astype(np.float32)
    test[c]  = test[c].fillna(med).astype(np.float32)

# 
def rmse_log(y_true_log, y_pred_log):
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))

def _log1p_clip(s):
    return np.log1p(pd.to_numeric(s, errors='coerce').fillna(0).clip(lower=0)).astype(np.float32)

def make_splits(df, val_start, val_end):
    trn = df[df['date'] <  val_start].copy()
    val = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
    trn = trn[trn['sales'].notna()]
    val = val[val['sales'].notna()]
    X_trn = trn[FEATS].astype(np.float32)
    y_trn = _log1p_clip(trn['sales'])
    X_val = val[FEATS].astype(np.float32)
    y_val = _log1p_clip(val['sales'])
    return X_trn, y_trn, X_val, y_val

#
BASE_PARAMS = dict(
    objective='reg:squarederror',
    eval_metric='rmse',
    tree_method='hist',
    seed=SEED
)
if USE_GPU:
    BASE_PARAMS['device'] = 'cuda'  

# Robust inner-CV objective: 최근 + 1년 전 창 (가중 평균) -
def xgb_objective(train_df, val_start, val_end,
                  *,  # recent 창 설정
                  n_inner_recent=3,
                  recent_days=400,
                  recent_step=30,
                  # 1년 anchor 창 설정
                  n_inner_year=2,
                  year_back_days=365,
                  year_step=30,
                  # 공통 창 설정
                  inner_window_len=15,
                  inner_gap_to_val=16,
                  min_trn_rows=50_000,
                  min_val_rows=2_000,
                  weight_recent=0.6,
                  weight_year=0.4,
                  debug=False):

    # 최근 구간 후보 윈도우
    recent_windows = []
    for k in range(max(n_inner_recent*2, n_inner_recent)):  # 넉넉히 생성 후 필터
        end   = val_start - pd.Timedelta(days=inner_gap_to_val + k*recent_step)
        start = end - pd.Timedelta(days=inner_window_len - 1)
        recent_windows.append(('recent', pd.Timestamp(start), pd.Timestamp(end)))

    # 1년 전(anchor) 윈도우 (val_start - 1y 기준으로 뒤로 year_step 간격)
    year_windows = []
    base_anchor_end = val_start - pd.Timedelta(days=year_back_days + inner_gap_to_val)
    for k in range(max(n_inner_year*2, n_inner_year)):
        end   = base_anchor_end - pd.Timedelta(days=k*year_step)
        start = end - pd.Timedelta(days=inner_window_len - 1)
        # 반드시 val_start 이전 데이터만 사용
        if end < val_start:
            year_windows.append(('year', pd.Timestamp(start), pd.Timestamp(end)))

    candidate_windows = recent_windows + year_windows
    # 시간 오름차순 정렬
    candidate_windows.sort(key=lambda x: x[2])  # end 기준

    if debug:
        print(f"[inner] candidates: recent={len(recent_windows)}, year={len(year_windows)}")

    def objective(trial: optuna.Trial):
        params = dict(
            learning_rate   = trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
            max_depth       = trial.suggest_int  ('max_depth', 4, 12),
            min_child_weight= trial.suggest_float('min_child_weight', 1.0, 20.0),
            subsample       = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree= trial.suggest_float('colsample_bytree', 0.6, 1.0),
            **BASE_PARAMS
        )
        # L1/L2
        params['lambda'] = trial.suggest_float('lambda', 0.0, 2.0)
        params['alpha']  = trial.suggest_float('alpha',  0.0, 2.0)

        scores_recent, scores_year = [], []
        used_recent, used_year = 0, 0

        for tag, iw_start, iw_end in candidate_windows:
            # 창 별 train/val 생성
            trn = train_df[(train_df['date'] < iw_start)]
            val = train_df[(train_df['date'] >= iw_start) & (train_df['date'] <= iw_end)]
            trn = trn[trn['sales'].notna()]
            val = val[val['sales'].notna()]

            if trn.shape[0] < min_trn_rows or val.shape[0] < min_val_rows:
                continue

            dtr = xgb.DMatrix(
                trn[FEATS].astype(np.float32),
                label=np.log1p(trn['sales'].clip(lower=0)).to_numpy(np.float32)
            )
            dvl = xgb.DMatrix(
                val[FEATS].astype(np.float32),
                label=np.log1p(val['sales'].clip(lower=0)).to_numpy(np.float32)
            )

            bst = xgb.train(
                params, dtr,
                num_boost_round=MAX_ROUNDS,
                evals=[(dvl,'val')],
                early_stopping_rounds=EARLY_STOP,
                verbose_eval=False
            )
            pred_log = bst.predict(dvl, iteration_range=(0, bst.best_iteration+1))
            yv_log   = np.log1p(val['sales'].clip(lower=0)).to_numpy(np.float32)
            score = float(np.sqrt(np.mean((yv_log - pred_log)**2)))

            if tag == 'recent' and used_recent < n_inner_recent:
                scores_recent.append(score); used_recent += 1
            elif tag == 'year' and used_year < n_inner_year:
                scores_year.append(score);   used_year   += 1

            if used_recent >= n_inner_recent and used_year >= n_inner_year:
                break

        # 가중 평균 (둘 다 있으면), 하나만 있으면 그 값 사용, 없으면 큰 페널티
        if scores_recent and scores_year:
            return weight_recent*np.mean(scores_recent) + weight_year*np.mean(scores_year)
        elif scores_recent:
            return float(np.mean(scores_recent))
        elif scores_year:
            return float(np.mean(scores_year))
        else:
            if debug: print("[inner] no usable windows → penalty")
            return 1e9

    return objective

#  Outer folds 정의 & 튜닝 ----
outer_folds = [
    (pd.Timestamp('2017-08-01'), pd.Timestamp('2017-08-15')),  # 테스트 직전
    (pd.Timestamp('2017-01-01'), pd.Timestamp('2017-01-15')),  # 반대 시즌
]

best_bundles = []   # [{'params':..., 'outer_score':..., 'best_iter':...}, ...]

for (vstart, vend) in outer_folds:
    # Inner 튜닝
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(
        xgb_objective(
            train, vstart, vend,
            n_inner_recent=INNER_SPLITS_RECENT,
            recent_days=400,
            recent_step=30,
            n_inner_year=INNER_SPLITS_YEAR,
            year_back_days=365,
            year_step=30,
            inner_window_len=15,
            inner_gap_to_val=16,
            weight_recent=0.6,
            weight_year=0.4,
            debug=False
        ),
        n_trials=N_TRIALS,
        show_progress_bar=False
    )

    # Inner 최적 파라미터로 Outer 평가
    bestp = dict(BASE_PARAMS)
    bestp.update(study.best_params)

    X_trn, y_trn, X_val, y_val = make_splits(train, vstart, vend)
    dtr = xgb.DMatrix(X_trn, label=y_trn)
    dvl = xgb.DMatrix(X_val, label=y_val)

    bst = xgb.train(bestp, dtr, num_boost_round=MAX_ROUNDS,
                    evals=[(dvl,'val')], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    pred_log   = bst.predict(dvl, iteration_range=(0, bst.best_iteration+1))
    outer_score= rmse_log(y_val, pred_log)
    best_iter  = int(bst.best_iteration + 1)

    best_bundles.append({
        'params': bestp,
        'outer_score': float(outer_score),
        'best_iter': best_iter
    })
    print(f"[Outer {vstart.date()}~{vend.date()}] RMSE(log1p)={outer_score:.5f}, best_round={best_iter}")

# Outer 성능이 좋은 
best_idx = int(np.argmin([b['outer_score'] for b in best_bundles]))
FINAL_PARAMS = best_bundles[best_idx]['params']
BEST_ROUND   = best_bundles[best_idx]['best_iter']
print("Chosen FINAL_PARAMS:", FINAL_PARAMS)
print("Chosen BEST_ROUND:", BEST_ROUND)

#Final train on all train & submit ----
dall = xgb.DMatrix(
    train[FEATS].astype(np.float32),
    label=_log1p_clip(train['sales'])
)
bst_final = xgb.train(FINAL_PARAMS, dall, num_boost_round=BEST_ROUND, verbose_eval=False)

dtest = xgb.DMatrix(test[FEATS].astype(np.float32))
test_pred_log = bst_final.predict(dtest)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

sub = test[['id']].copy()
sub['sales'] = test_pred.astype(float)
sub.to_csv('/kaggle/working/submission_nested_xgb.csv', index=False)
print('Saved -> /kaggle/working/submission_nested_xgb.csv')

```

    [I 2025-08-18 04:22:35,701] A new study created in memory with name: no-name-c0d2db93-d232-4f84-9a09-3e9c149d4dec
    [I 2025-08-18 04:31:01,642] Trial 0 finished with value: 0.4707276105880738 and parameters: {'learning_rate': 0.08784557936108438, 'max_depth': 8, 'min_child_weight': 10.72606166671397, 'subsample': 0.8441028908451985, 'colsample_bytree': 0.7971955934444767, 'lambda': 0.4289239955828703, 'alpha': 0.48796003857516235}. Best is trial 0 with value: 0.4707276105880738.
    [I 2025-08-18 04:40:43,259] Trial 1 finished with value: 0.45074030160903933 and parameters: {'learning_rate': 0.07466628121356979, 'max_depth': 10, 'min_child_weight': 6.901931086064769, 'subsample': 0.7616332697570256, 'colsample_bytree': 0.8387200191357007, 'lambda': 0.4828695337572342, 'alpha': 1.8114396828678088}. Best is trial 1 with value: 0.45074030160903933.
    [I 2025-08-18 05:02:45,978] Trial 2 finished with value: 0.4430024325847626 and parameters: {'learning_rate': 0.028856378820378666, 'max_depth': 9, 'min_child_weight': 5.977459761804991, 'subsample': 0.9505358881481207, 'colsample_bytree': 0.9676301722988521, 'lambda': 0.20994580977352673, 'alpha': 1.9413622188311883}. Best is trial 2 with value: 0.4430024325847626.
    [I 2025-08-18 05:13:29,121] Trial 3 finished with value: 0.6124200105667115 and parameters: {'learning_rate': 0.02210331195400937, 'max_depth': 4, 'min_child_weight': 11.11903656054751, 'subsample': 0.6109818181387561, 'colsample_bytree': 0.7349988033071791, 'lambda': 0.3866719147561466, 'alpha': 0.5373600327568264}. Best is trial 2 with value: 0.4430024325847626.
    [I 2025-08-18 05:19:03,100] Trial 4 finished with value: 0.46756984591484074 and parameters: {'learning_rate': 0.18962833427410933, 'max_depth': 12, 'min_child_weight': 16.577824547537723, 'subsample': 0.8620900144280699, 'colsample_bytree': 0.6221725799648963, 'lambda': 1.378321708072992, 'alpha': 1.2735708456832626}. Best is trial 2 with value: 0.4430024325847626.
    [I 2025-08-18 05:22:51,406] A new study created in memory with name: no-name-8e3b7b43-830a-44c0-8d1c-612a951195c8


    [Outer 2017-08-01~2017-08-15] RMSE(log1p)=0.40996, best_round=5997


    [I 2025-08-18 05:34:22,179] Trial 0 finished with value: 0.4958416819572449 and parameters: {'learning_rate': 0.08784557936108438, 'max_depth': 8, 'min_child_weight': 10.72606166671397, 'subsample': 0.8441028908451985, 'colsample_bytree': 0.7971955934444767, 'lambda': 0.4289239955828703, 'alpha': 0.48796003857516235}. Best is trial 0 with value: 0.4958416819572449.
    [I 2025-08-18 05:45:33,709] Trial 1 finished with value: 0.4861896693706512 and parameters: {'learning_rate': 0.07466628121356979, 'max_depth': 10, 'min_child_weight': 6.901931086064769, 'subsample': 0.7616332697570256, 'colsample_bytree': 0.8387200191357007, 'lambda': 0.4828695337572342, 'alpha': 1.8114396828678088}. Best is trial 1 with value: 0.4861896693706512.
    [I 2025-08-18 06:05:52,706] Trial 2 finished with value: 0.4786391258239746 and parameters: {'learning_rate': 0.028856378820378666, 'max_depth': 9, 'min_child_weight': 5.977459761804991, 'subsample': 0.9505358881481207, 'colsample_bytree': 0.9676301722988521, 'lambda': 0.20994580977352673, 'alpha': 1.9413622188311883}. Best is trial 2 with value: 0.4786391258239746.
    [I 2025-08-18 06:16:18,950] Trial 3 finished with value: 0.7022951006889343 and parameters: {'learning_rate': 0.02210331195400937, 'max_depth': 4, 'min_child_weight': 11.11903656054751, 'subsample': 0.6109818181387561, 'colsample_bytree': 0.7349988033071791, 'lambda': 0.3866719147561466, 'alpha': 0.5373600327568264}. Best is trial 2 with value: 0.4786391258239746.
    [I 2025-08-18 06:21:34,591] Trial 4 finished with value: 0.4993247628211975 and parameters: {'learning_rate': 0.18962833427410933, 'max_depth': 12, 'min_child_weight': 16.577824547537723, 'subsample': 0.8620900144280699, 'colsample_bytree': 0.6221725799648963, 'lambda': 1.378321708072992, 'alpha': 1.2735708456832626}. Best is trial 2 with value: 0.4786391258239746.


    [Outer 2017-01-01~2017-01-15] RMSE(log1p)=0.59748, best_round=2462
    Chosen FINAL_PARAMS: {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist', 'seed': 73, 'device': 'cuda', 'learning_rate': 0.028856378820378666, 'max_depth': 9, 'min_child_weight': 5.977459761804991, 'subsample': 0.9505358881481207, 'colsample_bytree': 0.9676301722988521, 'lambda': 0.20994580977352673, 'alpha': 1.9413622188311883}
    Chosen BEST_ROUND: 5997
    Saved -> /kaggle/working/submission_nested_xgb.csv


.44799

Reference:
mirinaepark/store-sales-time-series-forecasting -p /path/to/dest

