# LightET-FusionNet 重現

[![Paper](https://img.shields.io/badge/paper-IEEE_Access-00629B)](https://doi.org/10.1109/ACCESS.2026.3660760)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ExtraTrees-F7931E?logo=scikitlearn&logoColor=white)
![Data](https://img.shields.io/badge/data-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![Reproduction](https://img.shields.io/badge/params-exact_match-2ea44f)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

重現的論文:
> Shahzad, Amjad, Mushtaq, Chughtai, "LightET-FusionNet as a Lightweight Deep
> Ensemble Model for Scalable Maize Disease Classification", IEEE Access,
> 2026. 沒有找到官方程式碼;這是從 Section II(公式 1-7、Figure 1-2)和
> Section II-E 獨立重新實作的 clean-room 版本。

## Pretrained Models

發布在 [Releases](https://github.com/Amelieee-m-c/Light-FusionNet/releases/tag/v1.0-official):

| 檔案 | 內容 | 備註 |
|---|---|---|
| `backbone_seed3_best.pt` | CNN backbone(EfficientNetV2B0+SE),輸出 128 維 embedding | seed 3,三個 seed 裡 ExtraTrees 表現最好(97.14%) |
| `scaler_seed3_best.joblib` | StandardScaler | 對應 seed 3 的 embedding 標準化參數 |
| `extratrees_seed3_best.joblib` | 訓練好的 ExtraTrees 分類器 | 200 棵樹,max_depth=20 |
| `backbone_seed1.pt` / `backbone_seed2.pt` | 另外兩個 seed 的 CNN backbone | 供對照,沒有附對應的 scaler/ExtraTrees |

完整 pipeline 要三個檔案(backbone + scaler + ExtraTrees)一起載入才能用,
只有 backbone 只能拿到 128 維 embedding,還不能直接分類:

```python
import torch
import joblib
from src.model import LightETBackbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightETBackbone(num_classes=4).to(device)
model.load_state_dict(torch.load("backbone_seed3_best.pt", map_location=device))
model.eval()

scaler = joblib.load("scaler_seed3_best.joblib")
clf = joblib.load("extratrees_seed3_best.joblib")

# x: 一個 batch 的影像 tensor,先做跟訓練時一樣的前處理(resize 256 + Normalize(0.5,0.5,0.5))
embedding = model.embed(x).cpu().numpy()      # (B, 128)
embedding_scaled = scaler.transform(embedding)

# clf.predict(...) 是 sklearn 內建的 soft voting(平均每棵樹的機率)。
# 論文描述的是 hard voting(每棵樹各投一票、多數決),對應寫法:
import numpy as np
tree_preds = np.stack([t.predict(embedding_scaled) for t in clf.estimators_])  # (200, B)
pred = np.array([np.bincount(tree_preds[:, i].astype(int)).argmax() for i in range(tree_preds.shape[1])])
```

## 重現結果

玉米葉,4 類,3-seed 平均值 ± 標準差:

| 分類器 | 論文 Accuracy | 重現 Accuracy |
|---|---|---|
| Gradient Boosting | 96.66% | 96.93 ± 0.40% |
| Logistic Regression | 96.82% | 96.71 ± 0.49% |
| SVM | 96.66% | 96.71 ± 0.27% |
| Voting Ensemble | 96.97% | 96.98 ± 0.34% |
| XGBoost | 96.66% | 96.50 ± 0.57% |
| LightGBM | 96.34% | 96.98 ± 0.13% |
| **ExtraTrees("LightET-FusionNet")** | **97.30%**(論文裡最好的) | 96.71 ± 0.27%(hard voting,這裡不是最好的) |

參數量幾乎完全對上:可訓練參數 6,629,652 + BatchNorm buffer 60,667 =
總共 6,690,319,對比論文宣稱的 6.62M + 63,168 = 6.69M——是這次所有重現
專案裡架構對得最準的一次。唯一沒有重現成功的地方:論文的核心主張是
ExtraTrees 明顯是最好的分類器(比第二名高 0.33 個百分點)。這裡它排在
中段,被 LightGBM 和 Voting Ensemble 超過。

**2026-08-13 更新**:原本 `max_depth` 用的是 sklearn 預設值 `None`,因為
論文對這個值做過 GridSearchCV(選項是 None 或 20),文字沒有明講最後選
哪個。後來確認論文實際用的是 **20**,改過去重跑——但結果**完全沒變**
(ExtraTrees 三個 seed 平均還是 96.77±0.33%,數字一模一樣)。原因是這個
資料集在不限制深度時,ExtraTrees 的樹本身就沒有長超過 20 層,所以
`max_depth=20` 對這個資料集來說是個沒有實際作用的限制。這排除了
「max_depth 猜錯」這個候選解釋。

**2026-08-18 更新**:論文原文其實還有一句更具體的話沒對齊——「The ET
classifier follows a **hard-voting** strategy in which all decision trees
contribute equally to the final prediction without assigning explicit
weights」。sklearn 的 `ExtraTreesClassifier.predict()` 預設其實是
**soft voting**(把每棵樹預測的機率平均、再取最大值),不是字面上「每棵
樹各投一票、多數決」的 hard voting——這是 sklearn 一個容易被忽略的實作
細節,我們之前一直用預設的 `.predict()`,沒有特別處理。改成真正的
hard voting(拿 `clf.estimators_` 裡每棵樹各自的 `.predict()` 結果做多數決,
見 `train_phase2.py::predict_hard_voting`)重跑後:**96.71 ± 0.27%**,
跟 soft voting 的 96.77 ± 0.33% 幾乎一樣(甚至還低了 0.06pp)。

`max_depth` 和 hard voting 這兩個從論文文字挖出來的具體細節,都測過了,
**都對結果沒有實質影響**——ExtraTrees 沒有重現出論文「明顯最佳」這個
結果的真正原因,還是沒找到,可能藏在論文 GridSearchCV 調的其他參數、
或是訓練資料切分/前處理上還沒對齊的細節裡,也可能單純是論文那次實驗的
隨機性。

## 範圍

重現論文的**主要架構跟核心結果**:EfficientNetV2B0 + 自訂 SE block
(phase 1,端對端微調)-> 128 維 embedding -> ExtraTrees 集成分類器
(phase 2)= "LightET-FusionNet",跟論文 Table 4 裡其他 6 種傳統分類器
一起比較,照論文協定跑 3 個獨立 seed 取平均。

**不在範圍內**:Table 2/3 那個「5 種 backbone × 有無 SE」的消融實驗網格
(VGG16、ShuffleNetV2、MobileNetV2、DenseNet121、EfficientNetV2B0,各跑
兩次)——這裡沒有額外做這 10 次完整訓練,直接採用論文自己的結論
(EfficientNetV2B0+SE 是最好的 backbone),沒有重新推導一次。

## 資料集

跟 `DenseViT_repro` 已經下載好的「Corn or Maize Leaf Disease Dataset」
是同一個(`data/raw/corn`)——這篇論文引用的是不同的 Kaggle 鏡像
(`sahityamamillapalli/...`),DenseViT_repro 用的是
(`smaranjitghose/...`),但兩者是同一份 4188 張圖、4 類資料集的不同上傳
版本(確認過:檔案大小完全一樣,每個類別的數量也跟論文寫的完全對上:
Blight 1146、Common Rust 1306、Gray Leaf Spot 574、Healthy 1162)。不需要
重新下載。

切分方式:70/15/15 分層 train/val/test(`data_prep/make_split.py`),完全
對應論文協定,每個 seed(1、2、3)都重新切一次。

## 架構(`src/model.py`)

透過 `timm` 使用 `tf_efficientnetv2_b0`——直接對應論文實際使用的
Keras/TF 權重(不是 torchvision 尺寸不同的 `efficientnet_v2_s/m/l`,那些
沒有對應 "B0" 規模的版本)-> GAP(1280 維,內建在 timm 的
`num_classes=0` pooling 裡)-> 自訂 SE block(對 pooled 向量做 channel
gating,reduction 1:8)-> Dense(256, ReLU) -> Dropout(0.3) ->
Dense(128, ReLU) -> [只有 phase-1 訓練時才有] Dense(num_classes)。

**參數量:可訓練 6,629,652 + buffer(BatchNorm 統計量)60,667 = 總共
6,690,319**,對比論文宣稱的可訓練 6.62M + non-trainable 63,168 = 總共
6.69M——幾乎完全一致,是目前這一系列重現裡對得最準的一次。

### 已解決的論文矛盾之處(權威版本見 `model.py` 的 docstring)

1. **SE block 作用對象的矛盾**:文字說 SE 是作用在 GAP 之後的 pooled
   向量上;但公式 (4) 的寫法卻暗示是作用在 GAP 之前的空間特徵圖上。這裡
   採用文字描述 + Figure 2(圖裡 SE 明確畫在 "Avg Pooling" 之後):SE 是
   對 pooled 的 1280 維向量做 gating。
2. **特徵萃取層的矛盾**:文字說是「倒數第三層 dense layer」(照這樣算
   應該是 Dropout 層,對應 256 維);但 Figure 2 明確標示 ExtraTrees 的
   輸入是「128 Features」。這裡相信 Figure 2 給出的具體數字——特徵是從
   Dense(128) 那一層取出。
3. Dense(128) 的活化函數沒有寫;這裡用 ReLU(跟 Dense(256) 一致)。

## 訓練(`src/train_phase1.py`、`src/train_phase2.py`、`src/run_all_seeds.py`)

Phase 1 完全照 Section II-E:Adam lr=1e-4,label smoothing 0.1,最多
50 epoch,early stopping 監控**驗證集 accuracy**(不是 loss),patience=10,
用 sklearn 算出的 class weight,還原最佳驗證表現的權重。前處理:resize
到 256×256,「EfficientNetV2 preprocessing」實作成
Normalize(mean=0.5, std=0.5)(對應到 [-1,1],對應 Keras 的
`efficientnet_v2.preprocess_input`)。

Phase 2:萃取 128 維 embedding,做 StandardScaler,接著訓練 ExtraTrees
(200 estimators,max_features='sqrt',class_weight='balanced'——論文對
max_depth 做了「None 或 20」的 GridSearchCV,文字沒有明確給出最終選哪個,
這裡用 `None`),以及 GradientBoosting/LogisticRegression/SVM/XGBoost/
LightGBM/VotingEnsemble,做完整對應 Table 4 的比較。分類器只用 train
切分的特徵去 fit(論文沒說 val 的特徵是否也一起用來訓練分類器;這裡讓
val 只用在 phase-1 的 early stopping,不進入 phase-2 的訓練)。

`run_all_seeds.py` 會針對 seed {1,2,3} 跑完整個 pipeline,並把每個分類器
的結果算成 mean ± std,彙整到 `runs/summary_3seeds.json`,對應論文
Table 2-4 的報告格式。

## 環境需求

Python 3.10+、PyTorch(建議 CUDA 版)、`timm`、scikit-learn、xgboost、
lightgbm、matplotlib。

## 授權

本 repo 程式碼採用 MIT 授權。不包含任何資料集圖片——Kaggle 來源見上方
「資料集」段落,有其原始授權。
