# YoloaSAM 開發與除錯完整記錄

## 1. 環境建置指令
- 建立專案目錄結構
  `mkdir -p YOLO_Inference_Platform/{config,weights,src/{detectors,depth_estimators,fusion,pipeline,visualization,utils},scripts,samples,output,logs}`
- 建立 micromamba 虛擬環境
  `micromamba create -y -n YoloaSAM python=3.10`
- 安裝套件（因避免衝突分三批安裝）
  `micromamba install -y -n YoloaSAM -c conda-forge numpy opencv matplotlib`
  `micromamba install -y -n YoloaSAM -c pytorch -c conda-forge pytorch torchvision cpuonly`
  `micromamba install -y -n YoloaSAM -c conda-forge transformers timm huggingface_hub ultralytics openvino`

## 2. 遇到的錯誤與解決方式
**錯誤 (1)：`AutoImageProcessor requires the Torchvision library but it was not found`**
- **情境**：在將 Depth Anything V2 轉換為 OpenVINO 格式時，`transformers` 拋出找不到 PyTorch / Torchvision 的錯誤。
- **原因**：Transformers 5.17.0 要求 PyTorch 必須為 `>= 2.5` 版本，但環境中安裝的穩定版 CPU PyTorch 為 2.4.0。因此 `transformers` 自動停用了 PyTorch 支援。
- **改善方法**：將 `transformers` 降版至 `<5.0` (具體為 4.57.6)，這與 PyTorch 2.4.0 完美相容。
- **解決指令**：`micromamba install -y -n YoloaSAM -c conda-forge 'transformers<5.0'`

**錯誤 (2)：`ModuleNotFoundError: No module named 'src'`**
- **情境**：在專案根目錄下執行 `python src/main.py --source samples/zidane.jpg --source_type image` 時發生。
- **原因**：Python 執行器無法在 `sys.path` 找到以 `src` 命名空間開頭的模組。
- **改善方法**：在執行時注入環境變數 `PYTHONPATH=.`，將當前目錄加入模組搜尋路徑。
- **解決指令**：`micromamba run -n YoloaSAM env PYTHONPATH=. python src/main.py --source samples/zidane.jpg --source_type image`

## 3. 程式架構與工廠方法 (Factory Method Pattern) 實作摘要
本專案採用工廠方法來解耦各個模組：
- **`DetectorFactory`**：用於建立 `BaseDetector` 的子類別 (如 `YoloDetector`)。
- **`DepthEstimatorFactory`**：用於建立 `BaseDepthEstimator` 的子類別 (如 `DepthAnythingOpenVINO`)。
- **`PipelineFactory`**：根據輸入類型建立 `ImagePipeline` 或 `VideoPipeline`。
模組化的好處是，未來如果要更換其他的深度估計模型（例如 MiDaS），只需要新增一個類別繼承 `BaseDepthEstimator`，並在 Factory 註冊即可，完全不影響主程式的邏輯。

## 4. 權重轉換與加速
- 使用腳本 `scripts/export_openvino.py`
- YOLO11s 透過 `model.export(format="openvino")` 匯出
- Depth Anything V2 透過 `ov.convert_model` 加上 dummy input 轉換
- 雙模型統一在 OpenVINO (CPU) 下執行推論，速度大幅提升。

## 5. 端到端推論測試
- 測試圖片：`samples/zidane.jpg`
- 成功輸出結果至 `output/result.jpg`，圖中成功融合了 2D 框以及從深度圖換算出的距離 (Dist) 資訊。
