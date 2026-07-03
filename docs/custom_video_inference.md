# Custom Video Inference (Sparse4D)

這段流程將自訂格式影片（如 TLV / YUV）轉成 Sparse4D 可吃的
`nuscenes_*_infos_*.pkl`，再走既有 `tools/test.py` 做推論與產圖。

## 一鍵腳本（2.86）

```bash
bash tools/run_custom_video_inference_2_86.sh \
  /path/to/your/video.tlv \
  /path/to/checkpoint.ckpt \
  --plugin /path/to/plugin.dll \
  --param /path/to/param.yaml \
  --camera-map "CAM_FRONT:0,CAM_FRONT_RIGHT:1,CAM_FRONT_LEFT:2,CAM_BACK:3,CAM_BACK_LEFT:4,CAM_BACK_RIGHT:5"
```

## 參數

- `--plugin`：TLV/YUV decoder plugin（如 `.dll/.so`）
- `--param`：對應 decoder 參數檔
- `--start` / `--end`：frame 讀取區間（`end` 不含）
- `--num-cams`：預期攝影機數，預設 6
- `--camera-map`：`name:index` 對照，例如 `CAM_FRONT:0`
- `--intrinsics`：可選，3x3 內參 json（不給就用 identity）
- `--disable-eval`：建議做影片結果先關閉 evaluate（若你沒有 GT）
- `--no-inference`：只產生 info pkl，不做推論

## 產物預設路徑

- `--ann-file`：`outputs/custom_video/custom_video_infos_test.pkl`
- `--images-root`：`outputs/custom_video/images`
- `--out`：`outputs/custom_video/results.pkl`
- `--show-dir`：`outputs/custom_video/visual`
- `--video-out`：`outputs/custom_video/visual/video.avi`

## 補充

- `ototester` 還有提供無 plugin 的 `DataGetter` 路徑作 fallback；若 plugin/param 都沒設，會嘗試 `DataGetter`，最後再 fallback OpenCV。
- 目前版的姿態外參（ego / sensor）若無法從 header 取到會用 identity 與 odom 推估，建議後續再換成你的實際外參檔。
