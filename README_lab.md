# Лабораторная: SR-inpainting в SUPIR

## Цель и задание
- Разобраться в устройстве диффузионных image-to-image моделей на примере SUPIR (https://github.com/Fanghua-Yu/SUPIR).
- Изменить процесс сэмплинга: улучшать только область, заданную маской, остальное оставлять как есть (SR-inpainting в латенте).
- Протестировать разные маски (вся 1, вся 0, комбинированные формы) на нескольких изображениях с деградациями (DIV2K bicubic x4 и др.).
- Оценить PSNR, LPIPS, CLIP-IQA; собрать CSV и графики.
- Сделать отчёт (этот README) и выложить код.

## Что сделано
- Масочный бленд в латентном пространстве на каждом шаге диффузии: внутри маски — предсказанный SR, вне маски — исходный латент LQ.
- Поддержка пакетной прогонки по всем маскам для каждого входного изображения; сохранение с суффиксом маски.
- Метрики и графики: PSNR/LPIPS (CLIP-IQA при наличии clipiqa-pytorch), бар-плоты по маскам.
- Обработан ресайз HR к размеру SR при расхождении и защита от разных spatial shape при tile VAE.

## Формула бленда
$\hat{x}_0 = M \odot x_0^{\text{SR}} + (1 - M) \odot x_0^{\text{LQ}}$, где $M$ — маска, $x_0^{\text{SR}}$  предсказанный латент, $x_0^{\text{LQ}}$ латент исходного LQ (или HR, если берём эталон). Маска и HR приводятся к размеру латента.

## Основные изменения в коде
- Бленд в денойзере: [SUPIR/models/SUPIR_model.py](SUPIR/models/SUPIR_model.py).
- Прогон по всем маскам и суффиксы имён: [test.py](test.py).
- Метрики, разбор имён, плоты: [tools/evaluate_metrics.py](tools/evaluate_metrics.py).

## Маски
- Расположение: [tools/masks_640](tools/masks_640).
- Варианты: circle_center, diag_band, square_center, star_center (а также можно добавить полностью 1 или 0).

## Команды запуска
- Инференс (пример, 5 изображений, все маски):
  ```bash
  conda run -n SUPIR python test.py \
    --no_llava \
    --img_dir D:/cv_diff_lab/data/div2k_lr \
    --hr_dir D:/cv_diff_lab/data/div2k_hr \
    --mask_dir D:/cv_diff_lab/SUPIR/tools/masks_640 \
    --save_dir D:/cv_diff_lab/SUPIR/images_out_div2k_masks_fixed \
    --min_size 640 \
    --use_tile_vae --encoder_tile_size 256 --decoder_tile_size 64 \
    --loading_half_params --edm_steps 40 --max_images 5
  ```

- Метрики и плоты (нужны matplotlib, clipiqa-pytorch для CLIP-IQA):
  ```bash
  conda run -n SUPIR python tools/evaluate_metrics.py \
    --sr_dir D:/cv_diff_lab/SUPIR/images_out_div2k_masks_fixed \
    --hr_dir D:/cv_diff_lab/data/div2k_hr \
    --out_dir D:/cv_diff_lab/SUPIR/metrics_diffmask \
    --out_csv metrics_div2k.csv \
    --plots_prefix metrics_div2k \
    --device cuda
  ```

## Результаты
- CSV: [metrics_diffmask/metrics_div2k.csv](metrics_diffmask/metrics_div2k.csv)
- Плоты: [metrics_diffmask/metrics_div2k_psnr.png](metrics_diffmask/metrics_div2k_psnr.png), [metrics_diffmask/metrics_div2k_lpips.png](metrics_diffmask/metrics_div2k_lpips.png)
- CLIP-IQA появится в CSV/графике, если установлен clipiqa-pytorch (иначе столбец пуст и график не строится).

### Пример маски и результата
- Маска (circle_center): [tools/masks_640/mask_circle_center.png](tools/masks_640/mask_circle_center.png)
- Вывод с этой маской: [images_out_div2k_masks_fixed/0000_mask_circle_center_0.png](images_out_div2k_masks_fixed/0000_mask_circle_center_0.png)

## Данные
- Использованы LR/HR пары DIV2K bicubic x4 (LR: `data/div2k_lr`, HR: `data/div2k_hr`).

## Примечания
- HR ищется по имени SR без суффиксов `_mask..._idx` (например, `0000_mask_star_center_0.png` → `0000.png`).
- При предупреждении OpenMP о дублировании рантайма можно на время запуска метрик задать переменную окружения `KMP_DUPLICATE_LIB_OK=TRUE`.
- Для CLIP-IQA и плотов установить: `pip install clipiqa-pytorch matplotlib` в окружении SUPIR.
