# Запуск Cleaner с лучшими подтверждёнными метриками

## Что выбрать

- Эталон лидерборда: submissions/161.csv, 89.16. Две гипотезы формы на объект, ручные поправки объёма и take на просмотренных кадрах. Это архив отправки для сравнения.
- Воспроизводимый автоматический финал с одной рамкой: /home/cleaner/CleanerVolume/submit/submission_physical_rank_v3.csv. Среднее по четырём группам object_id 87.47 (89.12 / 85.22 / 88.01 / 87.52), платформа 84.45. SHA-256: 5557b998ad4fe1e8442a8335aaa87836aa765b91a82c90c3d870246398b459a6.
- Новые варианты Claude не показали устойчивого улучшения над v3. Полное описание: reports/for_claude_RESULTS.md.

## Как запустить v3

Полная инструкция с размещением весов, кэшей и зависимостей: /home/cleaner/CleanerVolume/claude_eval/package/astramis_solution/README_REPRODUCE.md. Сборка использует Aion Lab для RGB-D и Mac Apple Silicon для финальных проходов MLX. В архиве указаны Python 3.12 на сервере, mlx-vlm 0.8.0 на Mac и Qwen3-VL-4B-Instruct-4bit.

На Mac после настройки путей выполнить из каталога, содержащего CleanerVolume:

```sh
HF_HUB_OFFLINE=1 "$MLX_PYTHON" CleanerVolume/experiments/form_volume_v2/build_final.py --ssh "$CLEANER_SSH" --model "$QWEN_MODEL" --dataset "$CLEANER_DATASET" --out submit/submission_form_volume_v2.csv
HF_HUB_OFFLINE=1 "$MLX_PYTHON" CleanerVolume/experiments/physical_v3/crop_review.py --model "$QWEN_MODEL" --input submit/submission_form_volume_v2.csv --frames "$CLEANER_DATASET/real/test" --out submit/submission_physical_rank_v3.csv
```

Для повтора на исходных 137 тестовых кадрах можно добавить --frozen-detections к первому вызову после восстановления снимков детектора из архива. Для новых кадров запускается детектор. Проверить колонки image_id,prediction, покрытие кадров, максимум одну рамку на кадр и десять полей каждой детекции. Метрики на размеченных кадрах измеряются официальным references/score.py с разбиением по object_id.

Команды сверены с серверным README. Новый прогон Mac/MLX здесь не выполнялся.
