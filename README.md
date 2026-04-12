# tinygpt

В репозитории есть четыре готовых run-скрипта под сценарии, которые вы описали:

1. `runs/from_scratch.sh`
   Полный pipeline обучения: tokenizer, pretrain, SFT, затем eval.

2. `runs/pretrain_with_nanochat_d32.sh`
   `nanochat-d32` tokenizer плюс pretrain, без обучения собственного tokenizer.

3. `runs/distill_from_nanochat_d32.sh`
   Дистилляция из `karpathy/nanochat-d32` в student, заранее обученный скриптом `runs/pretrain_with_nanochat_d32.sh`.
   Это важно: текущая online-KL дистилляция в `tinygpt` требует идентичный tokenizer у teacher и student.

4. `runs/smoke.sh`
   Локальный smoke-test для CPU или дешёвого GPU.

## Быстрый старт

Из корня `tinygpt`:

```bash
bash runs/from_scratch.sh
bash runs/pretrain_with_nanochat_d32.sh
bash runs/distill_from_nanochat_d32.sh
bash runs/smoke.sh
```

## Общие переменные окружения

Скрипты специально упрощены и почти всё держат прямо в командах.
Оставлены только несколько override-переменных:

- `WANDB_RUN` для имени run в Weights & Biases. По умолчанию используется `dummy`.
- `NPROC_PER_NODE` для числа GPU-процессов в GPU-скриптах.
- `DEVICE_TYPE` и `TEACHER_DEVICE` там, где это действительно нужно.

Примеры:

```bash
WANDB_RUN=my_student bash runs/pretrain_with_nanochat_d32.sh
WANDB_RUN=my_distill TEACHER_DEVICE=cpu bash runs/distill_from_nanochat_d32.sh
DEVICE_TYPE=cpu bash runs/smoke.sh
```

Артефакты и вспомогательные файлы сохраняются в `data/`, а не в `out/`.

## Примечания по budget-лейблам

`100$` и `1000$` в названиях скриптов — это удобные ориентиры по порядку бюджета, а не жёсткая гарантия цены.
Итоговая стоимость зависит от аренды GPU, числа GPU, длительности запуска и выбранных override-параметров.
