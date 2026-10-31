from chronos import BaseChronosPipeline

print("Скачиваем и загружаем Chronos-2 Small...")

pipeline = BaseChronosPipeline.from_pretrained(
    "autogluon/chronos-2-small",
    device_map="cuda",
)

print("Модель успешно загружена!")