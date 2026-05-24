# DigitalEpoch: HOLD Cloud Detection

Проект для подготовки синтетического датасета и обучения YOLO-модели, которая находит инженерные пометки-«облачка» на PDF-чертежах.

В репозиторий нельзя загружать документы заказчика, PDF-чертежи, датасеты, архивы, результаты обучения и веса моделей. В GitHub должны храниться только код, инструкции и конфигурационные файлы.

## Структура проекта

```text
DigitalEpoch/
  src/
    generate_dataset.py      # генератор синтетического YOLO-датасета
    visualize_labels.py      # визуальная проверка YOLO-разметки
    infer_pdf_folder.py      # инференс модели по папке PDF
  notebooks/
    train_yolo_colab.ipynb   # ноутбук обучения в Google Colab
  input_pdfs/
    .gitkeep                 # локальная папка для PDF, содержимое не коммитится
  output/
    .gitkeep                 # локальная папка для результатов, содержимое не коммитится
  README.md
  requirements.txt
  .gitignore
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## 1. Генерация датасета

На вход подаётся папка с публичными PDF или изображениями без уже существующих облачков.

```bash
python src/generate_dataset.py ^
  --input-dir "./input_pdfs" ^
  --output-dir "./hold_dataset" ^
  --samples 500 ^
  --min-clouds 0 ^
  --max-clouds 5 ^
  --hold-font-size 26 ^
  --preview ^
  --clean-output
```

Результат:

```text
hold_dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
  previews/
  data.yaml
  metadata.jsonl
```

Особенности генератора:

- на одном изображении может быть от 0 до N облачков;
- облачка отличаются размером, толщиной линии, размером дуг и случайными искажениями формы;
- размер дуги не жёстко привязан к размеру облачка;
- слово `HOLD` имеет фиксированный размер и не масштабируется вместе с облачком;
- `HOLD` может находиться внутри, рядом или отсутствовать;
- bbox в YOLO-разметке соответствует облачку, а не тексту.

## 2. Проверка разметки без labelImg

Скрипт создаёт отдельную папку с preview-картинками, где поверх изображений нарисованы bbox из `.txt`-разметки.

```bash
python src/visualize_labels.py ^
  --dataset-dir "./hold_dataset" ^
  --output-dir "./output/label_previews" ^
  --split all
```

Это позволяет быстро пролистать разметку обычным просмотрщиком изображений.

## 3. Обучение YOLO в Google Colab

Для первого запуска удобнее локально сгенерировать датасет, заархивировать его и загрузить архив в Google Drive.

```powershell
Compress-Archive -Path .\hold_dataset\* -DestinationPath .\hold_dataset.zip -Force
```

Далее архив используется в Colab-ноутбуке. Для первого эксперимента рекомендуется лёгкая модель `yolov8n.pt`, например:

```bash
yolo detect train model=yolov8n.pt data=/content/hold_dataset/data.yaml epochs=20 imgsz=640 batch=4
```

Если ресурсов хватает, можно увеличить `imgsz` до 1024 и число эпох.

## 4. Инференс на папке PDF

Скрипт принимает папку с PDF-документами, в том числе многостраничными, рендерит каждую страницу и вызывает обученную модель.

Визуализация сохраняется без подписей классов и confidence, только с рамками найденных объектов.

```bash
python src/infer_pdf_folder.py ^
  --pdf-dir "./input_pdfs" ^
  --model "./weights/best.pt" ^
  --output-dir "./output/inference" ^
  --dpi 160 ^
  --imgsz 1024 ^
  --conf 0.25
```

Результат:

```text
output/inference/
  visualized/          # страницы PDF с найденными bbox
  detections.csv       # координаты найденных объектов
  detections.json      # координаты найденных объектов
```

## Что не загружать в GitHub

- PDF-документы;
- документы заказчика;
- публичные чертежи, использованные как фон;
- датасеты `hold_dataset/`;
- zip-архивы;
- веса моделей `.pt`;
- папки `runs/`, `output/`, `weights/`;
- изображения `.jpg/.png`.
