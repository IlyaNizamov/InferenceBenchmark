# LLM Inference Benchmark

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyQt6-GUI-green?style=for-the-badge&logo=qt&logoColor=white" alt="PyQt6"/>
  <img src="https://img.shields.io/badge/LangChain-Framework-orange?style=for-the-badge" alt="LangChain"/>
  <img src="https://img.shields.io/badge/License-Unlicense-brightgreen?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <b>Инструмент для бенчмаркинга производительности LLM моделей на различных движках инференса</b>
</p>

---

## Описание

**LLM Inference Benchmark** — это приложение для тестирования производительности языковых моделей (LLM) на различных движках инференса. Позволяет измерить скорость генерации токенов в последовательном и параллельном режимах.

### Ключевые возможности

- **Поддержка движков инференса**: vLLM, Ollama, llama.cpp
- **Два режима тестирования**: последовательный и параллельный
- **Structured Output**: тестирование с SO и без него
- **GUI интерфейс**: удобное приложение на PyQt6 с темой Dracula
- **Экспорт результатов**: сохранение в CSV и отправка на API
- **Реалистичные тесты**: 3 бизнес-сценария на русском языке

---

## Тестовые сценарии

Бенчмарк использует набор из 3 бизнес-сценариев, которые чередуются по кругу:

| Сценарий | Описание |
|----------|----------|
| **Перевод** | Перевод делового письма на английский язык |
| **Саммари** | Составление краткого содержания с выделением тезисов |
| **Ответное письмо** | Генерация ответа с уточняющими вопросами |

Сценарии определены в `src/prompts.py` и используют Pydantic-схемы для Structured Output.

---

## Метрики

| Метрика | Описание |
|---------|----------|
| **Последовательная скорость** | Токены/сек при одиночных запросах |
| **Параллельная скорость** | Средняя скорость при параллельных запросах |
| **Пропускная способность** | Общее количество токенов/сек при нагрузке |

Все метрики измеряются отдельно для режимов с Structured Output (SO) и без него.

---

## Установка

### Требования

- Python 3.12+
- Работающий сервер инференса (vLLM, Ollama или llama.cpp)

### Клонирование репозитория

```bash
git clone https://github.com/IlyaNizamov/InferenceBenchmark.git
cd InferenceBenchmark
```

### Создание виртуального окружения

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Установка зависимостей

```bash
pip install -r requirements.txt
```

Или установка основных пакетов вручную:
```bash
pip install langchain-ollama langchain-openai PyQt6
```

---

## Запуск

### GUI-приложение

```bash
python src/benchmark_gui.py
```

### Сборка исполняемого файла

Для создания standalone-приложения:

```bash
pip install -U pyinstaller

pyinstaller src/benchmark_gui.py --onefile --windowed --noconsole --hidden-import=langchain_openai --hidden-import=langchain_ollama --hidden-import=langchain --hidden-import=langchain_community --workpath build
```
### Готовая сборка

Можете скачать скомпилированную версию [здесь](dist/benchmark_gui.exe)

---

## Использование

### Вкладка «Справочники»

Чтобы можно было удобно фильтровать результаты тестов на сайте, часть данных идет в виде справочников с портала.
Вы можете добавить свои данные через эту вкладку.

API публичное и не требует авторизации, поэтому просьба внимательно относиться к данным! Не создавать дублей и не спамить.  

Управление справочными данными:
- Модели LLM
- Версии движков инференса
- Модели GPU

### Вкладка «Параметры»

1. Выберите **движок инференса** (vLLM, Ollama, llama.cpp)
2. Укажите **модель** и **URL API сервера**
3. Настройте **GPU конфигурацию**
4. Задайте количество **параллельных запросов** и **замеров**
5. Нажмите **«Запустить бенчмарк»**

Желательно заполнить поля **"Параметры запуска"** и **"Описание"**, это очень поможет другим людям воспроизвести тест если вы его опубликуете.

Пример:

"Параметры запуска" : "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-4B-Instruct-2507 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 16384"

"Описание": "Ryzen 5 3600, 62Gb G.Skill 3600, PCIe x16"

### Вкладка «Результаты»

После завершения теста отображаются все метрики. Результаты можно:
- Сохранить в CSV-файл
- Если хотите поделиться результатом теста, то можете отправить по API

### Вкладка «История»

Просмотр всех предыдущих запусков из локального CSV-файла.

---

## Структура проекта

```
InferenceBenchmark/
├── src/
│   ├── benchmark.py        # Основной класс LLMBenchmark
│   ├── benchmark_gui.py    # GUI-приложение на PyQt6
│   ├── prompts.py          # Тестовые сценарии и SO-схемы
│   └── logger_config.py    # Конфигурация логирования
├── requirements.txt        # Зависимости Python
├── LICENSE                 # Лицензия Unlicense
└── readme.md               # Документация
```

---

## Ссылки

- **Telegram-канал**: [https://t.me/+mXaCILG8i6czOTY6](https://t.me/+mXaCILG8i6czOTY6)
- **Результаты тестов**: [https://nizamov.school/benchmarks](https://nizamov.school/benchmarks)

---

## Лицензия

**Unlicense** — это общественное достояние. Вы можете свободно копировать, изменять, распространять и использовать код в любых целях, включая коммерческие, без каких-либо ограничений и указания авторства.
