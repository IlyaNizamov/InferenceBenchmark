"""
GUI интерфейс для бенчмарка LLM моделей
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

import requests
from PyQt6.QtCore import QThread, pyqtSignal, QSettings, Qt
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QFormLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QComboBox, QSpinBox, QCheckBox, QMessageBox,
    QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QInputDialog, QSizePolicy, QScrollArea
)

from benchmark import LLMBenchmark, get_app_path
from logger_config import configure_root_logger

logger = logging.getLogger(__name__)


class ApiDataManager:
    """Класс для управления данными из API"""

    BASE_URL = "https://nizamov.school/inference_benchmark/"

    def __init__(self):
        self.models = []
        self.inference_types = []
        self.inference_versions = []
        self.gpu_models = []
        self.results = []

    def fetch_data(self):
        """Загружает все данные из API"""
        try:
            # Загрузка моделей
            self.models = self.fetch_endpoint("model-names/")
            # Загрузка типов инференса
            self.inference_types = self.fetch_endpoint("inference-types/")
            # Загрузка версий инференса
            self.inference_versions = self.fetch_endpoint("inference-versions/")
            # Загрузка моделей GPU
            self.gpu_models = self.fetch_endpoint("gpu-models/")
            # Загрузка результатов
            self.results = self.fetch_endpoint("results/")

            return True
        except Exception as e:
            logger.error("Ошибка при загрузке данных из API: %s", e)
            return False

    def fetch_endpoint(self, endpoint: str) -> List[Dict]:
        """Загружает данные из конкретного эндпоинта"""
        url = self.BASE_URL + endpoint
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error("Ошибка при запросе к %s: %s", url, e)
            return []

    def create_entry(self, endpoint: str, data: Dict) -> Optional[Dict]:
        """Создает новую запись в API"""
        url = self.BASE_URL + endpoint
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error("Ошибка при создании записи в %s: %s", url, e)
            return None


class BenchmarkWorker(QThread):
    """Поток для выполнения бенчмарка"""

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            logger.info("=" * 80)
            logger.info("ЗАПУСК БЕНЧМАРКА")
            logger.info("Модель: %s", self.config['model'])
            logger.info("Инференс: %s", self.config['inference'])
            logger.info("URL: %s", self.config['base_url'])
            logger.info("GPU: %s", self.config['gpu_models'])
            logger.info("=" * 80)

            # Создаем экземпляр бенчмарка
            benchmark = LLMBenchmark(
                inference=self.config['inference'],
                model=self.config['model'],
                base_url=self.config['base_url'],
                launch=self.config['launch'],
                gpu_count=self.config['gpu_count'],
                gpu_models=self.config['gpu_models'],
                api_key=self.config['api_key'],
                description=self.config['description'],
                parallel=self.config['parallel'],
                inference_version=self.config.get('inference_version'),
                is_docker=self.config.get('is_docker', False)
            )

            # Запускаем бенчмарк
            self.progress.emit("Запуск прогревочного запроса...")
            benchmark.warmup()

            self.progress.emit("Запуск последовательного теста (без SO)...")
            seq_no_so = benchmark.run_sequential(
                num_runs=self.config['runs'],
                use_structured_output=False
            )

            self.progress.emit("Запуск последовательного теста (с SO)...")
            seq_so = benchmark.run_sequential(
                num_runs=self.config['runs'],
                use_structured_output=True
            )

            self.progress.emit("Запуск параллельного теста (без SO)...")
            par_no_so_result = asyncio.run(benchmark.run_parallel(
                num_parallel=self.config['parallel'],
                use_structured_output=False
            ))

            self.progress.emit("Запуск параллельного теста (с SO)...")
            par_so_result = asyncio.run(benchmark.run_parallel(
                num_parallel=self.config['parallel'],
                use_structured_output=True
            ))

            # Формируем результаты
            results = {
                'seq_no_so': seq_no_so['avg_speed'],
                'seq_so': seq_so['avg_speed'],
                'par_no_so': par_no_so_result['avg_speed'],
                'par_so': par_so_result['avg_speed'],
                'throughput_no_so': par_no_so_result['throughput'],
                'throughput_so': par_so_result['throughput'],
            }

            logger.info("=" * 80)
            logger.info("БЕНЧМАРК ЗАВЕРШЁН УСПЕШНО")
            logger.info("=" * 80)

            self.finished.emit(results)

        except Exception as e:
            logger.error("=" * 80)
            logger.error("ОШИБКА ПРИ ВЫПОЛНЕНИИ БЕНЧМАРКА")
            logger.error("Тип ошибки: %s", type(e).__name__)
            logger.error("Сообщение: %s", str(e))
            logger.error("-" * 80)
            import traceback
            logger.error(traceback.format_exc())
            logger.error("=" * 80)
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Основное окно приложения"""

    # Маппинг между отображаемыми именами инференса и именами в API
    INFERENCE_DISPLAY_TO_API = {
        "vLLM": "vLLM",
        "Ollama": "Ollama",
        "llama.cpp": "Llama.cpp"
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Benchmark GUI")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(900, 600)

        self.api_manager = ApiDataManager()
        self.current_results = None
        self.current_config = None
        self.has_unsaved_references = False
        self.pending_references = {
            'models': [],
            'inference_versions': [],
            'gpu_models': []
        }
        self.settings = QSettings("LLMBenchmark", "BenchmarkGUI")

        self.init_ui()
        self.load_api_data()
        self.load_settings()
        self.load_csv_history()

    def init_ui(self):
        """Инициализирует пользовательский интерфейс"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Создаем вкладки
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Вкладка параметров
        self.setup_params_tab()

        # Вкладка результатов
        self.setup_results_tab()

        # Вкладка истории
        self.setup_history_tab()

        # Вкладка справочников
        self.setup_references_tab()

        # Панель состояния
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Готово")

    def setup_params_tab(self):
        """Настройка вкладки параметров"""
        # Создаём область прокрутки
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Внутренний виджет для прокрутки
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)

        # Группа основных параметров
        params_group = QGroupBox("Параметры бенчмарка")
        params_form = QFormLayout(params_group)
        params_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Выбор движка инференса
        self.inference_combo = QComboBox()
        self.inference_combo.addItems(["vLLM", "Ollama", "llama.cpp"])
        self.inference_combo.setMinimumWidth(200)
        self.inference_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.inference_combo.currentTextChanged.connect(self.on_inference_changed)
        params_form.addRow("Движок инференса:", self.inference_combo)

        # Выбор версии инференса
        self.inference_version_combo = QComboBox()
        self.inference_version_combo.setEditable(True)
        self.inference_version_combo.setMinimumWidth(200)
        self.inference_version_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        params_form.addRow("Версия инференса:", self.inference_version_combo)

        # Выбор модели
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(200)
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        params_form.addRow("Модель:", self.model_combo)

        # URL API сервера
        self.base_url_edit = QLineEdit("http://localhost:8000/v1")
        self.base_url_edit.setMinimumWidth(200)
        params_form.addRow("URL API сервера:", self.base_url_edit)

        # Параметры запуска
        self.launch_edit = QLineEdit()
        self.launch_edit.setMinimumWidth(200)
        self.launch_edit.setPlaceholderText("Например: vllm serve model --tp 2 --gpu-memory-utilization 0.9")
        params_form.addRow("Параметры запуска:", self.launch_edit)

        # Таблица GPU
        self.gpu_table = QTableWidget()
        self.gpu_table.setColumnCount(3)
        self.gpu_table.setHorizontalHeaderLabels(["Модель GPU", "Количество", "Удалить"])
        self.gpu_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.gpu_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.gpu_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.gpu_table.setColumnWidth(1, 100)
        self.gpu_table.setColumnWidth(2, 80)
        self.gpu_table.verticalHeader().setVisible(False)
        self.gpu_table.setRowCount(0)
        self.gpu_table.setMinimumHeight(120)
        self.gpu_table.setMinimumWidth(400)

        params_form.addRow("GPU конфигурация:", self.gpu_table)

        # Кнопка добавления GPU
        self.add_gpu_btn = QPushButton("Добавить GPU")
        self.add_gpu_btn.clicked.connect(self.add_gpu_row)
        params_form.addRow("", self.add_gpu_btn)

        # Опциональные параметры
        optional_group = QGroupBox("Дополнительные параметры")
        optional_form = QFormLayout(optional_group)
        optional_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # API ключ
        self.api_key_edit = QLineEdit("local")
        self.api_key_edit.setMinimumWidth(200)
        optional_form.addRow("API ключ:", self.api_key_edit)

        # Описание
        self.description_edit = QLineEdit()
        self.description_edit.setMinimumWidth(200)
        optional_form.addRow("Описание:", self.description_edit)

        # Количество параллельных запросов
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 1000)
        self.parallel_spin.setValue(10)
        self.parallel_spin.setMinimumWidth(100)
        optional_form.addRow("Параллельных запросов:", self.parallel_spin)

        # Количество замеров
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(1, 100)
        self.runs_spin.setValue(3)
        self.runs_spin.setMinimumWidth(100)
        optional_form.addRow("Количество замеров:", self.runs_spin)

        # Запуск в Docker
        self.docker_checkbox = QCheckBox()
        optional_form.addRow("Запуск в Docker:", self.docker_checkbox)

        # Кнопка запуска
        self.start_button = QPushButton("Запустить бенчмарк")
        self.start_button.setObjectName("start_button")
        self.start_button.clicked.connect(self.start_benchmark)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Лог
        self.log_text = QTextEdit()
        self.log_text.setMinimumHeight(150)
        self.log_text.setReadOnly(True)

        params_layout.addWidget(params_group)
        params_layout.addWidget(optional_group)
        params_layout.addWidget(self.start_button)
        params_layout.addWidget(self.progress_bar)
        params_layout.addWidget(QLabel("Лог:"))
        params_layout.addWidget(self.log_text)
        params_layout.addStretch()

        # Помещаем виджет в область прокрутки
        scroll_area.setWidget(params_widget)

        self.tab_widget.addTab(scroll_area, "Параметры")

    def add_gpu_row(self, model_name="", count=1):
        """Добавляет новую строку в таблицу GPU"""
        row = self.gpu_table.rowCount()
        self.gpu_table.insertRow(row)

        # Добавляем комбобокс в первую ячейку
        combo = QComboBox()
        combo.setEditable(True)
        combo.setMinimumWidth(150)
        for gpu in self.api_manager.gpu_models:
            combo.addItem(gpu['name'], gpu['id'])

        # Устанавливаем значение, если передано
        if model_name:
            index = combo.findText(model_name)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.setCurrentText(model_name)

        self.gpu_table.setCellWidget(row, 0, combo)

        # Добавляем спинбокс во вторую ячейку
        spin = QSpinBox()
        spin.setRange(1, 10)
        spin.setValue(count)
        spin.setMinimumWidth(60)
        self.gpu_table.setCellWidget(row, 1, spin)

        # Добавляем кнопку удаления в третью ячейку
        delete_btn = QPushButton("✕")
        delete_btn.setFixedWidth(60)
        delete_btn.clicked.connect(lambda: self.remove_gpu_row(row))
        self.gpu_table.setCellWidget(row, 2, delete_btn)

        # Устанавливаем высоту строки
        self.gpu_table.setRowHeight(row, 40)

    def remove_gpu_row(self, row):
        """Удаляет строку из таблицы GPU"""
        self.gpu_table.removeRow(row)

        # Переподключаем все кнопки удаления, чтобы обновить индексы строк
        for i in range(self.gpu_table.rowCount()):
            delete_btn = self.gpu_table.cellWidget(i, 2)
            if delete_btn:
                delete_btn.clicked.disconnect()
                delete_btn.clicked.connect(lambda checked, row_idx=i: self.remove_gpu_row(row_idx))

    def setup_results_tab(self):
        """Настройка вкладки результатов"""
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Таблица результатов
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Параметр", "Значение", "Ед.изм.", "", "", "", "", ""
        ])

        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        results_layout.addWidget(QLabel("Результаты бенчмарка:"))
        results_layout.addWidget(self.results_table)

        # Кнопки для работы с результатами
        buttons_layout = QHBoxLayout()

        self.save_results_btn = QPushButton("Сохранить результаты в CSV")
        self.save_results_btn.clicked.connect(self.save_results_to_csv)
        buttons_layout.addWidget(self.save_results_btn)

        self.send_to_api_btn = QPushButton("Отправить результаты в API")
        self.send_to_api_btn.clicked.connect(self.send_results_to_api)
        self.send_to_api_btn.setEnabled(False)  # Активируется после завершения бенчмарка
        buttons_layout.addWidget(self.send_to_api_btn)

        results_layout.addLayout(buttons_layout)

        self.tab_widget.addTab(results_widget, "Результаты")

    def setup_history_tab(self):
        """Настройка вкладки истории"""
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)

        # Таблица истории результатов
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(14)
        self.history_table.setHorizontalHeaderLabels([
            "Дата и время",
            "Описание",
            "Модель",
            "Инференс",
            "Параметры запуска",
            "GPU кол-во",
            "GPU модели",
            "Параллельных",
            "Послед. без SO",
            "Послед. с SO",
            "Паралл. без SO",
            "Паралл. с SO",
            "Пропуск. без SO",
            "Пропуск. с SO"
        ])

        # Настройка режимов изменения размера колонок
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Дата
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Описание
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Модель
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Инференс
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)  # Параметры запуска
        for i in range(5, 14):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)  # Только чтение
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.verticalHeader().setVisible(False)

        history_layout.addWidget(QLabel("История запусков из CSV:"))
        history_layout.addWidget(self.history_table)

        # Кнопка обновления истории
        self.refresh_history_btn = QPushButton("Обновить историю")
        self.refresh_history_btn.clicked.connect(self.load_csv_history)
        history_layout.addWidget(self.refresh_history_btn)

        self.tab_widget.addTab(history_widget, "История")

    def setup_references_tab(self):
        """Настройка вкладки справочников"""
        references_widget = QWidget()
        references_layout = QVBoxLayout(references_widget)

        # Группа "Модели"
        models_group = QGroupBox("Модели")
        models_layout = QVBoxLayout(models_group)

        self.models_table = QTableWidget()
        self.models_table.setColumnCount(2)
        self.models_table.setHorizontalHeaderLabels(["ID", "Название модели"])
        self.models_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.models_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        models_layout.addWidget(self.models_table)

        models_buttons = QHBoxLayout()
        self.add_model_btn = QPushButton("Добавить модель")
        self.add_model_btn.clicked.connect(self.add_model)
        models_buttons.addWidget(self.add_model_btn)
        models_buttons.addStretch()
        models_layout.addLayout(models_buttons)

        references_layout.addWidget(models_group)

        # Группа "Версии инференса"
        versions_group = QGroupBox("Версии инференса")
        versions_layout = QVBoxLayout(versions_group)

        self.versions_table = QTableWidget()
        self.versions_table.setColumnCount(3)
        self.versions_table.setHorizontalHeaderLabels(["ID", "Инференс", "Версия"])
        self.versions_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.versions_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.versions_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.versions_table.verticalHeader().setVisible(False)
        self.versions_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        versions_layout.addWidget(self.versions_table)

        versions_buttons = QHBoxLayout()
        self.add_version_btn = QPushButton("Добавить версию")
        self.add_version_btn.clicked.connect(self.add_inference_version)
        versions_buttons.addWidget(self.add_version_btn)
        versions_buttons.addStretch()
        versions_layout.addLayout(versions_buttons)

        references_layout.addWidget(versions_group)

        # Группа "Модели GPU"
        gpu_models_group = QGroupBox("Модели GPU")
        gpu_models_layout = QVBoxLayout(gpu_models_group)

        self.gpu_models_table = QTableWidget()
        self.gpu_models_table.setColumnCount(2)
        self.gpu_models_table.setHorizontalHeaderLabels(["ID", "Название GPU"])
        self.gpu_models_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.gpu_models_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.gpu_models_table.verticalHeader().setVisible(False)
        self.gpu_models_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        gpu_models_layout.addWidget(self.gpu_models_table)

        gpu_buttons = QHBoxLayout()
        self.add_gpu_model_btn = QPushButton("Добавить GPU")
        self.add_gpu_model_btn.clicked.connect(self.add_gpu_model)
        gpu_buttons.addWidget(self.add_gpu_model_btn)
        gpu_buttons.addStretch()
        gpu_models_layout.addLayout(gpu_buttons)

        references_layout.addWidget(gpu_models_group)

        # Кнопка отправки изменений на портал
        self.send_references_btn = QPushButton("Отправить изменения на портал")
        self.send_references_btn.clicked.connect(self.send_references_to_api)
        self.send_references_btn.setEnabled(False)
        references_layout.addWidget(self.send_references_btn)

        self.tab_widget.addTab(references_widget, "Справочники")

        # Загружаем данные в таблицы
        self.load_references_data()

    def load_api_data(self):
        """Загружает данные из API"""
        success = self.api_manager.fetch_data()
        if success:
            # Заполняем комбобоксы
            self.update_combos()
            # Обновляем таблицы справочников
            self.load_references_data()
            self.status_bar.showMessage("Данные из API успешно загружены")
        else:
            self.status_bar.showMessage("Ошибка загрузки данных из API")
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить данные из API")

    def update_combos(self):
        """Обновляет содержимое комбобоксов"""
        # Обновляем модели
        logger.debug("update_combos: Загружаем модели из API")
        logger.debug("update_combos: Всего моделей: %d", len(self.api_manager.models))

        self.model_combo.clear()
        for idx, model in enumerate(self.api_manager.models):
            logger.debug("update_combos: Модель #%d: ID=%s, Name='%s'", idx, model['id'], model['name'])
            self.model_combo.addItem(model['name'], model['id'])

        logger.debug("update_combos: Модели загружены в комбобокс, всего элементов: %d", self.model_combo.count())

        # Обновляем версии инференса для текущего движка
        self.on_inference_changed(self.inference_combo.currentText())

        # Обновляем комбобоксы в таблице GPU
        for row in range(self.gpu_table.rowCount()):
            combo_widget = self.gpu_table.cellWidget(row, 0)
            if combo_widget and isinstance(combo_widget, QComboBox):
                # Сохраняем текущее значение
                current_value = combo_widget.currentText()

                # Очищаем и заполняем заново
                combo_widget.clear()
                for gpu in self.api_manager.gpu_models:
                    combo_widget.addItem(gpu['name'], gpu['id'])

                # Восстанавливаем предыдущее значение, если оно существует
                if current_value:
                    index = combo_widget.findText(current_value)
                    if index >= 0:
                        combo_widget.setCurrentIndex(index)
                # Если нет текущего значения, но есть доступные GPU, выбираем первый
                elif combo_widget.count() > 0:
                    combo_widget.setCurrentIndex(0)

    def on_inference_changed(self, inference_type):
        """Обработчик изменения типа инференса"""
        # Очищаем список версий (сбрасываем связанные данные)
        self.inference_version_combo.clear()

        # Находим ID выбранного типа инференса
        inference_api_name = self.INFERENCE_DISPLAY_TO_API.get(inference_type)
        inference_id = None

        if inference_api_name:
            for inf_type in self.api_manager.inference_types:
                if inf_type['name'] == inference_api_name:
                    inference_id = inf_type['id']
                    break

        # Если не нашли ID, пробуем найти по отображаемому имени
        if not inference_id:
            for inf_type in self.api_manager.inference_types:
                if inf_type['name'] == inference_type:
                    inference_id = inf_type['id']
                    break

        # Фильтруем версии по ID типа инференса
        if inference_id:
            filtered_versions = [
                v for v in self.api_manager.inference_versions
                if v.get('inference') == inference_id
            ]

            # Заполняем список версий
            for version in filtered_versions:
                self.inference_version_combo.addItem(version['version'], version.get('id'))

            logger.debug("Загружено %d версий для инференса '%s' (ID: %s)", len(filtered_versions), inference_type, inference_id)
        else:
            logger.warning("Не найден ID для типа инференса '%s'", inference_type)

    def start_benchmark(self):
        """Запускает бенчмарк"""
        # Собираем информацию о GPU
        gpu_models_names = []  # Имена для LLMBenchmark
        gpu_model_ids = []  # ID для отправки результатов
        total_gpu_count = 0

        for row in range(self.gpu_table.rowCount()):
            combo_widget = self.gpu_table.cellWidget(row, 0)
            spin_widget = self.gpu_table.cellWidget(row, 1)

            if combo_widget and spin_widget:
                model_name = combo_widget.currentText()
                model_id = combo_widget.currentData()  # ID хранится в userData
                count = spin_widget.value()

                # Добавляем модель GPU в списки в соответствии с количеством
                for _ in range(count):
                    gpu_models_names.append(model_name)
                    if model_id:
                        gpu_model_ids.append(model_id)

                total_gpu_count += count

        # Если нет GPU, добавляем одну заглушку
        if not gpu_models_names:
            gpu_models_names = ["Unknown GPU"]
            total_gpu_count = 1

        # Собираем конфигурацию
        inference_display = self.inference_combo.currentText()
        inference_api = self.INFERENCE_DISPLAY_TO_API.get(inference_display, inference_display.lower())

        # Находим ID типа инференса
        inference_type_id = None
        for inf_type in self.api_manager.inference_types:
            if inf_type['name'] == inference_api:
                inference_type_id = inf_type['id']
                break

        if not inference_type_id:
            QMessageBox.warning(self, "Ошибка", f"Тип инференса '{inference_display}' не найден в справочниках")
            return

        logger.debug("Выбран тип инференса: '%s' (API: '%s', ID: %s)", inference_display, inference_api, inference_type_id)

        # Находим ID модели
        model_name = self.model_combo.currentText()
        model_id_from_data = self.model_combo.currentData()  # ID хранится в userData

        logger.debug("Модель из комбобокса: '%s'", model_name)
        logger.debug("ID из currentData(): %s", model_id_from_data)
        logger.debug("Текущий индекс в комбобоксе: %d", self.model_combo.currentIndex())

        # Выводим все элементы комбобокса для отладки
        logger.debug("Содержимое комбобокса (%d элементов):", self.model_combo.count())
        for i in range(self.model_combo.count()):
            item_text = self.model_combo.itemText(i)
            item_data = self.model_combo.itemData(i)
            logger.debug("  [%d] Text='%s', Data=%s", i, item_text, item_data)

        # Проверяем, соответствует ли ID текущему тексту
        # (если модель введена вручную, currentData может вернуть старый ID)
        model_id = None
        if model_id_from_data:
            # Проверяем, что ID соответствует именно этому имени модели
            for model in self.api_manager.models:
                if model['id'] == model_id_from_data and model['name'] == model_name:
                    model_id = model_id_from_data
                    logger.debug("ID совпадает с текущим текстом")
                    break

        # Если ID не совпадает или не найден, ищем по имени
        if not model_id:
            logger.debug("ID не совпадает с текстом или не найден, ищем по имени в справочниках...")
            logger.debug("Доступные модели: %s", [(m['id'], m['name']) for m in self.api_manager.models])

            for model in self.api_manager.models:
                if model['name'] == model_name:
                    model_id = model['id']
                    logger.debug("Найдено точное совпадение по имени: ID=%s", model_id)
                    break

            if not model_id:
                logger.debug("Точное совпадение не найдено")

        if not model_id:
            QMessageBox.warning(
                self,
                "Ошибка",
                f"Модель '{model_name}' не найдена в справочниках.\n\n"
                "Добавьте её на вкладке 'Справочники' или выберите из списка."
            )
            return

        logger.info("Выбрана модель: '%s' (ID: %s)", model_name, model_id)

        # Получаем ID версии инференса (если выбрана)
        inference_version_name = self.inference_version_combo.currentText() or None
        inference_version_id = self.inference_version_combo.currentData() if inference_version_name else None

        if inference_version_name:
            logger.debug("Выбрана версия инференса: '%s' (ID: %s)", inference_version_name, inference_version_id)

        # Проверяем, что первая GPU найдена
        first_gpu_id = gpu_model_ids[0] if gpu_model_ids else None
        if not first_gpu_id:
            QMessageBox.warning(self, "Ошибка", "GPU модель не найдена в справочниках. Добавьте её на вкладке 'Справочники'")
            return

        logger.debug("Выбраны GPU: %s", list(zip(gpu_models_names, gpu_model_ids)))
        logger.debug("Первая GPU для API: '%s' (ID: %s)", gpu_models_names[0] if gpu_models_names else 'N/A', first_gpu_id)

        config = {
            'inference': inference_api,  # Используем API имя для совместимости с LLMBenchmark
            'inference_id': inference_type_id,  # ID для отправки результатов
            'model': model_name,
            'model_id': model_id,  # ID модели
            'base_url': self.base_url_edit.text(),
            'launch': self.launch_edit.text(),
            'gpu_count': total_gpu_count,
            'gpu_models': gpu_models_names,  # Имена для LLMBenchmark
            'gpu_model_ids': gpu_model_ids,  # ID для отправки результатов
            'first_gpu_id': first_gpu_id,  # ID первой GPU для API
            'api_key': self.api_key_edit.text(),
            'description': self.description_edit.text(),
            'parallel': self.parallel_spin.value(),
            'runs': self.runs_spin.value(),
            'inference_version': inference_version_name,
            'inference_version_id': inference_version_id,  # ID версии инференса
            'is_docker': self.docker_checkbox.isChecked()
        }

        # Проверяем обязательные поля
        if not config['model'] or not config['base_url']:
            QMessageBox.warning(self, "Ошибка", "Заполните обязательные поля: модель и URL API сервера")
            return

        # Сохраняем конфигурацию для дальнейшего использования
        self.current_config = config

        # Создаем поток для выполнения бенчмарка
        self.worker = BenchmarkWorker(config)
        self.worker.progress.connect(self.update_log)
        self.worker.finished.connect(self.on_benchmark_finished)
        self.worker.error.connect(self.on_benchmark_error)

        # Настраиваем интерфейс на время выполнения
        self.start_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Индикатор выполнения
        self.log_text.clear()

        # Запускаем поток
        self.worker.start()

    def update_log(self, message):
        """Обновляет лог выполнения"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def on_benchmark_finished(self, results):
        """Обработка завершения бенчмарка"""
        self.current_results = results

        # Обновляем таблицу результатов
        self.update_results_table(results)

        # Обновляем интерфейс
        self.start_button.setEnabled(True)
        self.send_to_api_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage("Бенчмарк завершен успешно")

        # Переключаемся на вкладку результатов
        self.tab_widget.setCurrentIndex(1)

        # Останавливаем поток
        self.worker.quit()
        self.worker.wait()

    def on_benchmark_error(self, error_msg):
        """Обработка ошибки бенчмарка"""
        logger.error("GUI: Получена ошибка из worker: %s", error_msg)

        self.start_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(f"Ошибка: {error_msg}")
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при выполнении бенчмарка:\n{error_msg}")

        # Останавливаем поток
        self.worker.quit()
        self.worker.wait()

    def update_results_table(self, results):
        """Обновляет таблицу результатов"""
        self.results_table.setRowCount(6)

        rows = [
            ("Последовательный без SO", f"{results['seq_no_so']:.2f}", "ток/сек"),
            ("Последовательный с SO", f"{results['seq_so']:.2f}", "ток/сек"),
            ("Параллельный без SO", f"{results['par_no_so']:.2f}", "ток/сек"),
            ("Параллельный с SO", f"{results['par_so']:.2f}", "ток/сек"),
            ("Пропускная без SO", f"{results['throughput_no_so']:.2f}", "ток/сек"),
            ("Пропускная с SO", f"{results['throughput_so']:.2f}", "ток/сек"),
        ]

        for i, (param, value, unit) in enumerate(rows):
            self.results_table.setItem(i, 0, QTableWidgetItem(param))
            self.results_table.setItem(i, 1, QTableWidgetItem(value))
            self.results_table.setItem(i, 2, QTableWidgetItem(unit))

    def save_results_to_csv(self):
        """Сохраняет результаты в CSV файл"""
        if not self.current_results or not self.current_config:
            QMessageBox.information(self, "Информация", "Нет результатов для сохранения")
            return

        try:
            # Создаем экземпляр бенчмарка с текущей конфигурацией
            benchmark = LLMBenchmark(
                inference=self.current_config['inference'],
                model=self.current_config['model'],
                base_url=self.current_config['base_url'],
                launch=self.current_config['launch'],
                gpu_count=self.current_config['gpu_count'],
                gpu_models=self.current_config['gpu_models'],
                api_key=self.current_config['api_key'],
                description=self.current_config['description'],
                parallel=self.current_config['parallel'],
                inference_version=self.current_config.get('inference_version'),
                is_docker=self.current_config.get('is_docker', False)
            )

            # Формируем результаты в нужном формате
            results = {
                'seq_no_so': self.current_results['seq_no_so'],
                'seq_so': self.current_results['seq_so'],
                'par_no_so': self.current_results['par_no_so'],
                'par_so': self.current_results['par_so'],
                'throughput_no_so': self.current_results['throughput_no_so'],
                'throughput_so': self.current_results['throughput_so'],
            }

            # Сохраняем в CSV
            benchmark._save_to_csv(results)

            csv_path = benchmark.CSV_FILE
            self.status_bar.showMessage(f"Результаты сохранены в {csv_path}")
            QMessageBox.information(self, "Сохранение", f"Результаты успешно сохранены в файл:\n{csv_path}")

            # Обновляем историю
            self.load_csv_history()

        except Exception as e:
            self.status_bar.showMessage(f"Ошибка сохранения: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результаты в CSV:\n{str(e)}")

    def send_results_to_api(self):
        """Отправляет результаты в API"""
        if not self.current_results or not self.current_config:
            QMessageBox.warning(self, "Ошибка", "Нет результатов для отправки")
            return

        # Проверяем наличие несохранённых изменений в справочниках
        if self.has_unsaved_references:
            QMessageBox.warning(
                self,
                "Ошибка",
                "В справочниках есть несохранённые изменения!\n\n"
                "Перейдите на вкладку 'Справочники' и отправьте изменения на портал "
                "перед отправкой результатов бенчмарка."
            )
            return

        try:
            self.status_bar.showMessage("Отправка результатов в API...")
            self.send_to_api_btn.setEnabled(False)

            # Получаем все ID из сохранённой конфигурации
            model_id = self.current_config.get('model_id')
            if not model_id:
                raise Exception("ID модели не найден в конфигурации. Пожалуйста, перезапустите бенчмарк.")

            inference_id = self.current_config.get('inference_id')
            if not inference_id:
                raise Exception("ID типа инференса не найден в конфигурации. Пожалуйста, перезапустите бенчмарк.")

            inference_version_id = self.current_config.get('inference_version_id')

            # Формируем gpu_configs из списка ID GPU
            gpu_model_ids = self.current_config.get('gpu_model_ids', [])
            if not gpu_model_ids:
                raise Exception("ID GPU не найдены в конфигурации. Пожалуйста, перезапустите бенчмарк.")

            # Группируем GPU по ID и считаем количество
            from collections import Counter
            gpu_counts = Counter(gpu_model_ids)

            # Формируем массив gpu_configs
            gpu_configs = [
                {"gpu_model": gpu_id, "count": count}
                for gpu_id, count in gpu_counts.items()
            ]

            logger.debug("GPU configs: %s", gpu_configs)

            # Формируем payload для отправки
            payload = {
                "description": self.current_config['description'][:255],
                "model_name": model_id,
                "inference": inference_id,
                "inference_version": inference_version_id,
                "launch_params": self.current_config['launch'],
                "parallel_count": self.current_config['parallel'],
                "gpu_configs": gpu_configs,
                "is_docker": self.current_config.get('is_docker', False),
                "sequential_no_so": round(self.current_results['seq_no_so'], 2),
                "sequential_with_so": round(self.current_results['seq_so'], 2),
                "parallel_no_so": round(self.current_results['par_no_so'], 2),
                "parallel_with_so": round(self.current_results['par_so'], 2),
                "throughput_no_so": round(self.current_results['throughput_no_so'], 2),
                "throughput_with_so": round(self.current_results['throughput_so'], 2),
            }

            logger.debug("Отправка результатов: %s", payload)

            # Отправляем результаты
            created_id = self.api_manager.create_entry("results/", payload)

            if created_id:
                logger.info("Результаты успешно отправлены с ID: %s", created_id)
                self.status_bar.showMessage("Результаты успешно отправлены в API")
                QMessageBox.information(self, "Успех", "Результаты успешно отправлены в API")
            else:
                raise Exception("API не вернул ID созданной записи")

        except Exception as e:
            logger.error("Ошибка отправки результатов: %s", e)
            import traceback
            logger.error(traceback.format_exc())

            self.status_bar.showMessage(f"Ошибка отправки: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось отправить результаты в API:\n{str(e)}")
        finally:
            self.send_to_api_btn.setEnabled(True)

    def save_settings(self):
        """Сохраняет настройки приложения"""
        # Сохраняем основные параметры
        self.settings.setValue("inference", self.inference_combo.currentText())
        self.settings.setValue("model", self.model_combo.currentText())
        self.settings.setValue("base_url", self.base_url_edit.text())
        self.settings.setValue("launch", self.launch_edit.text())
        self.settings.setValue("api_key", self.api_key_edit.text())
        self.settings.setValue("description", self.description_edit.text())
        self.settings.setValue("parallel", self.parallel_spin.value())
        self.settings.setValue("runs", self.runs_spin.value())
        self.settings.setValue("inference_version", self.inference_version_combo.currentText())
        self.settings.setValue("is_docker", self.docker_checkbox.isChecked())

        # Сохраняем данные таблицы GPU
        gpu_data = []
        for row in range(self.gpu_table.rowCount()):
            combo_widget = self.gpu_table.cellWidget(row, 0)
            spin_widget = self.gpu_table.cellWidget(row, 1)

            if combo_widget and spin_widget:
                gpu_data.append({
                    "model": combo_widget.currentText(),
                    "count": spin_widget.value()
                })

        self.settings.setValue("gpu_data", gpu_data)

    def load_settings(self):
        """Загружает настройки приложения"""
        # Загружаем основные параметры
        if self.settings.contains("inference"):
            inference = self.settings.value("inference")
            index = self.inference_combo.findText(inference)
            if index >= 0:
                self.inference_combo.setCurrentIndex(index)

        if self.settings.contains("model"):
            model = self.settings.value("model")
            # Ищем индекс модели в списке
            index = self.model_combo.findText(model)
            if index >= 0:
                # Устанавливаем индекс - это правильно выберет элемент с его ID
                self.model_combo.setCurrentIndex(index)
                logger.debug("load_settings: Восстановлена модель: '%s' (индекс: %d, ID: %s)", model, index, self.model_combo.itemData(index))
            else:
                # Если не найдено в списке, просто устанавливаем текст
                self.model_combo.setCurrentText(model)
                logger.debug("load_settings: Модель '%s' не найдена в списке, установлен текст вручную", model)

        if self.settings.contains("base_url"):
            self.base_url_edit.setText(self.settings.value("base_url"))

        if self.settings.contains("launch"):
            self.launch_edit.setText(self.settings.value("launch"))

        if self.settings.contains("api_key"):
            self.api_key_edit.setText(self.settings.value("api_key"))

        if self.settings.contains("description"):
            self.description_edit.setText(self.settings.value("description"))

        if self.settings.contains("parallel"):
            self.parallel_spin.setValue(int(self.settings.value("parallel")))

        if self.settings.contains("runs"):
            self.runs_spin.setValue(int(self.settings.value("runs")))

        if self.settings.contains("inference_version"):
            self.inference_version_combo.setCurrentText(self.settings.value("inference_version"))

        if self.settings.contains("is_docker"):
            self.docker_checkbox.setChecked(self.settings.value("is_docker", type=bool))

        # Загружаем данные таблицы GPU
        if self.settings.contains("gpu_data"):
            gpu_data = self.settings.value("gpu_data")
            if gpu_data:
                for gpu_entry in gpu_data:
                    self.add_gpu_row(gpu_entry["model"], gpu_entry["count"])

    def load_references_data(self):
        """Загружает данные справочников из API в таблицы"""
        # Загружаем модели
        self.models_table.setRowCount(0)
        for model in self.api_manager.models:
            row = self.models_table.rowCount()
            self.models_table.insertRow(row)
            self.models_table.setItem(row, 0, QTableWidgetItem(str(model['id'])))
            self.models_table.setItem(row, 1, QTableWidgetItem(model['name']))

        # Загружаем версии инференса
        self.versions_table.setRowCount(0)
        for version in self.api_manager.inference_versions:
            row = self.versions_table.rowCount()
            self.versions_table.insertRow(row)
            self.versions_table.setItem(row, 0, QTableWidgetItem(str(version['id'])))
            self.versions_table.setItem(row, 1, QTableWidgetItem(version.get('inference_display', '')))
            self.versions_table.setItem(row, 2, QTableWidgetItem(version['version']))

        # Загружаем модели GPU
        self.gpu_models_table.setRowCount(0)
        for gpu in self.api_manager.gpu_models:
            row = self.gpu_models_table.rowCount()
            self.gpu_models_table.insertRow(row)
            self.gpu_models_table.setItem(row, 0, QTableWidgetItem(str(gpu['id'])))
            self.gpu_models_table.setItem(row, 1, QTableWidgetItem(gpu['name']))

    def add_model(self):
        """Добавляет новую модель в локальный список"""
        model_name, ok = QInputDialog.getText(
            self, "Добавить модель", "Введите название модели:"
        )

        if ok and model_name.strip():
            model_name = model_name.strip()

            # Проверяем, не существует ли уже такая модель
            for i in range(self.models_table.rowCount()):
                if self.models_table.item(i, 1).text() == model_name:
                    QMessageBox.warning(self, "Ошибка", "Такая модель уже существует")
                    return

            # Добавляем в таблицу с временным ID
            row = self.models_table.rowCount()
            self.models_table.insertRow(row)
            self.models_table.setItem(row, 0, QTableWidgetItem("NEW"))
            self.models_table.setItem(row, 1, QTableWidgetItem(model_name))

            # Помечаем как новую запись чуть более ярким фоном
            highlight_color = QColor(98, 114, 164)  # #6272a4 из палитры Dracula
            self.models_table.item(row, 0).setBackground(highlight_color)
            self.models_table.item(row, 1).setBackground(highlight_color)

            # Добавляем в список ожидающих отправки
            self.pending_references['models'].append({'name': model_name})
            self.has_unsaved_references = True
            self.send_references_btn.setEnabled(True)

            self.status_bar.showMessage("Модель добавлена. Отправьте изменения на портал.")

    def add_inference_version(self):
        """Добавляет новую версию инференса в локальный список"""
        # Получаем список типов инференса из API
        if not self.api_manager.inference_types:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить типы инференса из API")
            return

        # Создаём список для отображения
        inference_types_display = [inf_type['name'] for inf_type in self.api_manager.inference_types]

        inference_type_name, ok = QInputDialog.getItem(
            self, "Добавить версию инференса", "Выберите тип инференса:", inference_types_display, 0, False
        )

        if not ok:
            return

        # Находим ID выбранного типа инференса
        inference_type_id = None
        for inf_type in self.api_manager.inference_types:
            if inf_type['name'] == inference_type_name:
                inference_type_id = inf_type['id']
                break

        if not inference_type_id:
            QMessageBox.warning(self, "Ошибка", "Не удалось найти ID типа инференса")
            return

        # Вводим версию
        version, ok = QInputDialog.getText(
            self, "Добавить версию инференса", f"Введите версию для {inference_type_name}:"
        )

        if ok and version.strip():
            version = version.strip()

            # Добавляем в таблицу с временным ID
            row = self.versions_table.rowCount()
            self.versions_table.insertRow(row)
            self.versions_table.setItem(row, 0, QTableWidgetItem("NEW"))
            self.versions_table.setItem(row, 1, QTableWidgetItem(inference_type_name))
            self.versions_table.setItem(row, 2, QTableWidgetItem(version))

            # Помечаем как новую запись чуть более ярким фоном
            highlight_color = QColor(98, 114, 164)  # #6272a4 из палитры Dracula
            self.versions_table.item(row, 0).setBackground(highlight_color)
            self.versions_table.item(row, 1).setBackground(highlight_color)
            self.versions_table.item(row, 2).setBackground(highlight_color)

            # Добавляем в список ожидающих отправки с ID типа инференса
            self.pending_references['inference_versions'].append({
                'inference_id': inference_type_id,
                'inference_name': inference_type_name,
                'version': version
            })
            self.has_unsaved_references = True
            self.send_references_btn.setEnabled(True)

            self.status_bar.showMessage("Версия инференса добавлена. Отправьте изменения на портал.")

    def add_gpu_model(self):
        """Добавляет новую модель GPU в локальный список"""
        gpu_name, ok = QInputDialog.getText(
            self, "Добавить GPU", "Введите название GPU (например, RTX 4090):"
        )

        if ok and gpu_name.strip():
            gpu_name = gpu_name.strip()

            # Проверяем, не существует ли уже такая GPU
            for i in range(self.gpu_models_table.rowCount()):
                if self.gpu_models_table.item(i, 1).text() == gpu_name:
                    QMessageBox.warning(self, "Ошибка", "Такая GPU уже существует")
                    return

            # Добавляем в таблицу с временным ID
            row = self.gpu_models_table.rowCount()
            self.gpu_models_table.insertRow(row)
            self.gpu_models_table.setItem(row, 0, QTableWidgetItem("NEW"))
            self.gpu_models_table.setItem(row, 1, QTableWidgetItem(gpu_name))

            # Помечаем как новую запись чуть более ярким фоном
            highlight_color = QColor(98, 114, 164)  # #6272a4 из палитры Dracula
            self.gpu_models_table.item(row, 0).setBackground(highlight_color)
            self.gpu_models_table.item(row, 1).setBackground(highlight_color)

            # Добавляем в список ожидающих отправки
            self.pending_references['gpu_models'].append({'name': gpu_name})
            self.has_unsaved_references = True
            self.send_references_btn.setEnabled(True)

            self.status_bar.showMessage("GPU добавлена. Отправьте изменения на портал.")

    def send_references_to_api(self):
        """Отправляет новые справочные данные на портал"""
        if not self.has_unsaved_references:
            QMessageBox.information(self, "Информация", "Нет изменений для отправки")
            return

        try:
            self.send_references_btn.setEnabled(False)
            self.status_bar.showMessage("Отправка изменений на портал...")
            success_count = 0
            error_count = 0

            # Отправляем новые модели
            for model_data in self.pending_references['models']:
                logger.debug("Отправка модели: %s", model_data)
                created_id = self.api_manager.create_entry("model-names/", model_data)
                if created_id:
                    logger.info("Модель '%s' создана с ID: %s", model_data['name'], created_id)
                    self.status_bar.showMessage(f"Модель '{model_data['name']}' создана")
                    success_count += 1
                else:
                    logger.error("Ошибка создания модели '%s'", model_data['name'])
                    error_count += 1

            # Отправляем новые версии инференса
            for version_data in self.pending_references['inference_versions']:
                # Используем сохранённый ID типа инференса
                inference_id = version_data['inference_id']
                inference_name = version_data['inference_name']

                data = {
                    'inference': inference_id,
                    'version': version_data['version']
                }
                logger.debug("Отправка версии инференса: %s для %s", data, inference_name)
                created_id = self.api_manager.create_entry("inference-versions/", data)
                if created_id:
                    logger.info("Версия инференса '%s' для %s создана с ID: %s", version_data['version'], inference_name, created_id)
                    self.status_bar.showMessage(f"Версия инференса '{version_data['version']}' создана")
                    success_count += 1
                else:
                    logger.error("Ошибка создания версии инференса '%s' для %s", version_data['version'], inference_name)
                    error_count += 1

            # Отправляем новые GPU модели
            for gpu_data in self.pending_references['gpu_models']:
                logger.debug("Отправка GPU: %s", gpu_data)
                created_id = self.api_manager.create_entry("gpu-models/", gpu_data)
                if created_id:
                    logger.info("GPU '%s' создана с ID: %s", gpu_data['name'], created_id)
                    self.status_bar.showMessage(f"GPU '{gpu_data['name']}' создана")
                    success_count += 1
                else:
                    logger.error("Ошибка создания GPU '%s'", gpu_data['name'])
                    error_count += 1

            # Если были ошибки, не очищаем pending_references
            if error_count == 0:
                # Очищаем список ожидающих только если все успешно
                self.pending_references = {
                    'models': [],
                    'inference_versions': [],
                    'gpu_models': []
                }
                self.has_unsaved_references = False
                self.send_references_btn.setEnabled(False)

                # Перезагружаем данные из API
                self.load_api_data()

                self.status_bar.showMessage(f"Все изменения успешно отправлены на портал ({success_count} записей)")
                QMessageBox.information(self, "Успех", f"Изменения успешно отправлены на портал\n\nСоздано записей: {success_count}")
            else:
                self.send_references_btn.setEnabled(True)
                self.status_bar.showMessage(f"Отправка завершена с ошибками: успешно {success_count}, ошибок {error_count}")
                QMessageBox.warning(
                    self, "Частичный успех",
                    f"Отправка завершена с ошибками:\n\nУспешно: {success_count}\nОшибок: {error_count}\n\n"
                    "Проверьте консоль для подробностей."
                )

        except Exception as e:
            logger.error("Исключение при отправке: %s", e)
            import traceback
            logger.error(traceback.format_exc())

            self.status_bar.showMessage(f"Ошибка отправки: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось отправить изменения на портал:\n{str(e)}")
            self.send_references_btn.setEnabled(True)

    def load_csv_history(self):
        """Загружает историю результатов из CSV файла"""
        import csv

        # Получаем путь к CSV файлу (используем тот же путь, что и в LLMBenchmark)
        csv_file = get_app_path() / "benchmark_results.csv"

        # Очищаем таблицу
        self.history_table.setRowCount(0)

        # Если файл не существует, это нормально при первом запуске
        if not csv_file.exists():
            self.status_bar.showMessage("История пуста. CSV файл будет создан после первого запуска бенчмарка.")
            return

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                rows = list(reader)

                # Проверяем, что файл не пустой и есть данные кроме заголовка
                if len(rows) <= 1:
                    self.status_bar.showMessage("История пуста")
                    return

                data_rows = rows[1:]  # Пропускаем заголовок

                # Фильтруем строки с недостаточным количеством колонок (минимум 13 для совместимости со старым форматом)
                valid_rows = [row for row in data_rows if len(row) >= 13]

                if not valid_rows:
                    self.status_bar.showMessage("В истории нет корректных записей")
                    return

                self.history_table.setRowCount(len(valid_rows))

                # Заполняем таблицу в обратном порядке (новые сверху)
                for i, row in enumerate(reversed(valid_rows)):
                    for j in range(self.history_table.columnCount()):
                        value = row[j] if j < len(row) else ""
                        item = QTableWidgetItem(str(value))
                        self.history_table.setItem(i, j, item)

                self.status_bar.showMessage(f"Загружено {len(valid_rows)} записей из истории")

        except Exception as e:
            # Выводим ошибку в консоль для отладки
            logger.error("Ошибка при загрузке истории: %s", e)
            import traceback
            logger.error(traceback.format_exc())

            self.status_bar.showMessage(f"Ошибка при загрузке истории")
            # Не показываем диалог с ошибкой, если это первый запуск

    def closeEvent(self, event):
        """Обработка события закрытия окна"""
        self.save_settings()
        event.accept()


def main():
    # Настраиваем логгер
    configure_root_logger(level=logging.INFO)

    # Настройки High DPI для корректного масштабирования на мониторах с 125%, 150% и т.д.
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # Устанавливаем шрифт
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    # Применяем тему Dracula
    dracula_stylesheet = """
    /* Основные цвета */
    QWidget {
        background-color: #282a36;
        color: #f8f8f2;
        font-size: 10pt;
    }

    /* Главное окно */
    QMainWindow {
        background-color: #282a36;
    }

    /* Вкладки */
    QTabWidget::pane {
        border: 1px solid #44475a;
        background-color: #282a36;
        border-radius: 4px;
    }

    QTabBar::tab {
        background-color: #44475a;
        color: #f8f8f2;
        padding: 8px 20px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }

    QTabBar::tab:selected {
        background-color: #bd93f9;
        color: #282a36;
        font-weight: bold;
    }

    QTabBar::tab:hover {
        background-color: #6272a4;
    }

    /* Группы */
    QGroupBox {
        border: 2px solid #44475a;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 15px;
        font-weight: bold;
        color: #bd93f9;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 10px;
        color: #bd93f9;
    }

    /* Метки */
    QLabel {
        color: #f8f8f2;
        background-color: transparent;
    }

    /* Поля ввода */
    QLineEdit {
        background-color: #44475a;
        color: #f8f8f2;
        border: 1px solid #6272a4;
        border-radius: 4px;
        padding: 6px;
        selection-background-color: #bd93f9;
    }

    QLineEdit:focus {
        border: 2px solid #bd93f9;
    }

    /* Текстовые области */
    QTextEdit {
        background-color: #44475a;
        color: #f8f8f2;
        border: 1px solid #6272a4;
        border-radius: 4px;
        padding: 6px;
        selection-background-color: #bd93f9;
    }

    QTextEdit:focus {
        border: 2px solid #bd93f9;
    }

    /* Комбобоксы */
    QComboBox {
        background-color: #44475a;
        color: #f8f8f2;
        border: 1px solid #6272a4;
        border-radius: 4px;
        padding: 6px;
        min-width: 100px;
    }

    QComboBox:hover {
        border: 1px solid #bd93f9;
    }

    QComboBox:focus {
        border: 2px solid #bd93f9;
    }

    QComboBox::drop-down {
        border: none;
        width: 25px;
        subcontrol-origin: padding;
        subcontrol-position: center right;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 7px solid #f8f8f2;
        width: 0px;
        height: 0px;
    }

    QComboBox QAbstractItemView {
        background-color: #44475a;
        color: #f8f8f2;
        selection-background-color: #bd93f9;
        selection-color: #282a36;
        border: 1px solid #6272a4;
        border-radius: 4px;
    }

    /* Спинбоксы */
    QSpinBox {
        background-color: #44475a;
        color: #f8f8f2;
        border: 1px solid #6272a4;
        border-radius: 4px;
        padding: 6px;
    }

    QSpinBox:focus {
        border: 2px solid #bd93f9;
    }

    QSpinBox::up-button, QSpinBox::down-button {
        background-color: #6272a4;
        border: none;
        width: 16px;
    }

    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
        background-color: #bd93f9;
    }

    QSpinBox::up-arrow {
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 4px solid #f8f8f2;
    }

    QSpinBox::down-arrow {
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid #f8f8f2;
    }

    /* Чекбоксы */
    QCheckBox {
        color: #f8f8f2;
        spacing: 8px;
    }

    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 2px solid #6272a4;
        border-radius: 3px;
        background-color: #44475a;
    }

    QCheckBox::indicator:hover {
        border: 2px solid #bd93f9;
    }

    QCheckBox::indicator:checked {
        background-color: #bd93f9;
        border: 2px solid #bd93f9;
    }

    QCheckBox::indicator:checked:hover {
        background-color: #ff79c6;
        border: 2px solid #ff79c6;
    }

    /* Кнопки */
    QPushButton {
        background-color: #6272a4;
        color: #f8f8f2;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }

    QPushButton:hover {
        background-color: #bd93f9;
        color: #282a36;
    }

    QPushButton:pressed {
        background-color: #ff79c6;
    }

    QPushButton:disabled {
        background-color: #44475a;
        color: #6272a4;
    }

    /* Главная кнопка запуска */
    QPushButton#start_button {
        background-color: #50fa7b;
        color: #282a36;
        padding: 12px;
        font-size: 11pt;
    }

    QPushButton#start_button:hover {
        background-color: #8be9fd;
    }

    QPushButton#start_button:pressed {
        background-color: #ffb86c;
    }

    /* Таблицы */
    QTableWidget {
        background-color: #282a36;
        alternate-background-color: #44475a;
        color: #f8f8f2;
        gridline-color: #44475a;
        border: 1px solid #6272a4;
        border-radius: 4px;
        selection-background-color: #bd93f9;
        selection-color: #282a36;
    }

    QTableWidget::item {
        padding: 5px;
    }

    QTableWidget::item:selected {
        background-color: #bd93f9;
        color: #282a36;
    }

    QHeaderView::section {
        background-color: #44475a;
        color: #bd93f9;
        padding: 8px;
        border: none;
        border-right: 1px solid #6272a4;
        border-bottom: 1px solid #6272a4;
        font-weight: bold;
    }

    QHeaderView::section:first {
        border-top-left-radius: 4px;
    }

    QHeaderView::section:last {
        border-top-right-radius: 4px;
        border-right: none;
    }

    /* Списки */
    QListWidget {
        background-color: #282a36;
        color: #f8f8f2;
        border: 1px solid #6272a4;
        border-radius: 4px;
        selection-background-color: #bd93f9;
        selection-color: #282a36;
    }

    QListWidget::item {
        padding: 8px;
    }

    QListWidget::item:hover {
        background-color: #44475a;
    }

    QListWidget::item:selected {
        background-color: #bd93f9;
        color: #282a36;
    }

    /* Прогресс бар */
    QProgressBar {
        background-color: #44475a;
        color: #f8f8f2;
        border: 1px solid #6272a4;
        border-radius: 4px;
        text-align: center;
        height: 25px;
    }

    QProgressBar::chunk {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #bd93f9, stop:0.5 #ff79c6, stop:1 #8be9fd);
        border-radius: 3px;
    }

    /* Скроллбары */
    QScrollBar:vertical {
        background-color: #282a36;
        width: 12px;
        border-radius: 6px;
    }

    QScrollBar::handle:vertical {
        background-color: #6272a4;
        border-radius: 6px;
        min-height: 20px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #bd93f9;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    QScrollBar:horizontal {
        background-color: #282a36;
        height: 12px;
        border-radius: 6px;
    }

    QScrollBar::handle:horizontal {
        background-color: #6272a4;
        border-radius: 6px;
        min-width: 20px;
    }

    QScrollBar::handle:horizontal:hover {
        background-color: #bd93f9;
    }

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }

    /* Область прокрутки */
    QScrollArea {
        background-color: #282a36;
        border: none;
    }

    QScrollArea > QWidget > QWidget {
        background-color: #282a36;
    }

    /* Статус бар */
    QStatusBar {
        background-color: #44475a;
        color: #f8f8f2;
    }

    /* Диалоги */
    QMessageBox {
        background-color: #282a36;
    }

    QMessageBox QLabel {
        color: #f8f8f2;
    }

    QMessageBox QPushButton {
        min-width: 80px;
    }
    """

    app.setStyleSheet(dracula_stylesheet)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()