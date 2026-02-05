"""
Модуль с классом LLMBenchmark для тестирования производительности LLM моделей
"""

import time
import asyncio
import logging
import csv
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from typing import Dict, List, Optional
from statistics import mean, stdev
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from prompts import TEST_CASES, TestCase

logger = logging.getLogger(__name__)


def get_app_path() -> Path:
    """Возвращает путь к директории приложения (работает и для exe, и для .py)"""
    if getattr(sys, 'frozen', False):
        # Запуск из exe (PyInstaller)
        return Path(sys.executable).parent
    else:
        # Запуск из исходников
        return Path(__file__).resolve().parent


class LLMBenchmark:
    """
    Класс для бенчмаркинга LLM модели

    Поддерживает последовательные и параллельные тесты производительности
    """

    CSV_FILE = get_app_path() / "benchmark_results.csv"
    BASE_API_URL = "https://nizamov.school/inference_benchmark/"
    RESULTS_API_URL = BASE_API_URL + "results/"

    def __init__(self, inference: str, model: str, base_url: str, launch: str,
                 gpu_count: int, gpu_models: list,
                 api_key: str = "local", description: str = "",
                 parallel: int = 10, inference_version: Optional[str] = None,
                 is_docker: bool = False):
        """
        Инициализация бенчмарка

        Args:
            inference: движок инференса (vllm, ollama, llama_cpp)
            model: имя модели
            base_url: URL API сервера
            launch: параметры запуска инференса
            gpu_count: количество GPU
            gpu_models: список моделей GPU
            api_key: ключ API
            description: описание теста
            parallel: количество параллельных запросов
            inference_version: версия инференса
            is_docker: запуск в Docker
        """
        self.inference_engine = inference
        self.model_name_from_config = model
        self.launch_params = launch
        self.description = description
        self.gpu_count = gpu_count
        self.gpu_models = ", ".join(gpu_models)
        self.parallel = parallel
        self.inference_version = inference_version
        self.is_docker = is_docker

        if inference == "ollama":
            self.llm = ChatOllama(base_url=base_url, model=model, temperature=0.1)
        else:
            self.llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model, temperature=0.1)

        # Кеш SO-обёрток для каждой схемы из тестовых сценариев
        self._so_cache = {}
        for tc in TEST_CASES:
            if tc.schema not in self._so_cache:
                self._so_cache[tc.schema] = self.llm.with_structured_output(
                    schema=tc.schema, include_raw=True
                )

        self.model_name = self._get_model_name()

        # IDs для API
        self.model_id = None
        self.inference_id = None
        self.inference_version_id = None
        self.gpu_model_ids = []

    def _get_model_name(self) -> str:
        """
        Получает имя модели из метаданных LLM клиента

        Returns:
            имя модели или "Unknown Model"
        """
        try:
            # Пытаемся получить имя модели из wrapped клиента
            if hasattr(self.llm, 'bound'):
                llm_model = self.llm.bound
            else:
                llm_model = self.llm

            # Для ChatOpenAI и подобных
            if hasattr(llm_model, 'model_name'):
                return llm_model.model_name
            elif hasattr(llm_model, 'model'):
                return llm_model.model
            else:
                return "Unknown Model"
        except Exception as e:
            return f"Unknown Model (ошибка: {str(e)})"

    def _fetch_api_data(self, endpoint: str) -> List[Dict]:
        """Получает данные из API"""
        url = self.BASE_API_URL + endpoint
        try:
            req = Request(url, headers={"Content-Type": "application/json"})
            with urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data
        except HTTPError as e:
            logger.error("Ошибка при запросе к API %s: HTTP %d — %s", url, e.code, e.reason)
            return []
        except URLError as e:
            logger.error("Не удалось подключиться к %s: %s", url, e.reason)
            return []
        except json.JSONDecodeError as e:
            logger.error("Ошибка при разборе JSON из %s: %s", url, e)
            return []

    def _create_api_entry(self, endpoint: str, data: Dict) -> Optional[int]:
        """Создает новую запись в API и возвращает ID"""
        url = self.BASE_API_URL + endpoint
        try:
            json_data = json.dumps(data).encode("utf-8")
            req = Request(url, data=json_data, headers={"Content-Type": "application/json"})
            with urlopen(req) as resp:
                response_data = json.loads(resp.read().decode('utf-8'))
                # Assuming the response contains the created object with an ID
                return response_data.get('id')
        except HTTPError as e:
            logger.error("Ошибка при создании записи в API %s: HTTP %d — %s", url, e.code, e.reason)
            return None
        except URLError as e:
            logger.error("Не удалось подключиться к %s для создания записи: %s", url, e.reason)
            return None
        except json.JSONDecodeError as e:
            logger.error("Ошибка при разборе JSON из %s: %s", url, e)
            return None

    def get_or_create_model_id(self, model_name: str) -> int:
        """Получает или создает ID модели"""
        models = self._fetch_api_data("model-names/")
        for model in models:
            if model['name'] == model_name:
                logger.info("Найдена существующая модель: %s (ID: %d)", model_name, model['id'])
                return model['id']

        # Создаем новую модель
        logger.info("Создание новой модели: %s", model_name)
        new_model_id = self._create_api_entry("model-names/", {"name": model_name})
        if new_model_id:
            logger.info("Создана новая модель: %s (ID: %d)", model_name, new_model_id)
            return new_model_id
        else:
            raise Exception(f"Не удалось создать модель: {model_name}")

    def get_or_create_inference_id(self, inference_type: str) -> int:
        """Получает или создает ID типа инференса"""
        inf_types = self._fetch_api_data("inference-types/")
        for inf_type in inf_types:
            if inf_type['name'] == inference_type:
                logger.info("Найден существующий тип инференса: %s (ID: %d)", inference_type, inf_type['id'])
                return inf_type['id']

        # Создаем новый тип инференса
        logger.info("Создание нового типа инференса: %s", inference_type)
        new_inf_id = self._create_api_entry("inference-types/", {"name": inference_type})
        if new_inf_id:
            logger.info("Создан новый тип инференса: %s (ID: %d)", inference_type, new_inf_id)
            return new_inf_id
        else:
            raise Exception(f"Не удалось создать тип инференса: {inference_type}")

    def get_or_create_inference_version_id(self, inference_id: int, version: str) -> Optional[int]:
        """Получает или создает ID версии инференса"""
        if not version:
            return None

        versions = self._fetch_api_data("inference-versions/")
        for ver in versions:
            if ver['inference'] == inference_id and ver['version'] == version:
                logger.info("Найдена существующая версия инференса: %s (ID: %d)", version, ver['id'])
                return ver['id']

        # Создаем новую версию инференса
        logger.info("Создание новой версии инференса: %s для типа %d", version, inference_id)
        new_ver_id = self._create_api_entry("inference-versions/", {"inference": inference_id, "version": version})
        if new_ver_id:
            logger.info("Создана новая версия инференса: %s (ID: %d)", version, new_ver_id)
            return new_ver_id
        else:
            logger.warning("Не удалось создать версию инференса: %s", version)
            return None

    def get_or_create_gpu_model_ids(self, gpu_models: List[str]) -> List[int]:
        """Получает или создает ID моделей GPU"""
        gpu_model_ids = []
        gpu_list = self._fetch_api_data("gpu-models/")

        for gpu_model in gpu_models:
            found = False
            for gpu in gpu_list:
                if gpu['name'] == gpu_model:
                    logger.info("Найдена существующая модель GPU: %s (ID: %d)", gpu_model, gpu['id'])
                    gpu_model_ids.append(gpu['id'])
                    found = True
                    break

            if not found:
                # Создаем новую модель GPU
                logger.info("Создание новой модели GPU: %s", gpu_model)
                new_gpu_id = self._create_api_entry("gpu-models/", {"name": gpu_model})
                if new_gpu_id:
                    logger.info("Создана новая модель GPU: %s (ID: %d)", gpu_model, new_gpu_id)
                    gpu_model_ids.append(new_gpu_id)
                else:
                    logger.error("Не удалось создать модель GPU: %s", gpu_model)
                    # Вместо исключения, добавляем заглушку
                    gpu_model_ids.append(1)  # используем ID 1 как fallback

        return gpu_model_ids

    def _save_to_csv(self, results: Dict):
        """Сохраняет результаты в CSV файл"""
        file_exists = self.CSV_FILE.exists()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')

            if not file_exists:
                writer.writerow([
                    'Дата и время',
                    'Описание',
                    'Модель',
                    'Инференс',
                    'Параметры запуска',
                    'GPU кол-во',
                    'GPU модели',
                    'Параллельных запросов',
                    'Послед. без SO (ток/сек)',
                    'Послед. с SO (ток/сек)',
                    'Паралл. без SO (ток/сек)',
                    'Паралл. с SO (ток/сек)',
                    'Пропускная без SO (ток/сек)',
                    'Пропускная с SO (ток/сек)',
                ])

            writer.writerow([
                timestamp,
                self.description,
                self.model_name_from_config,
                self.inference_engine,
                self.launch_params,
                self.gpu_count,
                self.gpu_models,
                self.parallel,
                f"{results['seq_no_so']:.2f}",
                f"{results['seq_so']:.2f}",
                f"{results['par_no_so']:.2f}",
                f"{results['par_so']:.2f}",
                f"{results['throughput_no_so']:.2f}",
                f"{results['throughput_so']:.2f}",
            ])

        logger.info("Результаты добавлены в %s", self.CSV_FILE)

    def prepare_api_data(self):
        """Подготавливает данные для отправки в API"""
        # Получаем или создаем все необходимые ID
        self.model_id = self.get_or_create_model_id(self.model_name_from_config)
        self.inference_id = self.get_or_create_inference_id(self.inference_engine)
        self.inference_version_id = self.get_or_create_inference_version_id(self.inference_id, self.inference_version) if self.inference_version else None
        self.gpu_model_ids = self.get_or_create_gpu_model_ids([gpu.strip() for gpu in self.gpu_models.split(",") if gpu.strip()])

        # Возвращаем ID первой GPU модели, как того требует API
        first_gpu_id = self.gpu_model_ids[0] if self.gpu_model_ids else 1

        return {
            "description": self.description[:255],  # ограничение API
            "model_name": self.model_id,
            "inference": self.inference_id,
            "inference_version": self.inference_version_id,
            "launch_params": self.launch_params,
            "gpu_count": self.gpu_count,
            "parallel_count": self.parallel,
            "gpu_model": first_gpu_id,  # API принимает только один ID GPU
            "is_docker": self.is_docker,
            "sequential_no_so": round(self.results['seq_no_so'], 2),
            "sequential_with_so": round(self.results['seq_so'], 2),
            "parallel_no_so": round(self.results['par_no_so'], 2),
            "parallel_with_so": round(self.results['par_so'], 2),
            "throughput_no_so": round(self.results['throughput_no_so'], 2),
            "throughput_with_so": round(self.results['throughput_so'], 2),
        }

    def _post_results(self, results: Dict):
        """Отправляет результаты POST-запросом на сервер"""
        # Сохраняем результаты для использования в prepare_api_data
        self.results = results

        payload = self.prepare_api_data()

        data = json.dumps(payload).encode("utf-8")
        req = Request(self.RESULTS_API_URL, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req) as resp:
                logger.info("Результаты отправлены на %s (HTTP %d)", self.RESULTS_API_URL, resp.status)
        except HTTPError as e:
            logger.error("Ошибка при отправке результатов: HTTP %d — %s", e.code, e.reason)
        except URLError as e:
            logger.error("Не удалось подключиться к %s: %s", self.RESULTS_API_URL, e.reason)

    def single_request(self, test_case: TestCase, use_structured_output: bool = False) -> Dict:
        """
        Выполняет один запрос к модели и возвращает статистику

        Args:
            test_case: тестовый сценарий (промпт + схема)
            use_structured_output: использовать ли structured output

        Returns:
            словарь со статистикой (время, токены, скорость)
        """
        start_time = time.time()

        if use_structured_output:
            so = self._so_cache[test_case.schema]
            result = so.invoke(test_case.prompt)
            raw_response = result["raw"]
        else:
            raw_response = self.llm.invoke(test_case.prompt)

        total_time = time.time() - start_time

        usage = getattr(raw_response, "usage_metadata", None)
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        else:
            input_tokens = 0
            output_tokens = 0

        tokens_per_second = output_tokens / total_time if total_time > 0 and output_tokens > 0 else 0

        return {
            "time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second
        }

    async def _parallel_request(self, test_case: TestCase, request_id: int, use_structured_output: bool = True) -> Dict:
        """Асинхронный запрос к модели (для параллельного выполнения)"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.single_request, test_case, use_structured_output
        )
        result["request_id"] = request_id
        return result

    async def run_parallel(self, num_parallel: int, use_structured_output: bool = True) -> Dict:
        """
        Запускает несколько параллельных запросов к модели

        Args:
            num_parallel: количество параллельных запросов
            use_structured_output: использовать ли structured output

        Returns:
            словарь с результатами: avg_speed и throughput
        """
        so_label = "с SO" if use_structured_output else "без SO"
        logger.info("Параллельный тест: %d запросов (%s)", num_parallel, so_label)

        start_time = time.time()

        tasks = [
            self._parallel_request(TEST_CASES[i % len(TEST_CASES)], i + 1, use_structured_output)
            for i in range(num_parallel)
        ]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        total_output_tokens = 0
        for result in results:
            logger.debug("Запрос #%d: %.2f сек, %d токенов, %.2f ток/сек",
                         result['request_id'], result['time'],
                         result['output_tokens'], result['tokens_per_second'])
            total_output_tokens += result['output_tokens']

        avg_speed = mean([r['tokens_per_second'] for r in results])
        throughput = total_output_tokens / total_time

        logger.info("Параллельный итог: %.2f сек, %d токенов, "
                    "средняя скорость %.2f ток/сек, пропускная способность %.2f ток/сек",
                    total_time, total_output_tokens, avg_speed, throughput)

        # Возвращаем результаты для CSV
        return {
            'avg_speed': avg_speed,
            'throughput': throughput
        }

    def warmup(self):
        """Прогревочный запрос"""
        result = self.single_request(TEST_CASES[0])
        logger.debug("Прогрев: %.2f сек (%.2f ток/сек)",
                     result['time'], result['tokens_per_second'])

    def run_sequential(self, num_runs: int = 3, use_structured_output: bool = False) -> Dict:
        """
        Последовательный бенчмарк. Сценарии чередуются по кругу (round-robin).

        Args:
            num_runs: количество замеров
            use_structured_output: использовать ли structured output

        Returns:
            словарь с avg_speed
        """
        label = "с SO" if use_structured_output else "без SO"
        logger.info("Последовательный тест (%s): %d замеров", label, num_runs)

        results = []
        for i in range(num_runs):
            tc = TEST_CASES[i % len(TEST_CASES)]
            result = self.single_request(tc, use_structured_output=use_structured_output)
            results.append(result)
            logger.debug("Замер %s %d/%d [%s]: %.2f сек, %d токенов, %.2f ток/сек",
                         label, i + 1, num_runs, tc.name, result['time'],
                         result['output_tokens'], result['tokens_per_second'])

        speeds = [r['tokens_per_second'] for r in results]
        std = stdev(speeds) if len(speeds) > 1 else 0
        logger.info("Последовательный (%s): средняя %.2f ток/сек (мин %.2f, макс %.2f, std %.2f)",
                    label, mean(speeds), min(speeds), max(speeds), std)

        return {'avg_speed': mean(speeds)}