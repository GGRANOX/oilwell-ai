import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone
import joblib
import logging
from datetime import datetime, timedelta
import time
import sys
import locale
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import unittest
import yaml
import os
from pathlib import Path
import warnings
from collections import deque
import hashlib
import json
from scipy.stats import randint, uniform
import requests
from io import StringIO

# Настройки для промышленного использования
warnings.filterwarnings('ignore')
try:
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
except locale.Error:
    # Локали ru_RU нет в системе (часто на Windows и в Docker) — остаёмся на системной
    locale.setlocale(locale.LC_ALL, '')


class IndustrialLogger:
    """Промышленная система логирования с ротацией и удаленным мониторингом"""

    def __init__(self):
        self.logger = logging.getLogger('OilWellAI')
        self.logger.setLevel(logging.INFO)

        # Формат логов
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(module)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Обработчик для файла с ротацией
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            'oil_well_ai.log',
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        # Обработчик для консоли
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Очистка старых обработчиков
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Мониторинг ошибок
        self.error_count = 0
        self.last_error_time = None

    def __del__(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def log_operation(self, message: str):
        """Логирование операционных событий"""
        self.logger.info(f"OPERATION: {message}")

    def log_warning(self, message: str):
        """Логирование предупреждений"""
        self.logger.warning(f"WARNING: {message}")

    def log_error(self, message: str):
        """Логирование ошибок с подсчетом частоты"""
        self.error_count += 1
        self.last_error_time = datetime.now()
        self.logger.error(f"ERROR: {message}")
        self._check_error_rate()

    def _check_error_rate(self):
        """Проверка частоты ошибок для оповещения"""
        if self.error_count >= 10 and (
                self.last_error_time and
                (datetime.now() - self.last_error_time) < timedelta(minutes=5)
        ):
            self._send_alert("Высокая частота ошибок в системе!")

    def _send_alert(self, message: str):
        """Отправка критических оповещений"""
        try:
            # Здесь может быть интеграция с Slack, Telegram или SMS
            self.logger.critical(f"ALERT: {message}")
            # requests.post(ALERT_WEBHOOK, json={"message": message})
        except Exception as e:
            self.logger.error(f"Ошибка отправки оповещения: {str(e)}")


class IndustrialOilWellAI:
    """Промышленная версия системы управления нефтяной скважиной"""

    def __init__(self, model_path: str = "oil_well_model.joblib"):
        self.logger = IndustrialLogger()

        # Инициализация моделей
        self.models = self._initialize_models()
        self.ensemble = self._create_ensemble()

        # Конфигурация системы
        self.config = self._load_config()
        self.model_path = model_path

        # Данные и состояние системы
        self.history = []
        self.critical_events = []
        self.data_buffer = deque(maxlen=100000)
        self.system_status = "initializing"

        # Параметры обучения
        self.min_training_samples = self.config['training_params'].get('min_training_samples', 5000)
        self.retrain_interval = timedelta(
            hours=self.config['training_params'].get('retrain_interval_hours', 24)
        )
        self.last_retrain_time = datetime.now() - self.retrain_interval

        # Инициализация системы
        self._initialize_system()

    def _initialize_models(self) -> Dict[str, BaseEstimator]:
        """Инициализация моделей с гиперпараметрами"""
        return {
            "rf": Pipeline([
                ('scaler', RobustScaler()),
                ('model', RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=5,
                    n_jobs=-1,
                    random_state=42
                ))
            ]),
            "gb": Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ))
            ]),
            "nn": Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    activation='relu',
                    solver='adam',
                    max_iter=2000,
                    early_stopping=True,
                    random_state=42
                ))
            ])
        }

    def _create_ensemble(self) -> VotingRegressor:
        """Создание ансамбля моделей"""
        return VotingRegressor([
            ('rf', self.models['rf']),
            ('gb', self.models['gb']),
            ('nn', self.models['nn'])
        ], weights=[0.4, 0.3, 0.3])

    def _load_config(self) -> Dict:
        """Загрузка конфигурации с проверкой"""
        config_path = Path('config.yaml')
        if not config_path.exists():
            self.logger.log_warning("Конфигурационный файл не найден, используются значения по умолчанию")
            return self._get_default_config()

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            # Проверка обязательных разделов
            required_sections = ['emergency_thresholds', 'training_params', 'data_columns']
            if not all(section in config for section in required_sections):
                missing = set(required_sections) - set(config.keys())
                self.logger.log_warning(
                    f"В конфигурации отсутствуют разделы: {missing}, "
                    "используются значения по умолчанию"
                )
                return self._get_default_config()

            return config

        except Exception as e:
            self.logger.log_error(f"Ошибка загрузки конфигурации: {str(e)}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Возвращает конфигурацию по умолчанию"""
        return {
            'emergency_thresholds': {
                'min_oil_level': 5.0,
                'max_pressure': 15.0,
                'max_current': 5.0,
                'max_vibration': 0.8,
                'max_temperature': 40.0
            },
            'training_params': {
                'test_size': 0.2,
                'initial_samples': 10000,
                'retrain_samples': 2000,
                'retrain_interval_hours': 24,
                'min_training_samples': 5000,
                'time_series_split': 5,
                'hyperparam_tuning': False
            },
            'data_columns': {
                'features': ['oil_level', 'pressure', 'current', 'vibration', 'temperature'],
                'target': 'optimal_power',
                'timestamp': 'datetime'
            },
            'system': {
                'max_response_time_ms': 500,
                'data_validation': True
            }
        }

    def _initialize_system(self):
        """Инициализация системы"""
        try:
            if self.load_model():
                self.system_status = "operational"
                self.logger.log_operation("Система успешно инициализирована с загруженной моделью")
            else:
                # Генерация тестовых данных, если файла нет
                if not Path("data.csv").exists():
                    self._generate_test_data()
                self.train()
                self.system_status = "operational"
                self.logger.log_operation("Система инициализирована с новой моделью")
        except Exception as e:
            self.system_status = "error"
            self.logger.log_error(f"Ошибка инициализации системы: {str(e)}")
            raise

    def _generate_test_data(self):
        """Генерация тестовых данных"""
        test_data = pd.DataFrame({
            'oil_level': np.random.uniform(10, 100, 1000),
            'pressure': np.random.uniform(0, 20, 1000),
            'current': np.random.uniform(0, 5, 1000),
            'vibration': np.random.uniform(0, 1, 1000),
            'temperature': np.random.uniform(30, 60, 1000),
            'optimal_power': np.random.uniform(50, 100, 1000),
            'datetime': pd.date_range(start='2023-01-01', periods=1000, freq='h')
        })
        test_data.to_csv("data.csv", index=False)
        self.logger.log_operation("Сгенерированы тестовые данные в data.csv")

    def load_data(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Загрузка данных из различных источников"""
        try:
            if isinstance(source, pd.DataFrame):
                return source.copy()

            if isinstance(source, (str, Path)):
                source = Path(source)
                if not source.exists():
                    raise FileNotFoundError(f"Файл {source} не найден")

                if source.suffix == '.csv':
                    df = pd.read_csv(source, parse_dates=['datetime'])
                elif source.suffix in ('.xlsx', '.xls'):
                    df = pd.read_excel(source, parse_dates=['datetime'])
                else:
                    raise ValueError(f"Неподдерживаемый формат файла: {source.suffix}")

                return df
            else:
                raise ValueError(f"Неизвестный тип данных: {type(source)}")
        except Exception as e:
            self.logger.log_error(f"Ошибка загрузки данных: {str(e)}")
            raise

    def check_data(self, df: pd.DataFrame, require_target: bool = True) -> bool:
        """Проверка данных; для предсказания целевой столбец не нужен (require_target=False)"""
        required_columns = list(self.config['data_columns']['features'])
        if require_target:
            required_columns.append(self.config['data_columns']['target'])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.log_warning(f"Отсутствуют столбцы: {missing_columns}")
            return False

        # Проверка на наличие пропусков
        if df[required_columns].isnull().any().any():
            self.logger.log_warning("В данных присутствуют пропущенные значения")
            return False

        # Проверка форматов данных
        for col in self.config['data_columns']['features']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.logger.log_warning(f"Столбец {col} не является числовым")
                return False

        self.logger.log_operation("Данные прошли проверку")
        return True

    def train(self, df: Optional[pd.DataFrame] = None):
        """Обучение модели на предоставленных или загруженных данных"""
        try:
            if df is None:
                df = self.load_data("data.csv")

            if not self.check_data(df):
                raise ValueError("Данные не прошли валидацию")

            features = self.config['data_columns']['features']
            target = self.config['data_columns']['target']

            X = df[features]
            y = df[target]

            tscv = TimeSeriesSplit(n_splits=self.config['training_params'].get('time_series_split', 5))

            # Оценка на каждом фолде: копия ансамбля учится на прошлом, проверяется на будущем
            maes, r2s = [], []
            for fold, (train_index, test_index) in enumerate(tscv.split(X), start=1):
                fold_model = clone(self.ensemble)
                fold_model.fit(X.iloc[train_index], y.iloc[train_index])
                preds = fold_model.predict(X.iloc[test_index])
                maes.append(mean_absolute_error(y.iloc[test_index], preds))
                r2s.append(r2_score(y.iloc[test_index], preds))
                self.logger.log_operation(f"Фолд {fold}: MAE={maes[-1]:.4f}, R2={r2s[-1]:.4f}")

            self.cv_metrics = {
                'mae_mean': float(np.mean(maes)), 'mae_std': float(np.std(maes)),
                'r2_mean': float(np.mean(r2s)), 'r2_std': float(np.std(r2s)),
                'folds': len(maes)
            }
            self.logger.log_operation(
                f"Кросс-валидация ({len(maes)} фолдов): "
                f"MAE={self.cv_metrics['mae_mean']:.4f} ± {self.cv_metrics['mae_std']:.4f}, "
                f"R2={self.cv_metrics['r2_mean']:.4f} ± {self.cv_metrics['r2_std']:.4f}"
            )

            # Финальная модель обучается на всех данных
            self.ensemble.fit(X, y)
            self.logger.log_operation("Обучение завершено на всех данных")

            self.last_retrain_time = datetime.now()
            joblib.dump(self.ensemble, self.model_path)
            self.logger.log_operation(f"Модель сохранена в {self.model_path}")

        except Exception as e:
            self.logger.log_error(f"Ошибка обучения модели: {str(e)}")
            raise

    def save_model(self):
        """Сохранение модели на диск"""
        try:
            joblib.dump(self.ensemble, self.model_path)
            self.logger.log_operation(f"Модель успешно сохранена: {self.model_path}")
        except Exception as e:
            self.logger.log_error(f"Ошибка сохранения модели: {str(e)}")

    def load_model(self) -> bool:
        """Загрузка модели с диска"""
        if Path(self.model_path).exists():
            try:
                self.ensemble = joblib.load(self.model_path)
                self.logger.log_operation(f"Модель успешно загружена: {self.model_path}")
                return True
            except Exception as e:
                self.logger.log_error(f"Ошибка загрузки модели: {str(e)}")
        return False

    def predict(self, df: Optional[pd.DataFrame] = None):
        """Предсказание с использованием обученной модели"""
        try:
            if df is None:
                self.logger.log_error("Необходимо передать данные для предсказания.")
                return None

            if isinstance(df, dict):  # Если передан словарь, преобразуем в DataFrame
                df = pd.DataFrame([df])

            if not self.check_data(df, require_target=False):
                self.logger.log_error("Данные для предсказания невалидны.")
                return None

            features = self.config['data_columns']['features']
            df = df[features]

            # Масштабирование не нужно: каждый пайплайн ансамбля масштабирует данные сам

            # Проверим, что модель обучена
            if not hasattr(self, 'ensemble') or self.ensemble is None:
                self.logger.log_error("Модель не была обучена.")
                return None

            # Предсказание
            self.logger.log_operation("Предсказание с использованием модели...")
            predictions = self.ensemble.predict(df)

            if len(predictions) == 0:
                self.logger.log_error("Предсказания пусты.")
                return None

            self.logger.log_operation(
                f"Предсказания: {predictions[:5]}...")  # Выводим первые 5 предсказаний для проверки

            return float(predictions[0])  # Возвращаем первое предсказание и приводим к типу float

        except Exception as e:
            self.logger.log_error(f"Ошибка при предсказании: {str(e)}")
            return None


class TestOilWellAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Удаляем старые файлы перед запуском тестов
        if Path("oil_well_model.joblib").exists():
            Path("oil_well_model.joblib").unlink()
        if Path("data.csv").exists():
            Path("data.csv").unlink()

    def setUp(self):
        self.ai_system = IndustrialOilWellAI()

    def test_training(self):
        # Тестирование обучения с генерацией данных
        self.ai_system._generate_test_data()
        df = self.ai_system.load_data("data.csv")
        self.ai_system.train(df)
        self.assertIsInstance(self.ai_system.ensemble, VotingRegressor)

    def test_prediction(self):
        # Тестирование предсказания
        input_data = {
            'oil_level': 50,
            'pressure': 10,
            'current': 3,
            'vibration': 0.5,
            'temperature': 40
        }
        prediction = self.ai_system.predict(input_data)
        self.assertIsInstance(prediction, float)

    def test_load_save_model(self):
        # Тестирование сохранения и загрузки модели
        self.ai_system._generate_test_data()
        self.ai_system.train()
        self.ai_system.save_model()
        self.assertTrue(self.ai_system.load_model())


if __name__ == '__main__':
    unittest.main()