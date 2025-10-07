"""
LLM-Use - Intelligent LLM Router (Production-Ready Edition)
Complete implementation with all components
"""

import os
import yaml
import json
import time
import hashlib
import tiktoken
import asyncio
import threading
import sqlite3
import random
import logging
import uuid
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Generator, Any, Union, AsyncGenerator
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict
from queue import Queue
from functools import lru_cache

# Scientific and NLP imports
import scipy.stats as stats
import spacy
import textstat
import language_tool_python
from sentence_transformers import SentenceTransformer, util

# ML imports
from transformers import pipeline
import torch

# API imports
import openai
from anthropic import AsyncAnthropic
import google.generativeai as genai
from groq import AsyncGroq

# Web framework
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # ← AGGIUNGI QUESTA RIGA
from pydantic import BaseModel
import uvicorn

# Metrics
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================
# DATA CLASSES
# ====================

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    provider: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    quality: int
    speed: str
    context_window: int
    supports_streaming: bool
    best_for: List[str]

@dataclass
class RoutingResult:
    """Routing decision result"""
    model_id: str
    model_name: str
    complexity: int
    reason: str
    estimated_tokens_input: int
    estimated_tokens_output: int
    estimated_cost: float
    is_free: bool
    timestamp: str

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_id: str
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency: float = 0.0
    error_rate: float = 0.0
    quality_scores: List[float] = field(default_factory=list)

@dataclass
class BenchmarkResult:
    """Detailed benchmark result"""
    model_id: str
    prompt_type: str
    latency: float
    tokens_per_second: float
    quality_score: float
    cost: float
    error: Optional[str] = None
    response_sample: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# ====================
# METRICS COLLECTOR
# ====================

class MetricsCollector:
    """Collect and export metrics - SINGLETON"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Inizializza solo una volta
        if self._initialized:
            return
            
        self.metrics = defaultdict(lambda: defaultdict(float))
        
        # Prometheus metrics - solo se non già registrate
        try:
            self.request_counter = Counter('llm_requests_total', 'Total LLM requests', ['model', 'provider'])
            self.token_counter = Counter('llm_tokens_total', 'Total tokens processed', ['model', 'type'])
            self.cost_gauge = Gauge('llm_cost_total', 'Total cost in USD')
            self.latency_histogram = Histogram('llm_latency_seconds', 'Request latency', ['model'])
            self.error_counter = Counter('llm_errors_total', 'Total errors', ['model', 'error_type'])
            
            # Start Prometheus server solo una volta
            start_http_server(8000)
            logger.info("Prometheus metrics available at http://localhost:8000")
            
        except Exception as e:
            # Se metriche già esistono, usale
            logger.warning(f"Metrics already exist, reusing: {e}")
            from prometheus_client import REGISTRY
            
            # Trova le metriche esistenti
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, '_name'):
                    if 'llm_requests_total' in collector._name:
                        self.request_counter = collector
                    elif 'llm_tokens_total' in collector._name:
                        self.token_counter = collector
                    elif 'llm_cost_total' in collector._name:
                        self.cost_gauge = collector
                    elif 'llm_latency_seconds' in collector._name:
                        self.latency_histogram = collector
                    elif 'llm_errors_total' in collector._name:
                        self.error_counter = collector
        
        self._initialized = True
    
    def record_request(self, model: str, provider: str, tokens_in: int, tokens_out: int, 
                       latency: float, cost: float, error: Optional[str] = None):
        """Record request metrics"""
        
        # Internal metrics
        self.metrics[model]['requests'] += 1
        self.metrics[model]['tokens_input'] += tokens_in
        self.metrics[model]['tokens_output'] += tokens_out
        self.metrics[model]['total_latency'] += latency
        self.metrics[model]['total_cost'] += cost
        
        if error:
            self.metrics[model]['errors'] += 1
        
        # Prometheus metrics
        self.request_counter.labels(model=model, provider=provider).inc()
        self.token_counter.labels(model=model, type='input').inc(tokens_in)
        self.token_counter.labels(model=model, type='output').inc(tokens_out)
        self.cost_gauge.inc(cost)
        self.latency_histogram.labels(model=model).observe(latency)
        
        if error:
            self.error_counter.labels(model=model, error_type=error).inc()
    
    def get_metrics(self) -> Dict:
        """Get all metrics"""
        return dict(self.metrics)

# ====================
# TOKEN COUNTER
# ====================

class TokenCounter:
    """Count tokens for different models"""
    
    def __init__(self):
        self.encoders = {}
        self._load_encoders()
    
    def _load_encoders(self):
        """Load tokenizers for different models"""
        try:
            # GPT models
            self.encoders['gpt-3.5-turbo'] = tiktoken.encoding_for_model('gpt-3.5-turbo')
            self.encoders['gpt-4'] = tiktoken.encoding_for_model('gpt-4')
            self.encoders['gpt-4-turbo-preview'] = tiktoken.encoding_for_model('gpt-4-turbo-preview')
        except:
            logger.warning("Failed to load some tiktoken encoders")
        
        # Default encoder
        try:
            self.encoders['default'] = tiktoken.get_encoding('cl100k_base')
        except:
            logger.warning("Failed to load default tiktoken encoder")
    
    def count_tokens(self, text: str, model: str = 'default') -> int:
        """Count tokens for text"""
        # Try specific encoder
        if model in self.encoders:
            return len(self.encoders[model].encode(text))
        
        # Try default encoder
        if 'default' in self.encoders:
            return len(self.encoders['default'].encode(text))
        
        # Model-specific approximations
        model_ratios = {
            'claude': 0.75,
            'gpt': 0.80,
            'mistral': 0.85,
            'llama': 0.85,
            'gemini': 0.78,
            'default': 0.75
        }
        
        # Better approximation based on character count
        char_count = len(text)
        for key in model_ratios:
            if key in model.lower():
                return int(char_count / 4 * model_ratios[key])
        
        return int(char_count / 4 * model_ratios['default'])

# ====================
# CONFIGURATION LOADER
# ====================

class ConfigLoader:
    """Load and manage configuration"""
    
    def __init__(self, config_path: str = "models.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.models = self._parse_models()
        
    def _load_config(self) -> Dict:
        """Load YAML configuration"""
        if not Path(self.config_path).exists():
            self._create_default_config()
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            "models": {
                "gpt-3.5-turbo": {
                    "name": "GPT-3.5 Turbo",
                    "provider": "openai",
                    "cost_per_1k_input": 0.0005,
                    "cost_per_1k_output": 0.0015,
                    "quality": 7,
                    "speed": "fast",
                    "context_window": 16385,
                    "supports_streaming": True,
                    "best_for": ["general", "chat", "simple_tasks"]
                },
                "gpt-4-turbo-preview": {
                    "name": "GPT-4 Turbo",
                    "provider": "openai",
                    "cost_per_1k_input": 0.01,
                    "cost_per_1k_output": 0.03,
                    "quality": 10,
                    "speed": "medium",
                    "context_window": 128000,
                    "supports_streaming": True,
                    "best_for": ["complex", "reasoning", "coding", "analysis"]
                },
                "claude-3-haiku": {
                    "name": "Claude 3 Haiku",
                    "provider": "anthropic",
                    "cost_per_1k_input": 0.00025,
                    "cost_per_1k_output": 0.00125,
                    "quality": 7,
                    "speed": "fast",
                    "context_window": 200000,
                    "supports_streaming": True,
                    "best_for": ["general", "chat", "summarization"]
                },
                "claude-3-opus": {
                    "name": "Claude 3 Opus",
                    "provider": "anthropic",
                    "cost_per_1k_input": 0.015,
                    "cost_per_1k_output": 0.075,
                    "quality": 10,
                    "speed": "slow",
                    "context_window": 200000,
                    "supports_streaming": True,
                    "best_for": ["analysis", "writing", "coding", "research"]
                },
                "gemini-pro": {
                    "name": "Gemini Pro",
                    "provider": "google",
                    "cost_per_1k_input": 0.00025,
                    "cost_per_1k_output": 0.0005,
                    "quality": 8,
                    "speed": "fast",
                    "context_window": 30720,
                    "supports_streaming": True,
                    "best_for": ["general", "analysis", "multimodal"]
                },
                "mixtral-8x7b": {
                    "name": "Mixtral 8x7B",
                    "provider": "groq",
                    "cost_per_1k_input": 0.00027,
                    "cost_per_1k_output": 0.00027,
                    "quality": 8,
                    "speed": "very_fast",
                    "context_window": 32768,
                    "supports_streaming": True,
                    "best_for": ["general", "coding", "fast_inference"]
                },
                "llama-3-70b": {
                    "name": "Llama 3 70B",
                    "provider": "groq",
                    "cost_per_1k_input": 0.00059,
                    "cost_per_1k_output": 0.00079,
                    "quality": 9,
                    "speed": "very_fast",
                    "context_window": 8192,
                    "supports_streaming": True,
                    "best_for": ["general", "reasoning", "instruction_following"]
                },
                "mistral": {
                    "name": "Mistral 7B (Local)",
                    "provider": "ollama",
                    "cost_per_1k_input": 0.0,
                    "cost_per_1k_output": 0.0,
                    "quality": 6,
                    "speed": "fast",
                    "context_window": 8192,
                    "supports_streaming": True,
                    "best_for": ["local", "private", "general"]
                }
            },
            "routing_rules": {
                "complexity_thresholds": {
                    "simple": 3,
                    "moderate": 6,
                    "complex": 10
                },
                "overrides": []
            },
            "providers": {
                "openai": {
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout": 30,
                    "base_url": "https://api.openai.com/v1"
                },
                "anthropic": {
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "timeout": 30,
                    "base_url": "https://api.anthropic.com"
                },
                "google": {
                    "api_key_env": "GOOGLE_API_KEY",
                    "timeout": 30
                },
                "groq": {
                    "api_key_env": "GROQ_API_KEY",
                    "timeout": 30,
                    "base_url": "https://api.groq.com/openai/v1"
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "timeout": 60
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    def _parse_models(self) -> Dict[str, ModelConfig]:
        """Parse models from config"""
        models = {}
        
        for model_id, model_data in self.config.get("models", {}).items():
            models[model_id] = ModelConfig(
                name=model_data.get("name", model_id),
                provider=model_data.get("provider", "unknown"),
                cost_per_1k_input=model_data.get("cost_per_1k_input", 0.001),
                cost_per_1k_output=model_data.get("cost_per_1k_output", 0.002),
                quality=model_data.get("quality", 5),
                speed=model_data.get("speed", "medium"),
                context_window=model_data.get("context_window", 4096),
                supports_streaming=model_data.get("supports_streaming", False),
                best_for=model_data.get("best_for", ["general"])
            )
            
        return models
    
    def get_models(self) -> Dict[str, ModelConfig]:
        """Get all models"""
        return self.models
    
    def get_routing_rules(self) -> Dict:
        """Get routing rules"""
        return self.config.get("routing_rules", {})
    
    def get_provider_settings(self) -> Dict:
        """Get provider settings"""
        return self.config.get("providers", {})

# ====================
# STREAMING CLIENT
# ====================

class StreamingLLMClient:
    """Production streaming implementation"""
    
    async def stream_openai(self, prompt: str, model: str) -> AsyncGenerator[str, None]:
        """Real OpenAI streaming"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'stream': True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        if line == 'data: [DONE]':
                            break
                        try:
                            data = json.loads(line[6:])
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
    
    async def stream_anthropic(self, prompt: str, model: str) -> AsyncGenerator[str, None]:
        """Real Anthropic streaming"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
        
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 1024,
            'stream': True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=payload
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if data.get('type') == 'content_block_delta':
                                yield data['delta']['text']
                        except:
                            continue
    
    async def stream_ollama(self, prompt: str, model: str) -> AsyncGenerator[str, None]:
        """Real Ollama streaming"""
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:11434/api/generate',
                json=payload
            ) as response:
                async for line in response.content:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            yield data['response']
                    except:
                        continue

# ====================
# QUALITY SCORER
# ====================

class AdvancedQualityScorer:
    """Production-grade response quality scoring"""
    
    def __init__(self):
        # Load NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Spacy model not loaded, installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        try:
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
        except:
            logger.warning("LanguageTool not available")
            self.grammar_tool = None
        
        # Scoring weights
        self.weights = {
            'relevance': 0.25,
            'completeness': 0.20,
            'coherence': 0.20,
            'grammar': 0.15,
            'clarity': 0.10,
            'factuality': 0.10
        }
        
        # Load embeddings model
        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            logger.warning("SentenceTransformer not available")
            self.embeddings_model = None
    
    def score(self, prompt: str, response: str, context: Dict = None) -> Tuple[float, Dict]:
        """Comprehensive quality scoring"""
        
        scores = {}
        details = {}
        
        # 1. RELEVANCE - Semantic similarity
        scores['relevance'], details['relevance'] = self._score_relevance(prompt, response)
        
        # 2. COMPLETENESS - Information coverage
        scores['completeness'], details['completeness'] = self._score_completeness(prompt, response)
        
        # 3. COHERENCE - Logical flow
        scores['coherence'], details['coherence'] = self._score_coherence(response)
        
        # 4. GRAMMAR - Language quality
        scores['grammar'], details['grammar'] = self._score_grammar(response)
        
        # 5. CLARITY - Readability
        scores['clarity'], details['clarity'] = self._score_clarity(response)
        
        # 6. FACTUALITY - Truthfulness
        scores['factuality'], details['factuality'] = self._score_factuality(response, context)
        
        # Calculate weighted score
        total_score = sum(scores[k] * self.weights[k] for k in self.weights)
        
        return total_score, {
            'total': total_score,
            'scores': scores,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
    
    def _score_relevance(self, prompt: str, response: str) -> Tuple[float, Dict]:
        """Score semantic relevance using embeddings"""
        
        if self.embeddings_model:
            # Get embeddings
            prompt_embedding = self.embeddings_model.encode(prompt, convert_to_tensor=True)
            response_embedding = self.embeddings_model.encode(response, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(prompt_embedding, response_embedding).item()
        else:
            # Fallback to word overlap
            prompt_words = set(prompt.lower().split())
            response_words = set(response.lower().split())
            overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
            similarity = min(1.0, overlap * 2)
        
        # Extract key concepts
        prompt_doc = self.nlp(prompt)
        response_doc = self.nlp(response)
        
        prompt_entities = set([ent.text.lower() for ent in prompt_doc.ents])
        response_entities = set([ent.text.lower() for ent in response_doc.ents])
        
        entity_overlap = len(prompt_entities & response_entities) / max(len(prompt_entities), 1) if prompt_entities else 0.5
        
        # Combined score
        score = (similarity * 0.7) + (entity_overlap * 0.3)
        
        return score, {
            'semantic_similarity': similarity,
            'entity_overlap': entity_overlap,
            'matched_entities': list(prompt_entities & response_entities)
        }
    
    def _score_completeness(self, prompt: str, response: str) -> Tuple[float, Dict]:
        """Score response completeness"""
        
        # Identify question type
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        prompt_lower = prompt.lower()
        
        question_types = [q for q in question_words if q in prompt_lower]
        
        # Check if response addresses each question type
        addressed = 0
        expected_patterns = {
            'what': ['is', 'are', 'means', 'refers to'],
            'why': ['because', 'due to', 'reason', 'since'],
            'how': ['by', 'through', 'using', 'with', 'step'],
            'when': ['at', 'on', 'in', 'during', 'time'],
            'where': ['at', 'in', 'on', 'location', 'place'],
            'who': ['person', 'people', 'individual', 'he', 'she', 'they'],
            'which': ['option', 'choice', 'alternative', 'either', 'or']
        }
        
        response_lower = response.lower()
        for q_type in question_types:
            if any(pattern in response_lower for pattern in expected_patterns.get(q_type, [])):
                addressed += 1
        
        completeness = addressed / max(len(question_types), 1) if question_types else 0.8
        
        # Check response length appropriateness
        expected_length = len(prompt.split()) * 3
        actual_length = len(response.split())
        length_ratio = min(actual_length / max(expected_length, 1), 1.0)
        
        # Check for structured response
        has_structure = any(marker in response for marker in ['1.', '2.', '•', '-', '\n\n'])
        structure_bonus = 0.1 if has_structure else 0
        
        score = (completeness * 0.5) + (length_ratio * 0.4) + structure_bonus
        
        return min(score, 1.0), {
            'question_types': question_types,
            'addressed': addressed,
            'length_ratio': length_ratio,
            'has_structure': has_structure
        }
    
    def _score_coherence(self, response: str) -> Tuple[float, Dict]:
        """Score logical coherence"""
        
        doc = self.nlp(response)
        sentences = list(doc.sents)
        
        if len(sentences) < 2:
            return 0.5, {'sentence_count': len(sentences)}
        
        # Check transition words
        transition_words = {
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'contrast': ['however', 'but', 'although', 'despite', 'nevertheless'],
            'cause': ['because', 'since', 'due to', 'as a result', 'therefore'],
            'sequence': ['first', 'second', 'then', 'finally', 'next'],
            'example': ['for example', 'such as', 'namely', 'specifically', 'including']
        }
        
        transitions_found = []
        for category, words in transition_words.items():
            for word in words:
                if word in response.lower():
                    transitions_found.append((category, word))
        
        transition_score = min(len(transitions_found) / 3, 1.0)
        
        # Check topic consistency
        entity_consistency = []
        for i in range(len(sentences) - 1):
            sent1_ents = set([ent.text for ent in sentences[i].ents])
            sent2_ents = set([ent.text for ent in sentences[i+1].ents])
            if sent1_ents and sent2_ents:
                overlap = len(sent1_ents & sent2_ents) / max(len(sent1_ents), 1)
                entity_consistency.append(overlap)
        
        consistency_score = np.mean(entity_consistency) if entity_consistency else 0.5
        
        # Check logical connectors
        logical_connectors = ['therefore', 'thus', 'hence', 'consequently', 'so']
        logic_score = min(sum(1 for conn in logical_connectors if conn in response.lower()) * 0.2, 1.0)
        
        score = (transition_score * 0.4) + (consistency_score * 0.4) + (logic_score * 0.2)
        
        return min(score, 1.0), {
            'transitions': transitions_found[:5],  # Limit to 5 for readability
            'entity_consistency': consistency_score,
            'logical_flow': logic_score
        }
    
    def _score_grammar(self, response: str) -> Tuple[float, Dict]:
        """Score grammatical correctness"""
        
        if self.grammar_tool:
            # Check with LanguageTool
            matches = self.grammar_tool.check(response)
            
            # Categorize errors
            errors = {'spelling': 0, 'grammar': 0, 'style': 0}
            for match in matches:
                if 'SPELL' in match.ruleId.upper():
                    errors['spelling'] += 1
                elif 'GRAMMAR' in match.ruleId.upper():
                    errors['grammar'] += 1
                else:
                    errors['style'] += 1
            
            # Calculate score
            word_count = len(response.split())
            error_rate = sum(errors.values()) / max(word_count, 1)
            score = max(0, 1 - (error_rate * 10))
            
            details = {
                'errors': errors,
                'total_errors': sum(errors.values()),
                'error_rate': error_rate,
                'suggestions': [match.message for match in matches[:3]]
            }
        else:
            # Fallback: basic checks
            score = 0.8  # Default score
            details = {'note': 'Grammar tool not available'}
        
        return score, details
    
    def _score_clarity(self, response: str) -> Tuple[float, Dict]:
        """Score readability and clarity"""
        
        # Calculate readability scores
        try:
            flesch_reading = textstat.flesch_reading_ease(response)
            gunning_fog = textstat.gunning_fog(response)
        except:
            flesch_reading = 60
            gunning_fog = 10
        
        # Convert to 0-1 scale
        flesch_score = max(0, min(1, (flesch_reading - 30) / 70))
        fog_score = max(0, 1 - ((gunning_fog - 6) / 12))
        
        # Check sentence length variation
        doc = self.nlp(response)
        sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
        
        if sentence_lengths:
            avg_length = np.mean(sentence_lengths)
            std_length = np.std(sentence_lengths)
            variation_score = min(std_length / max(avg_length, 1), 1.0)
        else:
            avg_length = 0
            variation_score = 0.5
        
        score = (flesch_score * 0.4) + (fog_score * 0.4) + (variation_score * 0.2)
        
        return score, {
            'flesch_reading_ease': flesch_reading,
            'gunning_fog': gunning_fog,
            'avg_sentence_length': avg_length,
            'sentence_variation': variation_score
        }
    
    def _score_factuality(self, response: str, context: Dict = None) -> Tuple[float, Dict]:
        """Score factual accuracy"""
        
        # Check for hedging language
        hedging_words = ['might', 'maybe', 'possibly', 'could be', 'perhaps', 
                        'it seems', 'appears to', 'likely', 'probably']
        hedging_count = sum(1 for word in hedging_words if word in response.lower())
        hedging_penalty = min(hedging_count * 0.1, 0.3)
        
        # Extract claims
        doc = self.nlp(response)
        
        # Extract numerical claims
        numbers = [token.text for token in doc if token.like_num]
        
        # Extract date claims
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        
        # Base factuality score
        base_score = 0.7
        
        # If context provided, verify claims
        verification_score = 1.0
        if context and 'facts' in context:
            known_facts = context['facts']
            for num in numbers:
                if num in known_facts and known_facts[num] == False:
                    verification_score -= 0.2
        
        score = (base_score * 0.6) + (verification_score * 0.4) - hedging_penalty
        
        return max(0, min(score, 1.0)), {
            'hedging_words': hedging_count,
            'numerical_claims': numbers[:5],
            'date_claims': dates[:5],
            'verification_score': verification_score
        }

# ====================
# MODEL BENCHMARKER
# ====================

class ProductionBenchmarker:
    """Production-grade model benchmarking"""
    
    def __init__(self, comprehensive: bool = False):
        self.comprehensive = comprehensive
        self.results: List[BenchmarkResult] = []
        self.quality_scorer = AdvancedQualityScorer()
        
        # Benchmark suites
        self.quick_suite = [
            {
                "prompt": "What is 2+2?",
                "type": "math_simple",
                "expected": "4",
                "max_tokens": 10
            },
            {
                "prompt": "Translate 'Hello' to French",
                "type": "translation",
                "expected": "Bonjour",
                "max_tokens": 20
            },
            {
                "prompt": "Complete: The capital of France is",
                "type": "factual",
                "expected": "Paris",
                "max_tokens": 10
            }
        ]
        
        self.comprehensive_suite = [
            # Reasoning
            {
                "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "type": "reasoning",
                "expected": "$0.05",
                "max_tokens": 100
            },
            # Coding
            {
                "prompt": "Write a Python function to calculate fibonacci numbers recursively",
                "type": "coding",
                "expected_keywords": ["def", "fibonacci", "return", "if", "else"],
                "max_tokens": 200
            },
            # Creative
            {
                "prompt": "Write a haiku about artificial intelligence",
                "type": "creative",
                "expected_format": "three_lines",
                "max_tokens": 50
            },
            # Analysis
            {
                "prompt": "Analyze the pros and cons of renewable energy",
                "type": "analysis",
                "expected_keywords": ["solar", "wind", "cost", "environment"],
                "max_tokens": 300
            },
            # Summarization
            {
                "prompt": "Summarize in one sentence: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
                "type": "summarization",
                "max_tokens": 50
            },
            # Instruction following
            {
                "prompt": "List exactly 3 benefits of exercise. Use bullet points.",
                "type": "instruction",
                "expected_format": "bullet_list",
                "expected_count": 3,
                "max_tokens": 100
            }
        ]
    
    async def benchmark_model(self, model_id: str, provider: str, client: Any, comprehensive: bool = None) -> Dict:
        """Benchmark a single model"""
        
        if comprehensive is None:
            comprehensive = self.comprehensive
        
        suite = self.comprehensive_suite if comprehensive else self.quick_suite
        results = []
        
        for test in suite:
            result = await self._run_single_test(model_id, provider, client, test)
            results.append(result)
            self.results.append(result)
        
        # Aggregate results
        return self._aggregate_results(model_id, results)
    
    async def _run_single_test(self, model_id: str, provider: str, client: Any, test: Dict) -> BenchmarkResult:
        """Run single benchmark test"""
        
        try:
            # Measure latency
            start_time = time.time()
            
            # Make actual API call
            response = await client._call_model(model_id, test['prompt'], provider)
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Count tokens
            token_counter = TokenCounter()
            prompt_tokens = token_counter.count_tokens(test['prompt'], model_id)
            response_tokens = token_counter.count_tokens(response, model_id)
            
            # Calculate tokens per second
            tps = response_tokens / latency if latency > 0 else 0
            
            # Calculate cost
            cost = client._calculate_actual_cost(model_id, prompt_tokens, response_tokens)
            
            # Score quality
            quality_score, quality_details = self._score_response(test, response)
            
            return BenchmarkResult(
                model_id=model_id,
                prompt_type=test['type'],
                latency=latency,
                tokens_per_second=tps,
                quality_score=quality_score,
                cost=cost,
                response_sample=response[:200],
                error=None
            )
            
        except Exception as e:
            return BenchmarkResult(
                model_id=model_id,
                prompt_type=test['type'],
                latency=0,
                tokens_per_second=0,
                quality_score=0,
                cost=0,
                error=str(e)
            )
    
    def _score_response(self, test: Dict, response: str) -> Tuple[float, Dict]:
        """Score response quality for benchmark"""
        
        score = 0.0
        details = {}
        
        # Check expected answer
        if 'expected' in test:
            if test['expected'].lower() in response.lower():
                score += 0.5
                details['exact_match'] = True
            else:
                details['exact_match'] = False
        
        # Check expected keywords
        if 'expected_keywords' in test:
            keywords_found = sum(1 for kw in test['expected_keywords'] 
                               if kw.lower() in response.lower())
            keyword_score = keywords_found / len(test['expected_keywords'])
            score += keyword_score * 0.3
            details['keywords_found'] = keywords_found
        
        # Check format
        if 'expected_format' in test:
            format_score = self._check_format(response, test['expected_format'])
            score += format_score * 0.2
            details['format_correct'] = format_score > 0.5
        
        # General quality score
        general_score, _ = self.quality_scorer.score(test['prompt'], response)
        score = (score * 0.7) + (general_score * 0.3)
        
        return min(score, 1.0), details
    
    def _check_format(self, response: str, expected_format: str) -> float:
        """Check if response matches expected format"""
        
        if expected_format == "bullet_list":
            bullets = ['•', '-', '*', '1.', '2.', '3.']
            return 1.0 if any(b in response for b in bullets) else 0.0
            
        elif expected_format == "three_lines":
            lines = response.strip().split('\n')
            return 1.0 if len(lines) == 3 else 0.0
            
        return 0.5
    
    def _aggregate_results(self, model_id: str, results: List[BenchmarkResult]) -> Dict:
        """Aggregate benchmark results"""
        
        valid_results = [r for r in results if r.error is None]
        
        if not valid_results:
            return {
                'model_id': model_id,
                'status': 'failed',
                'errors': [r.error for r in results if r.error]
            }
        
        return {
            'model_id': model_id,
            'status': 'success',
            'metrics': {
                'avg_latency': np.mean([r.latency for r in valid_results]),
                'p95_latency': np.percentile([r.latency for r in valid_results], 95),
                'avg_tps': np.mean([r.tokens_per_second for r in valid_results]),
                'avg_quality': np.mean([r.quality_score for r in valid_results]),
                'total_cost': sum(r.cost for r in valid_results),
                'success_rate': len(valid_results) / len(results)
            },
            'by_type': self._aggregate_by_type(valid_results),
            'timestamp': datetime.now().isoformat()
        }
    
    def _aggregate_by_type(self, results: List[BenchmarkResult]) -> Dict:
        """Aggregate results by prompt type"""
        
        by_type = defaultdict(list)
        for r in results:
            by_type[r.prompt_type].append(r)
        
        aggregated = {}
        for prompt_type, type_results in by_type.items():
            aggregated[prompt_type] = {
                'avg_latency': np.mean([r.latency for r in type_results]),
                'avg_quality': np.mean([r.quality_score for r in type_results]),
                'avg_tps': np.mean([r.tokens_per_second for r in type_results])
            }
        
        return aggregated
    
    async def benchmark_all_models(self, models: Dict[str, ModelConfig], client: Any) -> pd.DataFrame:
        """Benchmark all models and return comparison"""
        
        all_results = []
        
        # Run benchmarks sequentially to avoid rate limits
        for model_id, config in models.items():
            logger.info(f"Benchmarking {model_id}...")
            try:
                result = await self.benchmark_model(model_id, config.provider, client,self.comprehensive)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to benchmark {model_id}: {e}")
        
        # Create comparison dataframe
        comparison_data = []
        for result in all_results:
            if isinstance(result, dict) and result.get('status') == 'success':
                comparison_data.append({
                    'Model': result['model_id'],
                    'Avg Latency (s)': result['metrics']['avg_latency'],
                    'P95 Latency (s)': result['metrics']['p95_latency'],
                    'Tokens/Second': result['metrics']['avg_tps'],
                    'Quality Score': result['metrics']['avg_quality'],
                    'Total Cost': result['metrics']['total_cost'],
                    'Success Rate': result['metrics']['success_rate']
                })
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Add rankings
        df['Latency Rank'] = df['Avg Latency (s)'].rank()
        df['Speed Rank'] = df['Tokens/Second'].rank(ascending=False)
        df['Quality Rank'] = df['Quality Score'].rank(ascending=False)
        df['Cost Rank'] = df['Total Cost'].rank()
        
        # Overall score (lower is better)
        df['Overall Score'] = (
            df['Latency Rank'] * 0.2 +
            df['Speed Rank'] * 0.2 +
            df['Quality Rank'] * 0.4 +
            df['Cost Rank'] * 0.2
        )
        
        return df.sort_values('Overall Score')

# ====================
# A/B TEST MANAGER
# ====================

class ProductionABTestManager:
    """Production A/B testing with statistical significance"""
    
    def __init__(self, min_sample_size: int = 100, confidence_level: float = 0.95):
        self.min_sample_size = min_sample_size
        self.confidence_level = confidence_level
        self.active_tests = {}
        self.test_results = defaultdict(lambda: {
            'variant_a': {'responses': [], 'metrics': []},
            'variant_b': {'responses': [], 'metrics': []}
        })
        
        # Persistence
        self.db_path = "ab_tests.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                name TEXT,
                model_a TEXT,
                model_b TEXT,
                created_at TIMESTAMP,
                status TEXT,
                allocation_ratio REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT,
                variant TEXT,
                prompt TEXT,
                response TEXT,
                latency REAL,
                quality_score REAL,
                cost REAL,
                timestamp TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES ab_tests(test_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_test(self, 
                   name: str,
                   model_a: str, 
                   model_b: str,
                   allocation_ratio: float = 0.5,
                   metadata: Dict = None) -> str:
        """Create new A/B test"""
        
        test_id = str(uuid.uuid4())
        
        self.active_tests[test_id] = {
            'name': name,
            'model_a': model_a,
            'model_b': model_b,
            'allocation_ratio': allocation_ratio,
            'created_at': datetime.now(),
            'status': 'active',
            'metadata': metadata or {}
        }
        
        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ab_tests (test_id, name, model_a, model_b, created_at, status, allocation_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (test_id, name, model_a, model_b, datetime.now(), 'active', allocation_ratio))
        
        conn.commit()
        conn.close()
        
        return test_id
    
    def select_variant(self, test_id: str, user_id: Optional[str] = None) -> str:
        """Select variant for user (consistent if user_id provided)"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        
        if user_id:
            # Consistent assignment based on user_id
            hash_value = int(hashlib.md5(f"{test_id}:{user_id}".encode()).hexdigest(), 16)
            assignment = (hash_value % 100) / 100.0
        else:
            # Random assignment
            assignment = np.random.random()
        
        if assignment < test.get('allocation_ratio', 0.5):
            return test['model_a']
        else:
            return test['model_b']
    
    def record_result(self,
                     test_id: str,
                     variant_model: str,
                     prompt: str,
                     response: str,
                     latency: float,
                     quality_score: float,
                     cost: float):
        """Record test result"""
        
        if test_id not in self.active_tests:
            return
        
        test = self.active_tests[test_id]
        
        # Determine variant
        variant = 'variant_a' if variant_model == test['model_a'] else 'variant_b'
        
        # Store in memory
        self.test_results[test_id][variant]['responses'].append({
            'prompt': prompt,
            'response': response,
            'timestamp': datetime.now()
        })
        
        self.test_results[test_id][variant]['metrics'].append({
            'latency': latency,
            'quality_score': quality_score,
            'cost': cost
        })
        
        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ab_results (test_id, variant, prompt, response, latency, quality_score, cost, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (test_id, variant, prompt, response, latency, quality_score, cost, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def analyze_test(self, test_id: str) -> Dict:
        """Analyze test results with statistical significance"""
        
        if test_id not in self.test_results:
            return {'error': 'No results for test'}
        
        results = self.test_results[test_id]
        
        # Extract metrics
        a_metrics = results['variant_a']['metrics']
        b_metrics = results['variant_b']['metrics']
        
        if len(a_metrics) < self.min_sample_size or len(b_metrics) < self.min_sample_size:
            return {
                'status': 'insufficient_data',
                'samples_a': len(a_metrics),
                'samples_b': len(b_metrics),
                'required': self.min_sample_size
            }
        
        # Calculate statistics for each metric
        analysis = {
            'test_id': test_id,
            'samples_a': len(a_metrics),
            'samples_b': len(b_metrics),
            'metrics': {}
        }
        
        # Latency analysis
        a_latency = [m['latency'] for m in a_metrics]
        b_latency = [m['latency'] for m in b_metrics]
        analysis['metrics']['latency'] = self._compare_metrics(a_latency, b_latency, lower_is_better=True)
        
        # Quality analysis
        a_quality = [m['quality_score'] for m in a_metrics]
        b_quality = [m['quality_score'] for m in b_metrics]
        analysis['metrics']['quality'] = self._compare_metrics(a_quality, b_quality, lower_is_better=False)
        
        # Cost analysis
        a_cost = [m['cost'] for m in a_metrics]
        b_cost = [m['cost'] for m in b_metrics]
        analysis['metrics']['cost'] = self._compare_metrics(a_cost, b_cost, lower_is_better=True)
        
        # Overall winner
        analysis['winner'] = self._determine_winner(analysis['metrics'])
        
        # Calculate statistical power
        analysis['statistical_power'] = self._calculate_power(a_metrics, b_metrics)
        
        return analysis
    
    def _compare_metrics(self, a_values: List[float], b_values: List[float], 
                        lower_is_better: bool = False) -> Dict:
        """Compare two sets of metrics with statistical tests"""
        
        # Basic statistics
        result = {
            'mean_a': np.mean(a_values),
            'mean_b': np.mean(b_values),
            'std_a': np.std(a_values),
            'std_b': np.std(b_values),
            'median_a': np.median(a_values),
            'median_b': np.median(b_values)
        }
        
        # T-test for significance
        t_stat, p_value = stats.ttest_ind(a_values, b_values)
        result['t_statistic'] = t_stat
        result['p_value'] = p_value
        result['significant'] = p_value < (1 - self.confidence_level)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(a_values)**2 + np.std(b_values)**2) / 2)
        effect_size = (result['mean_a'] - result['mean_b']) / pooled_std if pooled_std > 0 else 0
        result['effect_size'] = abs(effect_size)
        
        # Determine better variant
        if lower_is_better:
            result['better_variant'] = 'a' if result['mean_a'] < result['mean_b'] else 'b'
            result['improvement'] = (result['mean_b'] - result['mean_a']) / result['mean_b'] * 100
        else:
            result['better_variant'] = 'a' if result['mean_a'] > result['mean_b'] else 'b'
            result['improvement'] = (result['mean_a'] - result['mean_b']) / result['mean_b'] * 100
        
        # Confidence interval
        ci = stats.t.interval(
            self.confidence_level,
            len(a_values) + len(b_values) - 2,
            loc=result['mean_a'] - result['mean_b'],
            scale=pooled_std * np.sqrt(1/len(a_values) + 1/len(b_values))
        )
        result['confidence_interval'] = ci
        
        return result
    
    def _determine_winner(self, metrics: Dict) -> str:
        """Determine overall winner based on all metrics"""
        
        scores = {'a': 0, 'b': 0}
        
        # Weight different metrics
        weights = {
            'latency': 0.3,
            'quality': 0.5,
            'cost': 0.2
        }
        
        for metric_name, metric_data in metrics.items():
            if metric_data['significant']:
                winner = metric_data['better_variant']
                scores[winner] += weights.get(metric_name, 0.33)
        
        if scores['a'] > scores['b']:
            return 'variant_a'
        elif scores['b'] > scores['a']:
            return 'variant_b'
        else:
            return 'no_winner'
    
    def _calculate_power(self, a_metrics: List, b_metrics: List) -> float:
        """Calculate statistical power of the test"""
        
        # Simplified power calculation
        n = min(len(a_metrics), len(b_metrics))
        
        # Power increases with sample size
        power = 1 - (1 / np.sqrt(n))
        
        return min(power, 0.99)

# ====================
# SMART ROUTER
# ====================

class SmartRouter:
    """Advanced routing with all features"""
    
    def __init__(self, config_path: str = "models.yaml", verbose: bool = True):
        self.verbose = verbose
        self.config = ConfigLoader(config_path)
        self.models = self.config.get_models()
        self.routing_rules = self.config.get_routing_rules()
        
        # Advanced components
        self.token_counter = TokenCounter()
        self.metrics = MetricsCollector()
        
        # Stats
        self.stats = {
            "total_requests": 0,
            "total_cost": 0.0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "model_usage": defaultdict(int),
            "avg_quality": 0.0
        }
        
        if self.verbose:
            logger.info(f"✅ Loaded {len(self.models)} models from {config_path}")
    
    def evaluate_complexity(self, prompt: str) -> Tuple[int, str]:
        """Hybrid: length + linguistic features"""
        
        length = len(prompt)
        words = prompt.split()
        
        # Fast path for obvious cases
        if length <= 10:
            return 1, "very short"
        if length > 500:
            return 10, "very long"
        
        # Calculate lexical diversity
        unique_ratio = len(set(words)) / max(len(words), 1)
        
        # Base complexity from length
        if length <= 30:
            base = 2
        elif length <= 60:
            base = 4
        elif length <= 120:
            base = 6
        elif length <= 250:
            base = 8
        else:
            base = 9
        
        # Adjust for lexical diversity
        if unique_ratio > 0.9:  # Very diverse = technical
            base += 2
        elif unique_ratio > 0.7:
            base += 1
        elif unique_ratio < 0.3:  # Very repetitive
            base -= 1
        
        # Check for complexity indicators
        complexity_indicators = [
            'analyze', 'explain', 'compare', 'evaluate', 'design',
            'implement', 'optimize', 'debug', 'research', 'prove'
        ]
        
        prompt_lower = prompt.lower()
        if any(indicator in prompt_lower for indicator in complexity_indicators):
            base += 1
        
        # Check for code markers
        code_markers = ['```', 'def ', 'function', 'class ', 'import ', 'return ']
        if any(marker in prompt for marker in code_markers):
            base += 1
        
        complexity = max(1, min(10, base))
        
        return complexity, f"{length}ch, {unique_ratio:.2f}div"
    
    def select_model(self, complexity: int, prompt: str, prefer_local: bool = False) -> str:
        """Select the RIGHT SIZE model for the task complexity"""
        
        available_models = self._get_available_models()
        
        if not available_models:
            raise Exception("No models available!")
        
        # Prefer local if requested and available
        if prefer_local:
            local_models = {m: c for m, c in available_models.items() if c.cost_per_1k_input == 0}
            if local_models:
                # Find best quality local model
                return max(local_models.items(), key=lambda x: x[1].quality)[0]
        
        # Find model with quality closest to complexity
        ideal_quality = complexity
        
        # Find best match
        best_model = None
        best_score = float('-inf')
        
        for model_id, config in available_models.items():
            # Base score: inverse of quality distance
            quality_distance = abs(config.quality - ideal_quality)
            score = 10 - quality_distance
            
            # Bonus for exact match
            if quality_distance == 0:
                score += 5
            
            # Penalty for overkill
            if config.quality > ideal_quality + 3:
                score -= 10
            
            # Penalty for underpowered
            if config.quality < ideal_quality - 2:
                score -= 5
            
            # Slight preference for free models
            if config.cost_per_1k_input == 0:
                score += 1
            
            # Speed bonus for simple tasks
            if complexity <= 3 and config.speed in ['fast', 'very_fast']:
                score += 2
            
            # Context window check
            prompt_tokens = self.token_counter.count_tokens(prompt)
            if prompt_tokens > config.context_window:
                score -= 20  # Heavy penalty if prompt doesn't fit
            
            if score > best_score:
                best_score = score
                best_model = model_id
        
        return best_model
    
    def _get_available_models(self) -> Dict[str, ModelConfig]:
        """Get available models based on configured providers"""
        available = {}
        
        for model_id, config in self.models.items():
            provider_settings = self.config.get_provider_settings().get(config.provider, {})
            
            # Check provider availability
            if config.provider in ["openai", "anthropic", "google", "groq"]:
                api_key_env = provider_settings.get("api_key_env")
                if api_key_env and not os.getenv(api_key_env):
                    continue
            elif config.provider == "ollama":
                if not self._check_ollama():
                    continue
            
            available[model_id] = config
        
        return available
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def route(self, prompt: str, prefer_local: bool = False) -> RoutingResult:
        """Route request with full metrics"""
        
        # Evaluate complexity
        complexity, reason = self.evaluate_complexity(prompt)
        
        # Count input tokens
        input_tokens = self.token_counter.count_tokens(prompt)
        
        # Estimate output tokens
        output_tokens = min(input_tokens * 3, 2000)
        
        # Select model
        model_id = self.select_model(complexity, prompt, prefer_local)
        model_config = self.models[model_id]
        
        # Calculate cost
        cost = ((input_tokens * model_config.cost_per_1k_input + 
                output_tokens * model_config.cost_per_1k_output) / 1000)
        
        # Create result
        result = RoutingResult(
            model_id=model_id,
            model_name=model_config.name,
            complexity=complexity,
            reason=reason,
            estimated_tokens_input=input_tokens,
            estimated_tokens_output=output_tokens,
            estimated_cost=cost,
            is_free=model_config.cost_per_1k_input == 0,
            timestamp=datetime.now().isoformat()
        )
        
        # Update stats
        self.stats["total_requests"] += 1
        self.stats["total_cost"] += cost
        self.stats["total_tokens_input"] += input_tokens
        self.stats["total_tokens_output"] += output_tokens
        self.stats["model_usage"][model_id] += 1
        
        if self.verbose:
            self._print_routing_decision(result)
        
        return result
    
    def _print_routing_decision(self, result: RoutingResult):
        """Print routing decision"""
        print(f"\n{'='*50}")
        print(f"🎯 Routing Decision")
        print(f"{'='*50}")
        print(f"📊 Complexity: {result.complexity}/10 ({result.reason})")
        print(f"🤖 Selected: {result.model_name}")
        print(f"📝 Tokens: {result.estimated_tokens_input} in / {result.estimated_tokens_output} out")
        print(f"💰 Cost: {'FREE' if result.is_free else f'${result.estimated_cost:.4f}'}")
        print(f"{'='*50}")
    
    def get_stats(self) -> Dict:
        """Get routing statistics"""
        return self.stats


# ====================
# RESILIENT CLIENT
# ====================

class ResilientLLMClient:
    """Production client with fallbacks, caching, and circuit breakers"""
    
    def __init__(self, router: SmartRouter, 
                 cache_ttl: int = 3600,
                 max_retries: int = 3):
        self.router = router
        self.max_retries = max_retries
        self.streaming_client = StreamingLLMClient()
        
        # Initialize per-provider circuit breakers
        self.circuit_breakers = {}
        for model_id, config in router.models.items():
            provider = config.provider
            if provider not in self.circuit_breakers:
                self.circuit_breakers[provider] = CircuitBreaker(
                    failure_threshold=5,
                    recovery_timeout=60
                )
        
        # Response cache
        self.cache = LRUCache(maxsize=1000, ttl=cache_ttl)
        
        # Quality scorer for feedback
        self.quality_scorer = AdvancedQualityScorer()
        
        # A/B test manager
        self.ab_test_manager = None
    
    def set_ab_test_manager(self, manager: ProductionABTestManager):
        """Set A/B test manager"""
        self.ab_test_manager = manager
    
    async def chat(self, prompt: str, 
              stream: bool = False,
              prefer_local: bool = False,
              max_cost: Optional[float] = None,
              use_cache: bool = True,
              ab_test_id: Optional[str] = None,
              user_id: Optional[str] = None) -> Union[str, AsyncGenerator[str, None]]:
        """Main chat interface with all features"""
        
        # Check cache first
        cache_key = hashlib.md5(f"{prompt}:{prefer_local}:{max_cost}".encode()).hexdigest()
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                if stream:
                    async def cached_stream():
                        yield cached
                    return cached_stream()
                return cached
        
        # A/B testing variant selection
        if ab_test_id and self.ab_test_manager:
            model_id = self.ab_test_manager.select_variant(ab_test_id, user_id)
            routing = RoutingResult(
                model_id=model_id,
                model_name=self.router.models[model_id].name,
                complexity=5,  # Default for A/B test
                reason="A/B test variant",
                estimated_tokens_input=self.router.token_counter.count_tokens(prompt),
                estimated_tokens_output=500,
                estimated_cost=0,
                is_free=self.router.models[model_id].cost_per_1k_input == 0,
                timestamp=datetime.now().isoformat()
            )
        else:
            # Normal routing
            routing = self.router.route(prompt, prefer_local)
        
        # Apply cost limit if specified
        if max_cost and routing.estimated_cost > max_cost:
            # Find cheaper alternative
            routing = self._find_cheaper_alternative(prompt, max_cost)
            if not routing:
                raise ValueError(f"No model available within budget of ${max_cost}")
        
        # Build fallback chain
        fallback_chain = self._build_fallback_chain(routing.model_id, routing.complexity)
        
        # Try each model in chain
        last_error = None
        start_time = time.time()
        
        for model_id in fallback_chain:
            model_config = self.router.models[model_id]
            provider = model_config.provider
            
            # Check circuit breaker
            if self.circuit_breakers[provider].state == 'open':
                continue
            
            try:
                # Make the actual call
                if stream and model_config.supports_streaming:
                    # FIX: NON fare await del generator
                    response_gen = self._call_model_streaming(model_id, prompt, provider)
                    
                    # Wrap generator to cache full response
                    if use_cache:
                        response_gen = self._cache_streaming_response(response_gen, cache_key)
                    
                    return response_gen
                else:
                    response = await self._call_model_with_retry(
                        model_id, prompt, provider
                    )
                    
                    # Cache successful response
                    if use_cache:
                        self.cache.set(cache_key, response)
                    
                    # Record metrics
                    latency = time.time() - start_time
                    actual_tokens_out = self.router.token_counter.count_tokens(response)
                    actual_cost = self._calculate_actual_cost(
                        model_id, routing.estimated_tokens_input, actual_tokens_out
                    )
                    
                    # Quality scoring
                    quality_score, quality_details = self.quality_scorer.score(
                        prompt, response
                    )
                    
                    # Record A/B test result if applicable
                    if ab_test_id and self.ab_test_manager:
                        self.ab_test_manager.record_result(
                            ab_test_id, model_id, prompt, response,
                            latency, quality_score, actual_cost
                        )
                    
                    # Record metrics
                    self.router.metrics.record_request(
                        model=model_id,
                        provider=provider,
                        tokens_in=routing.estimated_tokens_input,
                        tokens_out=actual_tokens_out,
                        latency=latency,
                        cost=actual_cost
                    )
                    
                    return response
                    
            except Exception as e:
                last_error = e
                await self.circuit_breakers[provider]._on_failure()  # FIX: await aggiunto
                logging.warning(f"Model {model_id} failed: {e}")
                continue
        
        # All models failed
        raise Exception(f"All models failed. Last error: {last_error}")
    
    async def _call_model_with_retry(self, model_id: str, prompt: str, provider: str) -> str:
        """Call model with exponential backoff retry"""
        
        for attempt in range(self.max_retries):
            try:
                cb = self.circuit_breakers[provider]
                response = await cb.call(
                    self._call_model, model_id, prompt, provider
                )
                return response
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
    
    async def _call_model(self, model_id: str, prompt: str, provider: str) -> str:
        """Make actual API call to model"""
        
        if provider == "openai":
            return await self._call_openai(model_id, prompt)
        elif provider == "anthropic":
            return await self._call_anthropic(model_id, prompt)
        elif provider == "google":
            return await self._call_google(model_id, prompt)
        elif provider == "groq":
            return await self._call_groq(model_id, prompt)
        elif provider == "ollama":
            return await self._call_ollama(model_id, prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _call_model_streaming(self, model_id: str, prompt: str, provider: str) -> AsyncGenerator[str, None]:
        """Stream response from model - FIX: NON async"""
        
        if provider == "openai":
            return self.streaming_client.stream_openai(prompt, model_id)
        elif provider == "anthropic":
            return self.streaming_client.stream_anthropic(prompt, model_id)
        elif provider == "ollama":
            return self.streaming_client.stream_ollama(prompt, model_id)
        else:
            # Fallback to non-streaming
            async def fallback_stream():
                response = await self._call_model(model_id, prompt, provider)
                yield response
            
            return fallback_stream()
    
    def _cache_streaming_response(self, generator: AsyncGenerator[str, None], 
                             cache_key: str) -> AsyncGenerator[str, None]:
        """Cache streaming response while yielding - FIX: NON async"""
        
        async def cached_generator():
            full_response = []
            async for chunk in generator:
                full_response.append(chunk)
                yield chunk
            
            # Cache the complete response
            self.cache.set(cache_key, ''.join(full_response))
        
        return cached_generator()
    
    async def _call_openai(self, model: str, prompt: str) -> str:
        """OpenAI API call"""
        import aiohttp
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_data = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_data}")
    
    async def _call_anthropic(self, model: str, prompt: str) -> str:
        """Anthropic API call"""
        import aiohttp
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
        
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 2000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['content'][0]['text']
                else:
                    error_data = await response.text()
                    raise Exception(f"Anthropic API error {response.status}: {error_data}")
    
    async def _call_google(self, model: str, prompt: str) -> str:
        """Google Generative AI API call"""
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        genai.configure(api_key=api_key)
        
        model_instance = genai.GenerativeModel(model)
        response = await model_instance.generate_content_async(prompt)
        
        return response.text
    
    async def _call_groq(self, model: str, prompt: str) -> str:
        """Groq API call"""
        import aiohttp
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_data = await response.text()
                    raise Exception(f"Groq API error {response.status}: {error_data}")
    
    async def _call_ollama(self, model: str, prompt: str) -> str:
        """Ollama local API call"""
        import aiohttp
        
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:11434/api/generate',
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['response']
                else:
                    raise Exception(f"Ollama API error {response.status}")
    
    def _build_fallback_chain(self, primary_model: str, complexity: int) -> List[str]:
        """Build intelligent fallback chain"""
        
        primary_config = self.router.models[primary_model]
        fallback_candidates = []
        
        for model_id, config in self.router.models.items():
            if model_id == primary_model:
                continue
            
            # Skip if provider circuit is open
            if self.circuit_breakers[config.provider].state == 'open':
                continue
            
            # Calculate similarity score
            similarity = 0
            
            # Quality match (prefer similar quality)
            quality_diff = abs(config.quality - primary_config.quality)
            similarity += (10 - quality_diff) * 0.3
            
            # Same provider (familiar API)
            if config.provider == primary_config.provider:
                similarity += 3
            
            # Capability overlap
            capability_overlap = len(set(config.best_for) & set(primary_config.best_for))
            similarity += capability_overlap * 0.5
            
            # Cost consideration (slight preference for cheaper)
            if config.cost_per_1k_input < primary_config.cost_per_1k_input:
                similarity += 1
            
            fallback_candidates.append((model_id, similarity))
        
        # Sort by similarity
        fallback_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Build chain: primary + top 3 alternatives
        chain = [primary_model]
        chain.extend([model_id for model_id, _ in fallback_candidates[:3]])
        
        return chain
    
    def _find_cheaper_alternative(self, prompt: str, max_cost: float) -> Optional[RoutingResult]:
        """Find alternative model within budget"""
        
        complexity, _ = self.router.evaluate_complexity(prompt)
        input_tokens = self.router.token_counter.count_tokens(prompt)
        output_tokens = min(input_tokens * 3, 2000)
        
        # Find all models within budget
        viable_models = []
        
        for model_id, config in self.router.models.items():
            estimated_cost = ((input_tokens * config.cost_per_1k_input + 
                             output_tokens * config.cost_per_1k_output) / 1000)
            
            if estimated_cost <= max_cost:
                # Check if quality is reasonable for complexity
                if config.quality >= complexity - 2:
                    viable_models.append((model_id, config, estimated_cost))
        
        if not viable_models:
            return None
        
        # Choose best quality within budget
        best_model = max(viable_models, key=lambda x: x[1].quality)
        model_id, config, cost = best_model
        
        return RoutingResult(
            model_id=model_id,
            model_name=config.name,
            complexity=complexity,
            reason="budget_constraint",
            estimated_tokens_input=input_tokens,
            estimated_tokens_output=output_tokens,
            estimated_cost=cost,
            is_free=config.cost_per_1k_input == 0,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_actual_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate actual cost"""
        config = self.router.models[model_id]
        return ((input_tokens * config.cost_per_1k_input + 
                output_tokens * config.cost_per_1k_output) / 1000)

# ====================
# CIRCUIT BREAKER
# ====================

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        async with self._lock:
            if self.state == 'open':
                if self._should_attempt_reset():
                    self.state = 'half-open'
                else:
                    raise Exception(f"Circuit breaker is open (failures: {self.failure_count})")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.failure_count = 0
            self.state = 'closed'
    
    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logging.warning(f"Circuit breaker opened after {self.failure_count} failures")

# ====================
# LRU CACHE
# ====================

class LRUCache:
    """Thread-safe LRU cache with TTL"""
    
    def __init__(self, maxsize: int = 128, ttl: int = 3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.creation_times[key] > self.ttl:
                del self.cache[key]
                del self.access_times[key]
                del self.creation_times[key]
                return None
            
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self._lock:
            # Evict if at capacity
            if len(self.cache) >= self.maxsize and key not in self.cache:
                # Find LRU item
                lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[lru_key]
                del self.access_times[lru_key]
                del self.creation_times[lru_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.creation_times[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()

# ====================
# WEB API
# ====================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

class ChatRequest(BaseModel):
    prompt: str
    stream: bool = False
    prefer_local: bool = False
    max_cost: Optional[float] = None
    use_cache: bool = True
    ab_test_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    model_used: str
    cost: float
    latency: float
    cached: bool = False

def create_api(router: SmartRouter, client: ResilientLLMClient) -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(title="LLM-Use API", version="1.0.0")
    
    # Configurazione CORS per permettere richieste dal browser
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In produzione, specifica domini esatti
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.get("/models")
    async def list_models():
        """List available models"""
        available = router._get_available_models()
        return {
            "models": [
                {
                    "id": model_id,
                    "name": config.name,
                    "provider": config.provider,
                    "quality": config.quality,
                    "cost_per_1k_input": config.cost_per_1k_input,
                    "cost_per_1k_output": config.cost_per_1k_output,
                    "supports_streaming": config.supports_streaming
                }
                for model_id, config in available.items()
            ]
        }
    
    @app.post("/chat")
    async def chat(request: ChatRequest):
        """Chat endpoint"""
        
        start_time = time.time()
        
        try:
            if request.stream:
                # Streaming response
                async def generate():
                    response_gen = await client.chat(
                        request.prompt,
                        stream=True,
                        prefer_local=request.prefer_local,
                        max_cost=request.max_cost,
                        use_cache=request.use_cache,
                        ab_test_id=request.ab_test_id,
                        user_id=request.user_id
                    )
                    async for chunk in response_gen:
                        yield chunk
                
                return StreamingResponse(generate(), media_type="text/plain")
            else:
                # Regular response
                response = await client.chat(
                    request.prompt,
                    stream=False,
                    prefer_local=request.prefer_local,
                    max_cost=request.max_cost,
                    use_cache=request.use_cache,
                    ab_test_id=request.ab_test_id,
                    user_id=request.user_id
                )
                
                latency = time.time() - start_time
                
                # Get routing info for response
                routing = router.route(request.prompt, request.prefer_local)
                
                return ChatResponse(
                    response=response,
                    model_used=routing.model_name,
                    cost=routing.estimated_cost,
                    latency=latency,
                    cached=False  # TODO: Track cache hits
                )
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics")
    async def get_metrics():
        """Get metrics"""
        return {
            "routing_stats": router.get_stats(),
            "metrics": router.metrics.get_metrics()
        }
    
    @app.post("/benchmark/{model_id}")
    async def benchmark_model(model_id: str, comprehensive: bool = False):
        """Run benchmark for specific model"""
        
        if model_id not in router.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        benchmarker = ProductionBenchmarker(comprehensive=comprehensive)
        result = await benchmarker.benchmark_model(
            model_id, 
            router.models[model_id].provider,
            client,
            comprehensive
        )
        
        return result
    
    # OPZIONALE: Aggiungi endpoint GET per test facili
    @app.get("/test")
    async def test_chat(prompt: str = "Hello, how are you?"):
        """Test endpoint per browser"""
        response = await client.chat(prompt)
        routing = router.route(prompt, False)
        
        return {
            "prompt": prompt,
            "response": response,
            "model_used": routing.model_name,
            "cost": routing.estimated_cost
        }
    
    return app

# ====================
# MAIN APPLICATION
# ====================

async def run_production_server():
    """Run production server"""
    
    # Initialize components
    router = SmartRouter(verbose=False)
    client = ResilientLLMClient(router)
    
    # Create API
    app = create_api(router, client)
    
    # Run server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()

def interactive_demo():
    """Interactive demo with all features"""
    
    print("\n" + "="*60)
    print("  🚀 LLM-Use - Production-Ready Intelligent Router")
    print("="*60 + "\n")
    
    # Initialize
    router = SmartRouter(verbose=True)
    
    # Check available providers
    available_models = router._get_available_models()
    
    if not available_models:
        print("❌ No models available!")
        print("\n📝 Setup instructions:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export GOOGLE_API_KEY=...")
        print("  export GROQ_API_KEY=gsk_...")
        return
    
    print(f"✅ Available models: {len(available_models)}")
    for model_id, config in available_models.items():
        print(f"  - {config.name} ({config.provider})")
    
    # Create client
    client = ResilientLLMClient(router)
    
    # Optional: Create A/B test
    ab_manager = ProductionABTestManager()
    client.set_ab_test_manager(ab_manager)
    
    # FIX: Aggiungi variabile streaming
    use_streaming = False
    
    print("\n📝 Commands:")
    print("  /stream    - Toggle streaming mode")  # FIX: Aggiunto comando
    print("  /stats     - Show statistics")
    print("  /benchmark - Run model benchmarks")
    print("  /cache     - Show cache stats")
    print("  /ab        - Manage A/B tests")
    print("  /api       - Start API server")
    print("  /quit      - Exit")
    print("\n💬 Start chatting!\n")
    
    # Event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '/quit':
                break
            
            # FIX: Aggiungi comando streaming
            elif user_input.lower() == '/stream':
                use_streaming = not use_streaming
                print(f"✅ Streaming: {'ON' if use_streaming else 'OFF'}")
                continue
                
            elif user_input.lower() == '/stats':
                stats = router.get_stats()
                print(f"\n📊 Session Statistics:")
                print(f"  Total requests: {stats['total_requests']}")
                print(f"  Total cost: ${stats['total_cost']:.4f}")
                print(f"  Input tokens: {stats['total_tokens_input']:,}")
                print(f"  Output tokens: {stats['total_tokens_output']:,}")
                print(f"  Model usage:")
                for model, count in stats['model_usage'].items():
                    print(f"    - {router.models[model].name}: {count}x")
                
            elif user_input.lower() == '/benchmark':
                print("\n🔬 Running comprehensive benchmarks...")
                benchmarker = ProductionBenchmarker(comprehensive=True)
                
                async def run_bench():
                    return await benchmarker.benchmark_all_models(available_models, client)
                
                df = loop.run_until_complete(run_bench())
                print("\n📊 Benchmark Results:")
                print(df.to_string())
                
            elif user_input.lower() == '/cache':
                cache_stats = {
                    'size': len(client.cache.cache),
                    'max_size': client.cache.maxsize,
                    'ttl': client.cache.ttl
                }
                print(f"\n💾 Cache Statistics:")
                print(f"  Entries: {cache_stats['size']}/{cache_stats['max_size']}")
                print(f"  TTL: {cache_stats['ttl']}s")
                
            elif user_input.lower().startswith('/ab'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: /ab create <model_a> <model_b>")
                    print("       /ab list")
                    print("       /ab results <test_id>")
                elif parts[1] == 'create' and len(parts) == 4:
                    test_id = ab_manager.create_test(
                        name=f"Test {parts[2]} vs {parts[3]}",
                        model_a=parts[2],
                        model_b=parts[3]
                    )
                    print(f"✅ Created A/B test: {test_id}")
                elif parts[1] == 'list':
                    print("\n📊 Active A/B Tests:")
                    for test_id, test in ab_manager.active_tests.items():
                        print(f"  {test_id}: {test['name']}")
                elif parts[1] == 'results' and len(parts) == 3:
                    results = ab_manager.analyze_test(parts[2])
                    print(f"\n📊 A/B Test Results:")
                    print(json.dumps(results, indent=2))
                
            elif user_input.lower() == '/api':
                print("\n🌐 Starting API server on http://localhost:8080")
                print("Press Ctrl+C to stop")
                try:
                    loop.run_until_complete(run_production_server())
                except KeyboardInterrupt:
                    print("\n✅ API server stopped")
                
            else:
                # Regular chat - FIX COMPLETO
                print("\n🤖 Assistant: ", end="", flush=True)
                
                if use_streaming:
                    # Streaming mode - FIX: Gestione corretta del generator
                    async def stream_response():
                        response_gen = await client.chat(user_input, stream=True)
                        async for chunk in response_gen:
                            print(chunk, end="", flush=True)
                        print()  # Newline alla fine
                    
                    loop.run_until_complete(stream_response())
                else:
                    # Normal mode
                    async def get_response():
                        return await client.chat(user_input)
                    
                    response = loop.run_until_complete(get_response())
                    print(response)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    loop.close()
    
    # Final stats
    stats = router.get_stats()
    print(f"\n\n📊 Final Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total cost: ${stats['total_cost']:.4f}")
    if stats['total_requests'] > 0:
        print(f"  Average cost per request: ${stats['total_cost']/stats['total_requests']:.4f}")
    
    print("\n👋 Thanks for using LLM-Use!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "server":
            print("🚀 Starting production API server...")
            asyncio.run(run_production_server())
        elif command == "benchmark":
            print("🔬 Running benchmarks...")
            router = SmartRouter(verbose=False)
            client = ResilientLLMClient(router)  # ← AGGIUNGI QUESTO!
            benchmarker = ProductionBenchmarker(comprehensive=True)
            
            async def bench():
                models = router._get_available_models()
                return await benchmarker.benchmark_all_models(models,client)
            
            df = asyncio.run(bench())
            print("\n📊 Benchmark Results:")
            print(df.to_string())
            df.to_csv("benchmark_results.csv")
            print("\n✅ Results saved to benchmark_results.csv")
        else:
            print(f"Unknown command: {command}")
            print("Available commands: server, benchmark")
    else:
        interactive_demo()
            