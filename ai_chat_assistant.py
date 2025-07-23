#!/usr/bin/env python3
"""
AI Terminal Pal v2.0 - Supreme Developer Edition
The ultimate AI-powered terminal interface with enhanced features and elegant design

Made with 💟 by Vishnupriyan P R :)
"""

import os
import sys
import json
import csv
import subprocess
import datetime
import shutil
import time
import threading
import tempfile
import base64
import mimetypes
import logging
import re
import math
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
import asyncio
import configparser

# Core dependencies with error handling
try:
    from colorama import init, Fore, Back, Style
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.columns import Columns
    from rich.layout import Layout
    from rich.live import Live
    from rich.align import Align
    from rich.tree import Tree
    from rich.status import Status
    import requests
    import pyperclip
    import psutil
    from PIL import Image
    import PyPDF2
    import pandas as pd
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    import tiktoken
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("📦 Install with: pip install colorama rich requests pyperclip psutil pillow PyPDF2 pandas reportlab tiktoken")
    sys.exit(1)

# Initialize colorama and console
init(autoreset=True)
console = Console()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_terminal_pal.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AITerminalPal')

@dataclass
class AIResponse:
    """Enhanced structured AI response with metadata"""
    content: str
    provider: str
    model: str
    timestamp: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    response_time: Optional[float] = None
    context_length: Optional[int] = None

@dataclass 
class ModelInfo:
    """Model information structure"""
    name: str
    display_name: str
    context_length: int
    cost_per_token: float = 0.0
    description: str = ""

class AIProvider:
    """Enhanced base class for AI providers"""
    def __init__(self, name: str, api_key: str, model: str):
        self.name = name
        self.api_key = api_key
        self.model = model
        self.base_url = ""
        self.models: Dict[str, ModelInfo] = {}
        
    async def query(self, prompt: str, context: Optional[str] = None, 
                   temperature: float = 0.7, max_tokens: int = 4000) -> AIResponse:
        """Query the AI provider with enhanced parameters"""
        raise NotImplementedError
    
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    def get_available_models(self) -> List[ModelInfo]:
        return list(self.models.values())

class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__("OpenAI", api_key, model)
        self.base_url = "https://api.openai.com/v1"
        self.models = {
            "gpt-3.5-turbo": ModelInfo("gpt-3.5-turbo", "GPT-3.5 Turbo", 16384, 0.002, "Fast and efficient general-purpose model"),
            "gpt-4": ModelInfo("gpt-4", "GPT-4", 8192, 0.03, "Most capable model with superior reasoning"),
            "gpt-4-turbo": ModelInfo("gpt-4-turbo", "GPT-4 Turbo", 128000, 0.01, "Latest GPT-4 with extended context"),
            "gpt-4o": ModelInfo("gpt-4o", "GPT-4o", 128000, 0.005, "Optimized GPT-4 variant"),
            "gpt-4o-mini": ModelInfo("gpt-4o-mini", "GPT-4o Mini", 128000, 0.0015, "Lightweight but powerful")
        }
    
    async def query(self, prompt: str, context: Optional[str] = None, 
                   temperature: float = 0.7, max_tokens: int = 4000) -> AIResponse:
        start_time = time.time()
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            messages = [{"role": "user", "content": prompt}]
            if context:
                messages.insert(0, {"role": "system", "content": context})
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=90
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                usage = result.get('usage', {})
                return AIResponse(
                    content=result['choices'][0]['message']['content'],
                    provider=self.name,
                    model=self.model,
                    timestamp=datetime.datetime.now().isoformat(),
                    tokens_used=usage.get('total_tokens'),
                    response_time=response_time,
                    context_length=usage.get('prompt_tokens', 0)
                )
            else:
                return AIResponse(
                    content=f"API Error {response.status_code}: {response.text}",
                    provider=self.name,
                    model=self.model,
                    timestamp=datetime.datetime.now().isoformat(),
                    response_time=response_time
                )
                
        except Exception as e:
            return AIResponse(
                content=f"OpenAI Error: {str(e)}",
                provider=self.name,
                model=self.model,
                timestamp=datetime.datetime.now().isoformat(),
                response_time=time.time() - start_time
            )

class ClaudeProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__("Claude", api_key, model)
        self.base_url = "https://api.anthropic.com/v1"
        self.models = {
            "claude-3-haiku-20240307": ModelInfo("claude-3-haiku-20240307", "Claude 3 Haiku", 200000, 0.00025, "Fast and lightweight"),
            "claude-3-sonnet-20240229": ModelInfo("claude-3-sonnet-20240229", "Claude 3 Sonnet", 200000, 0.003, "Balanced performance and speed"),
            "claude-3-opus-20240229": ModelInfo("claude-3-opus-20240229", "Claude 3 Opus", 200000, 0.015, "Most powerful Claude model"),
            "claude-3-5-sonnet-20241022": ModelInfo("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", 200000, 0.003, "Latest enhanced model")
        }

class GeminiProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__("Gemini", api_key, model)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.models = {
            "gemini-pro": ModelInfo("gemini-pro", "Gemini Pro", 32768, 0.0005, "Google's flagship model"),
            "gemini-1.5-pro": ModelInfo("gemini-1.5-pro", "Gemini 1.5 Pro", 2097152, 0.001, "Extended context version"),
            "gemini-1.5-flash": ModelInfo("gemini-1.5-flash", "Gemini 1.5 Flash", 1048576, 0.0002, "Fast and efficient")
        }

class GroqProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        super().__init__("Groq", api_key, model)
        self.base_url = "https://api.groq.com/openai/v1"
        self.models = {
            "llama3-8b-8192": ModelInfo("llama3-8b-8192", "Llama 3 8B", 8192, 0.0001, "Fast and efficient open model"),
            "llama3-70b-8192": ModelInfo("llama3-70b-8192", "Llama 3 70B", 8192, 0.0008, "Powerful open model"),
            "mixtral-8x7b-32768": ModelInfo("mixtral-8x7b-32768", "Mixtral 8x7B", 32768, 0.0006, "Mixture of experts model"),
            "gemma-7b-it": ModelInfo("gemma-7b-it", "Gemma 7B", 8192, 0.0001, "Google's efficient model")
        }

class MistralProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        super().__init__("Mistral", api_key, model)
        self.base_url = "https://api.mistral.ai/v1"
        self.models = {
            "mistral-tiny": ModelInfo("mistral-tiny", "Mistral Tiny", 32000, 0.0001, "Ultra-fast for simple tasks"),
            "mistral-small": ModelInfo("mistral-small", "Mistral Small", 32000, 0.0006, "Balanced performance"),
            "mistral-medium": ModelInfo("mistral-medium", "Mistral Medium", 32000, 0.0027, "High performance model"),
            "mistral-large-latest": ModelInfo("mistral-large-latest", "Mistral Large", 32000, 0.008, "Most capable model"),
            "codestral-latest": ModelInfo("codestral-latest", "Codestral", 32000, 0.001, "Specialized for coding")
        }
    
    async def query(self, prompt: str, context: Optional[str] = None, 
                   temperature: float = 0.7, max_tokens: int = 4000) -> AIResponse:
        start_time = time.time()
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            messages = [{"role": "user", "content": prompt}]
            if context:
                messages.insert(0, {"role": "system", "content": context})
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=90
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                usage = result.get('usage', {})
                return AIResponse(
                    content=result['choices'][0]['message']['content'],
                    provider=self.name,
                    model=self.model,
                    timestamp=datetime.datetime.now().isoformat(),
                    tokens_used=usage.get('total_tokens'),
                    response_time=response_time
                )
            else:
                return AIResponse(
                    content=f"API Error {response.status_code}: {response.text}",
                    provider=self.name,
                    model=self.model,
                    timestamp=datetime.datetime.now().isoformat(),
                    response_time=response_time
                )
                
        except Exception as e:
            return AIResponse(
                content=f"Mistral Error: {str(e)}",
                provider=self.name,
                model=self.model,
                timestamp=datetime.datetime.now().isoformat(),
                response_time=time.time() - start_time
            )

class ProjectIntegrator:
    """Enhanced project file integration with advanced features"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.vue', '.svelte',
            '.java', '.c', '.cpp', '.cs', '.php', '.rb', '.go', '.rs', 
            '.swift', '.kt', '.scala', '.dart', '.r', '.jl',
            '.html', '.css', '.scss', '.less', '.xml', '.svg',
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.md', '.rst', '.txt', '.tex', '.adoc',
            '.sh', '.bat', '.ps1', '.zsh', '.fish',
            '.sql', '.graphql', '.proto', '.thrift'
        }
        self.ignore_patterns = {
            '__pycache__', '.git', '.svn', 'node_modules', 
            '.venv', 'venv', '.env', 'build', 'dist', 
            '.DS_Store', '*.pyc', '*.pyo', '*.pyd'
        }
    
    def scan_project(self) -> Dict[str, List[str]]:
        """Enhanced project scanning with categorization"""
        files = {
            'code': [], 'frontend': [], 'backend': [], 'config': [], 
            'docs': [], 'tests': [], 'scripts': [], 'data': []
        }
        
        try:
            for file_path in self.project_path.rglob('*'):
                if self._should_ignore(file_path):
                    continue
                    
                if file_path.is_file() and file_path.suffix in self.supported_extensions:
                    relative_path = str(file_path.relative_to(self.project_path))
                    category = self._categorize_file(file_path)
                    files[category].append(relative_path)
                        
        except Exception as e:
            logger.error(f"Project scan error: {e}")
            
        return files
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if file should be ignored"""
        for part in path.parts:
            if part in self.ignore_patterns:
                return True
        return False
    
    def _categorize_file(self, file_path: Path) -> str:
        """Categorize files based on extension and path"""
        suffix = file_path.suffix
        name = file_path.name.lower()
        path_str = str(file_path).lower()
        
        if 'test' in name or '/test' in path_str:
            return 'tests'
        elif suffix in {'.html', '.css', '.scss', '.js', '.jsx', '.ts', '.tsx', '.vue'}:
            return 'frontend'
        elif suffix in {'.py', '.java', '.go', '.rs', '.cpp', '.c'}:
            return 'backend'
        elif suffix in {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.xml'}:
            return 'config'
        elif suffix in {'.md', '.rst', '.txt', '.tex'}:
            return 'docs'
        elif suffix in {'.sh', '.bat', '.ps1'}:
            return 'scripts'
        elif suffix in {'.csv', '.json', '.xml'}:
            return 'data'
        else:
            return 'code'

class ThemeManager:
    """Advanced theme management system"""
    
    def __init__(self):
        self.themes = {
            'professional': {
                'primary': Fore.BLUE,
                'secondary': Fore.CYAN,
                'success': Fore.GREEN,
                'warning': Fore.YELLOW,
                'error': Fore.RED,
                'accent': Fore.MAGENTA,
                'muted': Fore.LIGHTBLACK_EX
            },
            'ocean': {
                'primary': Fore.BLUE,
                'secondary': Fore.LIGHTBLUE_EX,
                'success': Fore.GREEN,
                'warning': Fore.YELLOW,
                'error': Fore.RED,
                'accent': Fore.CYAN,
                'muted': Fore.LIGHTBLACK_EX
            },
            'forest': {
                'primary': Fore.GREEN,
                'secondary': Fore.LIGHTGREEN_EX,
                'success': Fore.GREEN,
                'warning': Fore.YELLOW,
                'error': Fore.RED,
                'accent': Fore.CYAN,
                'muted': Fore.LIGHTBLACK_EX
            },
            'minimal': {
                'primary': Fore.WHITE,
                'secondary': Fore.LIGHTWHITE_EX,
                'success': Fore.GREEN,
                'warning': Fore.YELLOW,
                'error': Fore.RED,
                'accent': Fore.BLUE,
                'muted': Fore.LIGHTBLACK_EX
            }
        }
        self.current_theme = 'professional'
    
    def get_color(self, color_type: str) -> str:
        return self.themes[self.current_theme].get(color_type, Fore.WHITE)
    
    def set_theme(self, theme_name: str) -> bool:
        if theme_name in self.themes:
            self.current_theme = theme_name
            return True
        return False

class NavigationHelper:
    """Enhanced navigation and help system"""
    
    def __init__(self):
        self.command_categories = {
            'setup': {
                'icon': '🔧',
                'description': 'Initial setup and configuration',
                'commands': {
                    '/setup': 'Interactive setup wizard',
                    '/config': 'Configure AI settings',
                    '/provider': 'Switch AI provider',
                    '/model': 'Select AI model by number',
                    '/theme': 'Change color theme'
                }
            },
            'ai': {
                'icon': '🤖',
                'description': 'AI interaction and queries',
                'commands': {
                    '/ask': 'Ask AI with context awareness',
                    '/chat': 'Interactive chat mode',
                    '/explain': 'Explain code or concepts',
                    '/generate': 'Generate code from description',
                    '/improve': 'Get improvement suggestions',
                    '/translate': 'Translate code between languages',
                    '/optimize': 'Optimize existing code'
                }
            },
            'files': {
                'icon': '📁',
                'description': 'File operations and management',
                'commands': {
                    '/attach': 'Attach file for AI analysis',
                    '/read': 'Read and display file content',
                    '/write': 'Write content to file',
                    '/edit': 'Edit file with AI assistance',
                    '/backup': 'Create file backup',
                    '/restore': 'Restore file from backup',
                    '/compare': 'Compare two files'
                }
            },
            'project': {
                'icon': '🚀',
                'description': 'Project management and analysis',
                'commands': {
                    '/scan': 'Scan project structure',
                    '/analyze': 'Analyze project architecture',
                    '/deps': 'Analyze dependencies',
                    '/metrics': 'Show project metrics',
                    '/tree': 'Display project tree',
                    '/search': 'Search in project files',
                    '/refactor': 'Project-wide refactoring'
                }
            },
            'dev': {
                'icon': '⚡',
                'description': 'Development tools and utilities',
                'commands': {
                    '/debug': 'Debug code with AI',
                    '/test': 'Generate and run tests',
                    '/lint': 'Code linting and quality check',
                    '/format': 'Auto-format code',
                    '/docs': 'Generate documentation',
                    '/api': 'API testing and documentation',
                    '/security': 'Security analysis'
                }
            },
            'export': {
                'icon': '📊',
                'description': 'Export and reporting',
                'commands': {
                    '/export': 'Export session data',
                    '/pdf': 'Generate PDF report',
                    '/report': 'Generate project report',
                    '/stats': 'Show detailed statistics',
                    '/history': 'View command history',
                    '/logs': 'View application logs'
                }
            },
            'system': {
                'icon': '🛠️',
                'description': 'System and utility commands',
                'commands': {
                    '/status': 'System status overview',
                    '/monitor': 'Resource monitoring',
                    '/copy': 'Copy to clipboard',
                    '/paste': 'Paste from clipboard',
                    '/clear': 'Clear terminal',
                    '/update': 'Check for updates',
                    '/help': 'Show this help',
                    '/exit': 'Exit application'
                }
            }
        }
    
    def get_category_help(self, category: str) -> Optional[Dict]:
        return self.command_categories.get(category)
    
    def get_all_categories(self) -> List[str]:
        return list(self.command_categories.keys())

class AITerminalPal:
    """Supreme AI Terminal Pal v2.0 with enhanced features"""
    
    def __init__(self):
        self.version = "2.0 Supreme"
        self.commands = {}
        self.session_log = []
        self.error_log = []
        self.chat_history = []
        self.current_project_path = os.getcwd()
        
        # Enhanced components
        self.theme_manager = ThemeManager()
        self.navigation_helper = NavigationHelper()
        self.project_integrator = ProjectIntegrator(self.current_project_path)
        
        # Configuration
        self.config_dir = Path.home() / ".ai_terminal_pal_v2"
        self.config_file = self.config_dir / "config.json"
        self.themes_file = self.config_dir / "themes.json"
        self.history_file = self.config_dir / "history.json"
        
        # AI Providers with enhanced support
        self.ai_provider: Optional[AIProvider] = None
        self.available_providers = {
            "OpenAI": {
                "class": OpenAIProvider,
                "description": "Industry-leading AI models from OpenAI",
                "icon": "🚀"
            },
            "Claude": {
                "class": ClaudeProvider,
                "description": "Anthropic's powerful and safe AI assistant",
                "icon": "🧠"
            },
            "Gemini": {
                "class": GeminiProvider,
                "description": "Google's multimodal AI platform",
                "icon": "💎"
            },
            "Groq": {
                "class": GroqProvider,
                "description": "Ultra-fast inference for open models",
                "icon": "⚡"
            },
            "Mistral": {
                "class": MistralProvider,
                "description": "European AI excellence with privacy focus",
                "icon": "🌟"
            }
        }
        
        # Enhanced configuration
        self.config = {}
        self.performance_stats = {
            'total_queries': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'session_start': datetime.datetime.now(),
            'avg_response_time': 0.0
        }
        
        # Initialize
        self.setup_directories()
        self.load_config()
        self.register_enhanced_commands()
        
    def setup_directories(self):
        """Create enhanced directory structure"""
        directories = [
            self.config_dir,
            self.config_dir / "backups",
            self.config_dir / "sessions",
            self.config_dir / "exports",
            self.config_dir / "themes"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)
    
    def load_config(self):
        """Load enhanced configuration"""
        default_config = {
            "providers": {},
            "current_provider": None,
            "current_model": None,
            "theme": "professional",
            "auto_copy": True,
            "auto_save": True,
            "project_integration": True,
            "temperature": 0.7,
            "max_tokens": 4000,
            "response_format": "markdown",
            "show_stats": True,
            "animation_speed": "normal",
            "context_awareness": True,
            "safety_mode": True
        }
        
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    
            self.config = default_config
            self.theme_manager.set_theme(self.config.get('theme', 'professional'))
            
        except Exception as e:
            logger.error(f"Config loading error: {e}")
            self.config = default_config
            
        self.save_config()
    
    def save_config(self):
        """Save configuration with error handling"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Config saving error: {e}")
    
    def display_enhanced_banner(self):
        """Display the clean AI Terminal Pal V2 banner with blue gradient"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
        # Get terminal dimensions with safety check
        terminal_width = min(shutil.get_terminal_size().columns, 150)
    
        # Clean ASCII art banner for "AI TERMINAL PAL V2"
        banner_lines = [
        "╔" + "═" * (terminal_width - 2) + "╗",
        "║" + " " * (terminal_width - 2) + "║",
        "║     █████╗ ██╗    ████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗         ██████╗  █████╗ ██╗         ██╗   ██╗ ██████╗ " + " " * max(0, terminal_width - 122) + "║",
        "║    ██╔══██╗██║    ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║         ██╔══██╗██╔══██╗██║         ██║   ██║ ╚════██╗" + " " * max(0, terminal_width - 122) + "║",
        "║    ███████║██║       ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║         ██████╔╝███████║██║         ██║   ██║  █████╔╝" + " " * max(0, terminal_width - 122) + "║",
        "║    ██╔══██║██║       ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║         ██╔═══╝ ██╔══██║██║         ╚██╗ ██╔╝ ██" + " " * max(0, terminal_width - 122) + "║",
        "║    ██║  ██║██║       ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗    ██║     ██║  ██║███████╗     ╚████╔╝  ██████╗" + " " * max(0, terminal_width - 122) + "║",
        "║    ╚═╝  ╚═╝╚═╝       ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝╚══════╝      ╚═══╝   ╚═════╝ " + " " * max(0, terminal_width - 122) + "║",
        "║" + " " * (terminal_width - 2) + "║",
        "║" + "🚀 ULTIMATE DEVELOPER EDITION 🚀".center(terminal_width - 2) + "║",
        "║" + "⚡ Multi-AI Provider Support: OpenAI • Claude • Gemini • Groq • Mistral ⚡".center(terminal_width - 2) + "║",
        "║" + " " * (terminal_width - 2) + "║",
        "║" + "Made with 💟 by Vishnupriyan P R :)".center(terminal_width - 2) + "║",
        "║" + " " * (terminal_width - 2) + "║",
        "╚" + "═" * (terminal_width - 2) + "╝"
    ]
    
        # Display banner with beautiful blue gradient effect
        blue_colors = [Fore.LIGHTBLUE_EX, Fore.BLUE, Fore.BLUE, Fore.BLUE, Fore.BLUE, Fore.BLUE, Fore.LIGHTBLUE_EX]
    
        for i, line in enumerate(banner_lines):
            # Apply gradient effect to the title lines (lines 2-6)
            if 2 <= i <= 6:
                color = blue_colors[i - 2] if i - 2 < len(blue_colors) else Fore.BLUE
                print(f"{color}{Style.BRIGHT}{line}{Style.RESET_ALL}")
            else:
                # Regular blue for other lines
                print(f"{Fore.BLUE}{Style.BRIGHT}{line}{Style.RESET_ALL}")
    
        # Enhanced separator line
        separator_line = "═" * terminal_width
        print(f"{Fore.CYAN}{Style.BRIGHT}{separator_line}{Style.RESET_ALL}")
    
        # Call the other methods INSIDE this method (not at class level)
        self.display_status_panel()
        self.display_navigation_hints()


    def display_status_panel(self):
        """Display elegant status information"""
        provider_info = "Not configured"
        if self.ai_provider:
            provider_info = f"{self.ai_provider.name} • {self.ai_provider.model}"
        
        # Create status table
        status_data = [
            ["🤖 AI Provider", provider_info],
            ["📁 Project Path", str(Path(self.current_project_path).name)],
            ["🎨 Theme", self.config.get('theme', 'professional').title()],
            ["📊 Session Queries", str(self.performance_stats['total_queries'])],
            ["🕒 Uptime", str(datetime.datetime.now() - self.performance_stats['session_start']).split('.')[0]]
        ]
        
        # Create elegant status display
        console.print("\n")
        status_table = Table(show_header=False, border_style="blue", box=None)
        status_table.add_column(style="cyan", width=20)
        status_table.add_column(style="white")
        
        for label, value in status_data:
            status_table.add_row(label, value)
        
        console.print(Align.center(status_table))
    
    def display_navigation_hints(self):
        """Display helpful navigation hints"""
        hints = [
            "💡 Quick Start: /setup → /ask your question → /help for more commands",
            "🔍 Navigation: /nav to see all categories • /nav <category> for specific help",
            "⚡ Power User: Use @filename in queries • Chain commands with &&"
        ]
        
        console.print("\n")
        for hint in hints:
            console.print(f"   {hint}", style="dim cyan")
        console.print("\n" + "─" * min(80, shutil.get_terminal_size().columns), style="blue")
        console.print()
    
    def register_enhanced_commands(self):
        """Register all enhanced commands with improved organization"""
        self.commands = {
            # Setup & Configuration
            '/setup': self.enhanced_setup,
            '/config': self.configure_settings,
            '/provider': self.switch_provider,
            '/model': self.select_model_by_number,
            '/theme': self.change_theme,
            '/customize': self.customize_interface,
            
            # Navigation & Help
            '/help': self.show_enhanced_help,
            '/nav': self.show_navigation,
            '/quick': self.quick_start_guide,
            '/tips': self.show_pro_tips,
            '/shortcuts': self.show_shortcuts,
            
            # AI Interaction
            '/ask': self.ask_ai_enhanced,
            '/chat': self.start_enhanced_chat,
            '/explain': self.explain_with_ai,
            '/generate': self.generate_code,
            '/improve': self.improve_code,
            '/translate': self.translate_code,
            '/optimize': self.optimize_code,
            '/brainstorm': self.brainstorm_session,
            
            # File Operations
            '/attach': self.attach_file,
            '/read': self.read_file_enhanced,
            '/write': self.write_file_enhanced,
            '/edit': self.edit_with_ai,
            '/backup': self.backup_file,
            '/restore': self.restore_file,
            '/compare': self.compare_files,
            
            # Project Management
            '/scan': self.scan_project_enhanced,
            '/analyze': self.analyze_project,
            '/deps': self.analyze_dependencies,
            '/metrics': self.project_metrics,
            '/tree': self.display_project_tree,
            '/search': self.search_project,
            '/refactor': self.refactor_project,
            
            # Development Tools
            '/debug': self.debug_with_ai,
            '/test': self.generate_tests,
            '/lint': self.lint_code,
            '/format': self.format_code,
            '/docs': self.generate_documentation,
            '/api': self.api_tools,
            '/security': self.security_analysis,
            '/performance': self.performance_analysis,
            
            # Export & Reporting
            '/export': self.export_enhanced,
            '/pdf': self.generate_pdf_report,
            '/report': self.generate_project_report,
            '/stats': self.show_detailed_stats,
            '/history': self.show_enhanced_history,
            '/logs': self.show_logs,
            
            # System & Utilities
            '/status': self.show_system_status,
            '/monitor': self.system_monitor,
            '/copy': self.copy_to_clipboard,
            '/paste': self.paste_from_clipboard,
            '/clear': self.clear_screen,
            '/update': self.check_updates,
            '/benchmark': self.run_benchmarks,
            
            # Basic Commands
            '/exit': self.exit_app,
            '/quit': self.exit_app,
            '/restart': self.restart_app
        }
    
    async def enhanced_setup(self, args=None):
        """Enhanced setup wizard with improved UX"""
        console.print(Panel.fit(
            "[bold blue]🎯 Welcome to AI Terminal Pal v2.0 Setup![/]\n"
            "[white]Let's configure your AI providers and personalize your experience[/]",
            title="🚀 Setup Wizard",
            border_style="blue"
        ))
        
        # Show provider selection with enhanced display
        self.display_provider_selection()
        
        # Provider selection with numbers
        provider_list = list(self.available_providers.keys())
        
        while True:
            console.print("\n[cyan]Select AI Provider (enter number):[/]")
            for i, provider in enumerate(provider_list, 1):
                info = self.available_providers[provider]
                console.print(f"  {i}. {info['icon']} {provider} - {info['description']}")
            
            try:
                choice = IntPrompt.ask("Enter choice", choices=[str(i) for i in range(1, len(provider_list) + 1)])
                selected_provider = provider_list[choice - 1]
                break
            except Exception:
                console.print("[red]❌ Invalid selection. Please try again.[/]")
        
        # Model selection for chosen provider
        await self.setup_provider_models(selected_provider)
        
        # Theme selection
        await self.setup_theme_selection()
        
        # Performance test
        if Confirm.ask("\n🧪 Test AI connection and performance?", default=True):
            await self.run_setup_tests()
        
        # Final setup completion
        self.display_setup_completion()
    
    def display_provider_selection(self):
        """Display enhanced provider selection interface"""
        providers_table = Table(
            title="🤖 Available AI Providers",
            show_header=True,
            header_style="bold blue",
            border_style="blue"
        )
        providers_table.add_column("Provider", style="bold white", width=12)
        providers_table.add_column("Models", style="green", width=35)
        providers_table.add_column("Features", style="yellow", width=30)
        providers_table.add_column("Status", style="cyan", width=15)
        
        for name, info in self.available_providers.items():
            # Get sample models
            try:
                provider_instance = info["class"]("dummy_key", "dummy_model")
                models = list(provider_instance.models.keys())[:3]
                models_str = ", ".join(models) + ("..." if len(provider_instance.models) > 3 else "")
            except:
                models_str = "Multiple models available"
            
            # Features
            features = "Chat, Code, Analysis"
            
            # Status
            current_config = self.config.get("providers", {}).get(name, {})
            status = "✅ Configured" if current_config.get("api_key") else "❌ Not configured"
            
            providers_table.add_row(
                f"{info['icon']} {name}",
                models_str,
                features,
                status
            )
        
        console.print(providers_table)
    
    async def setup_provider_models(self, provider_name: str):
        """Setup models for selected provider"""
        provider_class = self.available_providers[provider_name]["class"]
        
        # Get existing API key or prompt for new one
        current_config = self.config.get("providers", {}).get(provider_name, {})
        
        if current_config.get("api_key"):
            console.print(f"[green]✅ Found existing API key for {provider_name}[/]")
            use_existing = Confirm.ask("Use existing API key?", default=True)
            if use_existing:
                api_key = current_config["api_key"]
            else:
                api_key = Prompt.ask(f"Enter new {provider_name} API key", password=True)
        else:
            console.print(f"\n[yellow]🔑 {provider_name} API Key Required[/]")
            api_key = Prompt.ask(f"Enter {provider_name} API key", password=True)
        
        if not api_key:
            console.print("[red]❌ API key is required to continue[/]")
            return
        
        # Initialize provider and show models
        try:
            provider_instance = provider_class(api_key, "dummy_model")
            models = provider_instance.get_available_models()
            
            console.print(f"\n[cyan]Available models for {provider_name}:[/]")
            for i, model in enumerate(models, 1):
                console.print(f"  {i}. {model.display_name}")
                console.print(f"     Context: {model.context_length:,} tokens | {model.description}")
            
            # Model selection
            try:
                choice = IntPrompt.ask("Select model (enter number)", choices=[str(i) for i in range(1, len(models) + 1)])
                selected_model = models[choice - 1]
            except:
                selected_model = models[0]  # Default to first model
            
            # Save configuration
            if "providers" not in self.config:
                self.config["providers"] = {}
            
            self.config["providers"][provider_name] = {
                "api_key": api_key,
                "model": selected_model.name
            }
            self.config["current_provider"] = provider_name
            self.config["current_model"] = selected_model.name
            
            # Initialize active provider
            self.ai_provider = provider_class(api_key, selected_model.name)
            
            console.print(f"[green]✅ {provider_name} configured with {selected_model.display_name}![/]")
            
        except Exception as e:
            console.print(f"[red]❌ Error setting up {provider_name}: {str(e)}[/]")
    
    async def setup_theme_selection(self):
        """Enhanced theme selection"""
        console.print("\n[cyan]🎨 Choose your interface theme:[/]")
        
        themes = list(self.theme_manager.themes.keys())
        for i, theme in enumerate(themes, 1):
            console.print(f"  {i}. {theme.title()} Theme")
        
        try:
            choice = IntPrompt.ask("Select theme", choices=[str(i) for i in range(1, len(themes) + 1)])
            selected_theme = themes[choice - 1]
            self.theme_manager.set_theme(selected_theme)
            self.config["theme"] = selected_theme
            console.print(f"[green]✅ Theme set to {selected_theme.title()}[/]")
        except:
            console.print("[yellow]⚠️ Using default Professional theme[/]")
    
    async def run_setup_tests(self):
        """Run comprehensive setup tests"""
        console.print("\n[blue]🧪 Running setup tests...[/]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            
            # Test API connection
            task1 = progress.add_task("Testing AI connection...", total=1)
            await self.test_ai_connection()
            progress.update(task1, completed=1)
            
            # Test project scanning
            task2 = progress.add_task("Scanning project structure...", total=1)
            self.project_integrator.scan_project()
            progress.update(task2, completed=1)
            
            # Test clipboard
            task3 = progress.add_task("Testing clipboard integration...", total=1)
            try:
                pyperclip.copy("test")
                progress.update(task3, completed=1)
                console.print("[green]✅ All tests passed![/]")
            except:
                console.print("[yellow]⚠️ Clipboard test failed (optional feature)[/]")
    
    def display_setup_completion(self):
        """Display setup completion message"""
        console.print("\n")
        completion_panel = Panel.fit(
            "[bold green]🎉 Setup Complete![/]\n\n"
            "[white]Your AI Terminal Pal is ready to use![/]\n"
            "[cyan]Try these commands to get started:[/]\n\n"
            "• [yellow]/ask[/] - Ask AI anything\n"
            "• [yellow]/chat[/] - Start interactive chat\n"
            "• [yellow]/scan[/] - Analyze your project\n"
            "• [yellow]/help[/] - View all commands\n"
            "• [yellow]/nav[/] - Navigation guide\n\n"
            "[dim]Tip: Use /tips for pro usage tips![/]",
            title="🚀 Ready to Go!",
            border_style="green"
        )
        console.print(completion_panel)
        
        self.save_config()
    
    def select_model_by_number(self, args):
        """Enhanced model selection by number"""
        if not self.ai_provider:
            console.print("[red]❌ No AI provider configured. Run /setup first[/]")
            return
        
        models = self.ai_provider.get_available_models()
        
        console.print(f"\n[cyan]Available models for {self.ai_provider.name}:[/]")
        
        model_table = Table(show_header=True, header_style="bold blue")
        model_table.add_column("#", style="cyan", width=3)
        model_table.add_column("Model", style="green", width=25)
        model_table.add_column("Context", style="yellow", width=12)
        model_table.add_column("Description", style="white", width=40)
        
        for i, model in enumerate(models, 1):
            model_table.add_row(
                str(i),
                model.display_name,
                f"{model.context_length:,}",
                model.description
            )
        
        console.print(model_table)
        
        try:
            if args and args[0].isdigit():
                choice = int(args[0])
            else:
                choice = IntPrompt.ask("Select model number", choices=[str(i) for i in range(1, len(models) + 1)])
            
            if 1 <= choice <= len(models):
                selected_model = models[choice - 1]
                
                # Update provider with new model
                provider_name = self.ai_provider.name
                api_key = self.ai_provider.api_key
                provider_class = self.available_providers[provider_name]["class"]
                
                self.ai_provider = provider_class(api_key, selected_model.name)
                
                # Update config
                self.config["current_model"] = selected_model.name
                self.config["providers"][provider_name]["model"] = selected_model.name
                self.save_config()
                
                console.print(f"[green]✅ Switched to {selected_model.display_name}[/]")
            else:
                console.print("[red]❌ Invalid model number[/]")
                
        except Exception as e:
            console.print(f"[red]❌ Error selecting model: {str(e)}[/]")
    
    def show_enhanced_help(self, args=None):
        """Display enhanced help system"""
        if args and len(args) > 0:
            category = args[0].lower()
            self.show_category_help(category)
        else:
            self.show_full_help()
    
    def show_full_help(self):
        """Show comprehensive help overview"""
        console.print(Panel.fit(
            f"[bold blue]AI Terminal Pal v{self.version} - Command Reference[/]\n"
            "[white]Use /nav <category> for detailed help on specific areas[/]",
            title="📚 Help System",
            border_style="blue"
        ))
        
        # Create categories overview
        categories_table = Table(show_header=True, header_style="bold blue")
        categories_table.add_column("Category", style="cyan", width=12)
        categories_table.add_column("Commands", style="green", width=8)
        categories_table.add_column("Description", style="white", width=50)
        
        for cat_name, cat_info in self.navigation_helper.command_categories.items():
            cmd_count = len(cat_info['commands'])
            categories_table.add_row(
                f"{cat_info['icon']} {cat_name.title()}",
                str(cmd_count),
                cat_info['description']
            )
        
        console.print(categories_table)
        
        # Quick commands
        console.print("\n[bold yellow]🚀 Quick Commands:[/]")
        quick_commands = [
            "/ask <question> - Ask AI anything",
            "/chat - Interactive chat mode", 
            "/scan - Analyze project",
            "/nav <category> - Category help",
            "/setup - Reconfigure settings"
        ]
        
        for cmd in quick_commands:
            console.print(f"  • {cmd}")
    
    def show_category_help(self, category: str):
        """Show help for specific category"""
        cat_info = self.navigation_helper.get_category_help(category)
        if not cat_info:
            console.print(f"[red]❌ Category '{category}' not found[/]")
            console.print(f"[yellow]Available categories: {', '.join(self.navigation_helper.get_all_categories())}[/]")
            return
        
        # Create detailed category help
        console.print(Panel.fit(
            f"[bold blue]{cat_info['icon']} {category.title()} Commands[/]\n"
            f"[white]{cat_info['description']}[/]",
            title=f"📖 {category.title()} Help",
            border_style="blue"
        ))
        
        cmd_table = Table(show_header=True, header_style="bold blue")
        cmd_table.add_column("Command", style="green", width=20)
        cmd_table.add_column("Description", style="white", width=60)
        
        for cmd, desc in cat_info['commands'].items():
            cmd_table.add_row(cmd, desc)
        
        console.print(cmd_table)
    
    def show_navigation(self, args=None):
        """Enhanced navigation system"""
        if args and len(args) > 0:
            self.show_category_help(args[0])
        else:
            console.print(Panel.fit(
                "[bold blue]🧭 Navigation Guide[/]\n"
                "[white]Use /nav <category> for specific help[/]",
                title="Navigation System",
                border_style="blue"
            ))
            
            # Display navigation tree
            nav_tree = Tree("📚 Command Categories")
            
            for cat_name, cat_info in self.navigation_helper.command_categories.items():
                category_branch = nav_tree.add(f"{cat_info['icon']} {cat_name.title()}")
                for cmd in list(cat_info['commands'].keys())[:3]:  # Show first 3 commands
                    category_branch.add(cmd)
                if len(cat_info['commands']) > 3:
                    category_branch.add(f"... +{len(cat_info['commands']) - 3} more")
            
            console.print(nav_tree)
    
    async def ask_ai_enhanced(self, args):
        """Enhanced AI interaction with advanced features"""
        if not args:
            console.print("[red]❌ Please provide a query. Example: /ask How to optimize Python code?[/]")
            return
        
        if not self.ai_provider:
            console.print("[red]❌ No AI provider configured. Run /setup first[/]")
            return
        
        query = " ".join(args)
        
        # Enhanced context building
        context_files = []
        file_refs = re.findall(r'@(\S+)', query)
        
        # Auto-detect project context if enabled
        if self.config.get('context_awareness', True) and not file_refs:
            # Add relevant project files automatically
            project_files = self.project_integrator.scan_project()
            if project_files.get('code'):
                # Add main files for context
                for main_file in project_files['code'][:2]:  # Limit to prevent token overflow
                    content = self.project_integrator.read_file(main_file)
                    if content and len(content) < 5000:  # Size limit
                        context_files.append(f"Project file: {main_file}\n{content[:2000]}...")
        
        # Process explicit file references
        for file_ref in file_refs:
            file_content = self.project_integrator.read_file(file_ref)
            if file_content:
                context_files.append(f"File: {file_ref}\n{file_content[:3000]}...")
                query = query.replace(f'@{file_ref}', f'the file {file_ref}')
        
        # Build comprehensive context
        context = None
        if context_files:
            context = "Project context:\n" + "\n\n".join(context_files)
        
        # Enhanced progress display
        start_time = time.time()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(
                    f"🤔 {self.ai_provider.name} ({self.ai_provider.model}) is thinking...", 
                    total=None
                )
                
                response = await self.ai_provider.query(
                    query, 
                    context, 
                    temperature=self.config.get('temperature', 0.7),
                    max_tokens=self.config.get('max_tokens', 4000)
                )
                progress.update(task, completed=True)
            
            # Update performance stats
            self.performance_stats['total_queries'] += 1
            if response.tokens_used:
                self.performance_stats['total_tokens'] += response.tokens_used
            
            response_time = time.time() - start_time
            self.performance_stats['avg_response_time'] = (
                (self.performance_stats['avg_response_time'] * (self.performance_stats['total_queries'] - 1) + response_time) 
                / self.performance_stats['total_queries']
            )
            
            # Enhanced logging
            log_entry = {
                'timestamp': response.timestamp,
                'type': 'ai_query',
                'provider': response.provider,
                'model': response.model,
                'query': query,
                'response': response.content,
                'tokens_used': response.tokens_used,
                'response_time': response_time,
                'context_files': file_refs,
                'context_length': len(context) if context else 0
            }
            self.session_log.append(log_entry)
            
            # Display enhanced response
            self.display_enhanced_ai_response(response, query)
            
            # Auto-copy functionality
            if self.config.get('auto_copy', True) and not response.content.startswith("Error"):
                try:
                    pyperclip.copy(response.content)
                    console.print("[dim green]📋 Response auto-copied to clipboard[/]")
                except Exception:
                    pass
            
            # Auto-save if enabled
            if self.config.get('auto_save', True):
                self.auto_save_session()
                
        except Exception as e:
            error_msg = f"AI query failed: {str(e)}"
            self.error_log.append(error_msg)
            console.print(f"[red]❌ {error_msg}[/]")
    
    def display_enhanced_ai_response(self, response: AIResponse, query: str):
        """Display AI response with supreme formatting"""
        # Check for code blocks
        has_code = "```" in response.content

        
        if has_code:
            self.display_code_response(response, query)
        else:
            self.display_text_response(response, query)
        
        # Enhanced metadata display
        self.display_response_metadata(response)
    
    def display_code_response(self, response: AIResponse, query: str):
        """Display response with code blocks"""
        parts = response.content.split("```")
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Text part
                if part.strip():
                    clean_text = self.clean_markdown_text(part)
                    console.print(Panel(
                        clean_text,
                        title=f"💬 {response.provider} Response",
                        border_style="blue"
                    ))
            else:  # Code part
                lines = part.strip().split('\n')
                language = "python"  # default
                code_content = part.strip()
                
                # Detect language
                if lines and lines[0].lower() in ['python', 'javascript', 'java', 'cpp', 'html', 'css', 'sql', 'bash', 'json', 'yaml']:
                    language = lines[0].lower()
                    code_content = '\n'.join(lines[1:])
                
                if code_content.strip():
                    try:
                        syntax = Syntax(
                            code_content, 
                            language, 
                            theme="monokai", 
                            line_numbers=True,
                            background_color="default"
                        )
                        
                        console.print("\n")
                        console.print(Panel(
                            syntax,
                            title=f"💻 Generated Code ({language.upper()})",
                            border_style="green",
                            subtitle=f"Lines: {len(code_content.splitlines())}"
                        ))
                        
                        # Enhanced save options
                        if Confirm.ask(f"💾 Save this {language} code to file?", default=False):
                            self.save_code_interactively(code_content, language)
                            
                    except Exception as e:
                        # Fallback to plain text
                        console.print(Panel(
                            code_content,
                            title=f"💻 Code ({language.upper()})",
                            border_style="green"
                        ))
    
    def display_text_response(self, response: AIResponse, query: str):
        """Display text-only response"""
        clean_content = self.clean_markdown_text(response.content)
        
        # Split long responses into readable chunks
        if len(clean_content) > 1000:
            chunks = self.split_text_intelligently(clean_content)
            for i, chunk in enumerate(chunks):
                title = f"💬 {response.provider} Response"
                if len(chunks) > 1:
                    title += f" (Part {i+1}/{len(chunks)})"
                    
                console.print(Panel(
                    chunk,
                    title=title,
                    border_style="blue"
                ))
                console.print()
        else:
            console.print(Panel(
                clean_content,
                title=f"💬 {response.provider} ({response.model})",
                border_style="blue"
            ))
    
    def clean_markdown_text(self, text: str) -> str:
        """Clean markdown for terminal display"""
        # Remove markdown formatting while preserving structure
        clean_text = text
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)  # Bold
        clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)      # Italic
        clean_text = re.sub(r'#{1,6}\s?', '', clean_text)         # Headers
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)       # Extra newlines
        return clean_text.strip()
    
    def split_text_intelligently(self, text: str, max_chunk_size: int = 800) -> List[str]:
        """Split text into readable chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def save_code_interactively(self, code_content: str, language: str):
        """Interactive code saving with enhanced features"""
        extension_map = {
            'python': 'py', 'javascript': 'js', 'typescript': 'ts',
            'java': 'java', 'cpp': 'cpp', 'c': 'c', 'html': 'html',
            'css': 'css', 'sql': 'sql', 'bash': 'sh', 'json': 'json',
            'yaml': 'yml', 'xml': 'xml'
        }
        
        default_ext = extension_map.get(language, 'txt')
        filename = Prompt.ask(
            "Enter filename", 
            default=f"ai_generated_code.{default_ext}"
        )
        
        try:
            # Create backup if file exists
            if os.path.exists(filename):
                backup_name = f"{filename}.backup_{int(time.time())}"
                shutil.copy2(filename, backup_name)
                console.print(f"[yellow]📦 Created backup: {backup_name}[/]")
            
            # Write file with enhanced metadata
            header_comment = self.generate_code_header(language)
            final_content = f"{header_comment}\n{code_content}"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(final_content)
            
            console.print(f"[green]✅ Code saved to {filename}[/]")
            
            # Offer additional actions
            if Confirm.ask("🚀 Open file in default editor?", default=False):
                try:
                    os.system(f'code "{filename}"' if shutil.which('code') else f'notepad "{filename}"')
                except:
                    console.print("[yellow]⚠️ Could not open editor[/]")
                    
        except Exception as e:
            console.print(f"[red]❌ Failed to save code: {str(e)}[/]")
    
    def generate_code_header(self, language: str) -> str:
        """Generate appropriate code header"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        comment_styles = {
            'python': f'"""\nGenerated by AI Terminal Pal v{self.version}\nCreated: {timestamp}\nProvider: {self.ai_provider.name if self.ai_provider else "Unknown"}\n"""',
            'javascript': f'/*\nGenerated by AI Terminal Pal v{self.version}\nCreated: {timestamp}\nProvider: {self.ai_provider.name if self.ai_provider else "Unknown"}\n*/',
            'java': f'/*\nGenerated by AI Terminal Pal v{self.version}\nCreated: {timestamp}\nProvider: {self.ai_provider.name if self.ai_provider else "Unknown"}\n*/',
            'cpp': f'/*\nGenerated by AI Terminal Pal v{self.version}\nCreated: {timestamp}\nProvider: {self.ai_provider.name if self.ai_provider else "Unknown"}\n*/',
            'html': f'<!-- Generated by AI Terminal Pal v{self.version} | Created: {timestamp} -->',
            'css': f'/* Generated by AI Terminal Pal v{self.version} | Created: {timestamp} */',
            'sql': f'-- Generated by AI Terminal Pal v{self.version}\n-- Created: {timestamp}',
            'bash': f'#!/bin/bash\n# Generated by AI Terminal Pal v{self.version}\n# Created: {timestamp}'
        }
        
        return comment_styles.get(language, f'# Generated by AI Terminal Pal v{self.version}\n# Created: {timestamp}')
    
    def display_response_metadata(self, response: AIResponse):
        """Display enhanced response metadata"""
        if not self.config.get('show_stats', True):
            return
            
        metadata_items = []
        
        if response.tokens_used:
            metadata_items.append(f"🔢 Tokens: {response.tokens_used:,}")
        
        if response.response_time:
            metadata_items.append(f"⏱️ Time: {response.response_time:.2f}s")
            
        if response.cost:
            metadata_items.append(f"💰 Cost: ${response.cost:.4f}")
            
        metadata_items.append(f"🕒 {datetime.datetime.fromisoformat(response.timestamp).strftime('%H:%M:%S')}")
        
        if metadata_items:
            console.print(f"[dim cyan]{'  |  '.join(metadata_items)}[/]")
    
    # Clear screen with enhanced banner
    def clear_screen(self, args):
        """Clear screen and show banner"""
        self.display_enhanced_banner()
    
    def exit_app(self, args):
        """Enhanced exit with statistics"""
        # Calculate session statistics
        session_duration = datetime.datetime.now() - self.performance_stats['session_start']
        
        exit_panel = Panel.fit(
            f"[bold green]Thanks for using AI Terminal Pal v{self.version}![/]\n\n"
            f"[cyan]Session Statistics:[/]\n"
            f"• Duration: {str(session_duration).split('.')[0]}\n"
            f"• Queries: {self.performance_stats['total_queries']}\n"
            f"• Total Tokens: {self.performance_stats['total_tokens']:,}\n"
            f"• Avg Response Time: {self.performance_stats['avg_response_time']:.2f}s\n\n"
            f"[yellow]💾 Session automatically saved[/]\n"
            f"[blue]Made with 💟 by Vishnupriyan P R :)[/]",
            title="👋 Goodbye!",
            border_style="green"
        )
        console.print(exit_panel)
        
        # Auto-save final session
        self.auto_save_session()
        sys.exit(0)
    
    def auto_save_session(self):
        """Automatically save session data"""
        try:
            session_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'performance_stats': self.performance_stats,
                'session_log': self.session_log[-50:],  # Keep last 50 entries
                'error_log': self.error_log[-20:],      # Keep last 20 errors
                'config_snapshot': self.config
            }
            
            session_file = self.config_dir / "sessions" / f"session_{int(time.time())}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
    
    # Placeholder implementations for remaining methods
    async def test_ai_connection(self):
        """Test AI connection"""
        if not self.ai_provider:
            return False
        try:
            response = await self.ai_provider.query("Hello! Please respond with 'Connection test successful'")
            return "successful" in response.content.lower()
        except:
            return False
    
    def configure_settings(self, args): 
        console.print("[yellow]⚠️ Advanced settings configuration coming in next update![/]")
    
    def switch_provider(self, args): 
        console.print("[yellow]⚠️ Provider switching interface coming in next update![/]")
    
    def change_theme(self, args): 
        if args and args[0] in self.theme_manager.themes:
            self.theme_manager.set_theme(args[0])
            self.config['theme'] = args[0]
            self.save_config()
            console.print(f"[green]✅ Theme changed to {args[0].title()}[/]")
            self.clear_screen([])
        else:
            console.print(f"[yellow]Available themes: {', '.join(self.theme_manager.themes.keys())}[/]")
    
    def customize_interface(self, args): 
        console.print("[yellow]⚠️ Interface customization coming in next update![/]")
    
    def quick_start_guide(self, args): 
        console.print("[yellow]⚠️ Interactive quick start guide coming in next update![/]")
    
    def show_pro_tips(self, args): 
        console.print("[yellow]⚠️ Pro tips system coming in next update![/]")
    
    def show_shortcuts(self, args): 
        console.print("[yellow]⚠️ Shortcuts reference coming in next update![/]")
    
    async def start_enhanced_chat(self, args): 
        console.print("[yellow]⚠️ Enhanced chat mode coming in next update![/]")
    
    def explain_with_ai(self, args): 
        console.print("[yellow]⚠️ AI explanation feature coming in next update![/]")
    
    def generate_code(self, args): 
        console.print("[yellow]⚠️ Code generation feature coming in next update![/]")
    
    def improve_code(self, args): 
        console.print("[yellow]⚠️ Code improvement feature coming in next update![/]")
    
    def translate_code(self, args): 
        console.print("[yellow]⚠️ Code translation feature coming in next update![/]")
    
    def optimize_code(self, args): 
        console.print("[yellow]⚠️ Code optimization feature coming in next update![/]")
    
    def brainstorm_session(self, args): 
        console.print("[yellow]⚠️ Brainstorming session feature coming in next update![/]")
    
    # Continue with placeholder implementations for all remaining methods...
    def attach_file(self, args): 
        console.print("[yellow]⚠️ File attachment feature coming in next update![/]")
    
    def read_file_enhanced(self, args): 
        console.print("[yellow]⚠️ Enhanced file reading coming in next update![/]")
    
    def write_file_enhanced(self, args): 
        console.print("[yellow]⚠️ Enhanced file writing coming in next update![/]")
    
    def edit_with_ai(self, args): 
        console.print("[yellow]⚠️ AI-powered editing coming in next update![/]")
    
    def backup_file(self, args): 
        console.print("[yellow]⚠️ File backup feature coming in next update![/]")
    
    def restore_file(self, args): 
        console.print("[yellow]⚠️ File restore feature coming in next update![/]")
    
    def compare_files(self, args): 
        console.print("[yellow]⚠️ File comparison feature coming in next update![/]")
    
    def scan_project_enhanced(self, args): 
        console.print("[yellow]⚠️ Enhanced project scanning coming in next update![/]")
    
    def analyze_project(self, args): 
        console.print("[yellow]⚠️ Project analysis feature coming in next update![/]")
    
    def analyze_dependencies(self, args): 
        console.print("[yellow]⚠️ Dependency analysis coming in next update![/]")
    
    def project_metrics(self, args): 
        console.print("[yellow]⚠️ Project metrics feature coming in next update![/]")
    
    def display_project_tree(self, args): 
        console.print("[yellow]⚠️ Project tree display coming in next update![/]")
    
    def search_project(self, args): 
        console.print("[yellow]⚠️ Project search feature coming in next update![/]")
    
    def refactor_project(self, args): 
        console.print("[yellow]⚠️ Project refactoring coming in next update![/]")
    
    def debug_with_ai(self, args): 
        console.print("[yellow]⚠️ AI debugging feature coming in next update![/]")
    
    def generate_tests(self, args): 
        console.print("[yellow]⚠️ Test generation feature coming in next update![/]")
    
    def lint_code(self, args): 
        console.print("[yellow]⚠️ Code linting feature coming in next update![/]")
    
    def format_code(self, args): 
        console.print("[yellow]⚠️ Code formatting feature coming in next update![/]")
    
    def generate_documentation(self, args): 
        console.print("[yellow]⚠️ Documentation generation coming in next update![/]")
    
    def api_tools(self, args): 
        console.print("[yellow]⚠️ API tools feature coming in next update![/]")
    
    def security_analysis(self, args): 
        console.print("[yellow]⚠️ Security analysis coming in next update![/]")
    
    def performance_analysis(self, args): 
        console.print("[yellow]⚠️ Performance analysis coming in next update![/]")
    
    def export_enhanced(self, args): 
        console.print("[yellow]⚠️ Enhanced export feature coming in next update![/]")
    
    def generate_pdf_report(self, args): 
        console.print("[yellow]⚠️ PDF report generation coming in next update![/]")
    
    def generate_project_report(self, args): 
        console.print("[yellow]⚠️ Project report generation coming in next update![/]")
    
    def show_detailed_stats(self, args): 
        console.print("[yellow]⚠️ Detailed statistics coming in next update![/]")
    
    def show_enhanced_history(self, args): 
        console.print("[yellow]⚠️ Enhanced history view coming in next update![/]")
    
    def show_logs(self, args): 
        console.print("[yellow]⚠️ Log viewer coming in next update![/]")
    
    def show_system_status(self, args): 
        console.print("[yellow]⚠️ System status feature coming in next update![/]")
    
    def system_monitor(self, args): 
        console.print("[yellow]⚠️ System monitoring coming in next update![/]")
    
    def copy_to_clipboard(self, args): 
        console.print("[yellow]⚠️ Clipboard operations coming in next update![/]")
    
    def paste_from_clipboard(self, args): 
        console.print("[yellow]⚠️ Clipboard operations coming in next update![/]")
    
    def check_updates(self, args): 
        console.print("[green]✅ You're running AI Terminal Pal v2.0 Supreme - Latest version![/]")
    
    def run_benchmarks(self, args): 
        console.print("[yellow]⚠️ Benchmarking feature coming in next update![/]")
    
    def restart_app(self, args): 
        console.print("[blue]🔄 Restarting AI Terminal Pal...[/]")
        python = sys.executable
        os.execl(python, python, *sys.argv)
    
    async def process_command(self, command_line: str):
        """Enhanced command processing"""
        parts = command_line.strip().split()
        if not parts:
            return
            
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command in self.commands:
            try:
                # Check if command is async
                if asyncio.iscoroutinefunction(self.commands[command]):
                    await self.commands[command](args)
                else:
                    self.commands[command](args)
            except Exception as e:
                error_msg = f"Command '{command}' failed: {str(e)}"
                self.error_log.append(error_msg)
                console.print(f"[red]❌ {error_msg}[/]")
                logger.error(f"Command error: {e}", exc_info=True)
        elif command.startswith('/'):
            console.print(f"[red]❌ Unknown command: {command}[/]")
            console.print("[yellow]💡 Type '/help' for all commands or '/nav' for navigation[/]")
        else:
            # Treat as AI query
            await self.ask_ai_enhanced(parts)
    
    def run(self):
        """Enhanced main application loop"""
        self.display_enhanced_banner()
        
        # Check setup status
        if not self.config.get("current_provider"):
            console.print(Panel.fit(
                "[bold yellow]⚠️ First-time setup required[/]\n"
                "[cyan]Run '/setup' to configure your AI providers and preferences[/]",
                title="🚀 Setup Required",
                border_style="yellow"
            ))
        
        # Main interaction loop
        while True:
            try:
                # Enhanced prompt with better styling
                provider_info = "No AI"
                if self.ai_provider:
                    provider_info = f"{self.ai_provider.name}:{self.ai_provider.model}"
                
                # Create elegant prompt
                primary_color = self.theme_manager.get_color('primary')
                secondary_color = self.theme_manager.get_color('secondary')
                accent_color = self.theme_manager.get_color('accent')
                
                prompt_line = (
                    f"{primary_color}┌─[{secondary_color}AI-Pal{primary_color}]─"
                    f"[{accent_color}v{self.version}{primary_color}]─"
                    f"[{secondary_color}{provider_info}{primary_color}]"
                )
                console.print(f"{prompt_line}{Style.RESET_ALL}")
                
                user_input = input(f"{primary_color}└─$ {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                # Process command
                asyncio.run(self.process_command(user_input))
                console.print()  # Add spacing
                
            except KeyboardInterrupt:
                console.print(f"\n{self.theme_manager.get_color('warning')}💡 Use '/exit' to quit gracefully{Style.RESET_ALL}")
            except EOFError:
                break
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.error_log.append(error_msg)
                console.print(f"[red]❌ {error_msg}[/]")
                logger.error(f"Unexpected error: {e}", exc_info=True)

def main():
    """Enhanced application entry point"""
    try:
        # Startup banner
        console.print("[bold blue]🚀 Starting AI Terminal Pal v2.0 Supreme Edition...[/]")
        
        # Initialize and run
        app = AITerminalPal()
        app.run()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Application interrupted by user[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Failed to start AI Terminal Pal: {str(e)}[/]")
        logger.error(f"Startup error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
