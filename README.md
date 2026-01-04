#  AI Terminal Pal v2.0 – Supreme Developer Edition

> _"Your all-in-one terminal-based AI dev sidekick — engineered for speed, clarity, and control."_  
> ✨ Powered by: **GPT‑4o**, **Claude 3 Opus**, **Gemini 1.5 Pro**, **Groq**, **Mistral**, **Ollama** and more...  
> 💡 Designed & crafted with precision by **Vishnupriyan P R**

> [![Built on - Python](https://img.shields.io/badge/Built--on-Python-blue)](#)
> ![Maintained - yes](https://img.shields.io/badge/AI%20Engines-GPT4o%20|%20Claude%20|%20Gemini%20|%20Groq%20|%20Mistral%20|%20Ollama%20|-purple)
> ![Terminal App](https://img.shields.io/badge/Interface-Terminal%20CLI-2F2F2F)
> [![License-MIT](https://img.shields.io/badge/License-MIT-red)](#)





---

## Overview :

AI Terminal Pal isn't your typical CLI toy — it's a full-blown developer productivity engine built into your terminal. Whether you're asking quick questions, analyzing codebases, generating boilerplate, or debugging a tangled mess, this tool understands your workflow. With multi-AI provider support, blazing speed, code-aware context building, and beautiful terminal output, it adapts to how *you* work.

---
<p align = center>
  
  ![Alt](https://repobeats.axiom.co/api/embed/a2078e4de9566f17d2b77f722c73de77d033a1dc.svg "Repobeats analytics image")

</p>

## Features at a Glance

- 🤖 **Multi-AI Support:** GPT-4o, Claude 3, Gemini 1.5, Mistral, Groq Llama3, Ollama, and more
- 🧠 **Contextual Intelligence:** File-aware responses using `@filename.py` or auto-scan
- 🖼️ **Themes & UI:** Dynamic banner, themed layouts (Professional, Forest, Ocean, Minimal)
- 📋 **Clipboard Smartness:** Auto-copy responses; paste into code right away
- 📦 **Project Integration:** Code analysis, tree view, metrics, and documentation generation
- 📊 **Live Stats:** Track tokens, cost, speed, and query logs
- 🧪 **Built-in Dev Tools:** Linting, testing, formatting, and debugging (AI-assisted)
- 📤 **Export Everything:** Generate PDFs, logs, reports, or backups from the CLI

---

##  Folder Structure

```
ai-terminal-pal-/📂
├── ai_terminal_pal.py    # Main app
├── .env                    # API keys (optional, or added during /setup)
├── README.md               # You're reading this
├── requirements.txt        # All dependencies
```

---

##  Installation

```bash
git clone https://github.com/vishnupriyanpr/Terminal-Pal
cd Terminal-Pal
pip install -r requirements.txt
```

Set your API keys in `.env` or input manually via `/setup`:

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
CLAUDE_API_KEY=...
```

---

##  Usage

```bash
python ai_terminal_pal.py
```

### Command Examples:

```bash
/setup                 # Launch the setup wizard
/ask What is LangChain?   # Ask a quick AI query
/scan                  # Analyze current project
/theme forest          # Apply the 'forest' theme
/attach app.py         # Attach file for AI context
```

> Use `/help` or `/nav` to explore all commands.

---

##  Supported Providers

This app is multi-AI out of the box. You can pick your preferred model during `/setup`.

| Provider  | Example Models                 | Max Context     | Speed       |
|-----------|--------------------------------|------------------|-------------|
| **OpenAI**    | `gpt-4o`, `gpt-4-turbo`        | Up to 128k      | 🔥 Fast     |
| **Claude**    | `opus`, `sonnet`, `haiku`      | Up to 200k      | 🚀 Blazing  |
| **Gemini**    | `1.5-pro`, `flash`, `pro`      | Up to 2M        | ⚡ Snappy   |
| **Groq**      | `llama3`, `mixtral`, `gemma`   | ~32k            | ⚡ Ultra-fast |
| **Mistral**   | `codestral`, `mistral-large`   | ~32k            | 💡 Smart    |
| **Ollama**   | any `model` you initialize   | ~32k +            | 🏠 Local   |

---

##  Theming System

Customize the terminal look with:

- 💼 `Professional`
- 🌊 `Ocean`
- 🌿 `Forest`
- ⚫ `Minimal`

Change themes using:

```bash
/theme forest
```

---

##  Sample Session Stats

```bash
🧠 Provider: GPT-4o
⏱️ Time Taken: 1.45s
🔢 Tokens Used: 1,152
💰 Estimated Cost: $0.0042
📁 Files Contextualized: utils.py, routes.py
```

---

##  Command Categories

| Category       | Commands                                                   |
|----------------|------------------------------------------------------------|
| Setup          | `/setup`, `/provider`, `/theme`, `/config`                |
| AI Interaction | `/ask`, `/chat`, `/generate`, `/explain`, `/improve`      |
| Project Tools  | `/scan`, `/tree`, `/metrics`, `/analyze`, `/deps`         |
| File Ops       | `/attach`, `/read`, `/write`, `/edit`, `/backup`, `/compare` |
| Dev Utilities  | `/lint`, `/test`, `/debug`, `/docs`, `/format`            |
| Export/Logs    | `/pdf`, `/export`, `/report`, `/history`, `/logs`         |
| System         | `/status`, `/monitor`, `/clear`, `/exit`, `/update`       |

---

##  Privacy & Security

- API keys stored locally in your config directory or .env
- No telemetry, no tracking — fully offline except for API calls
- Full transparency in logging and history
- Optional backup/restore for peace of mind

---

##  Sample Interactions

```bash
/ask Give me a FastAPI boilerplate
/ask @views.py — optimize this
/ask Explain difference between multiprocessing and multithreading
```

---

##  License

This project is licensed under the Apache 2.0 License.

---

##  Credits

<div align="center">
  <table style="width:100%;">
    <tr>
      <td align="center" style="width:50%;">
        <a href="https://github.com/vishnupriyanpr">
                  <img src="https://github.com/vishnupriyanpr.png?size=120" width="120px;" alt="Vishnupriyan P R"/>
      </a>
      </td>
      <td align="center" style="width:50%;">
        <blockquote>
          <p>“Tools should disappear into the background and let you build.”</p>
          <footer>— Vishnupriyan P R, <i>caffeinated coder ☕</i></footer>
        </blockquote>
      </td>
    </tr>
  </table>
</div>

