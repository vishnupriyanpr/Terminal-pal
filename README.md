#  Terminal-Pal : Your AI Chat Companion in the Terminal

> _"An elegant, no-nonsense AI-powered assistant for your command-line grind."_  
> Powered by **Gemini 2.0** • Built with ❤️ by **Vishnupriyan**

![Built with Gemini](https://img.shields.io/badge/Built%20with-Gemini%202.0-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

---

##  Overview 💡

**Terminal-Pal** is a minimalist AI assistant that lives in your terminal.  
Built using Python, the `rich` library, and Gemini 2.0's API, it's your go-to pal when you want instant help, explanations, or code — all within your CLI.

Whether you're debugging, learning, or just exploring ideas, Terminal-Pal turns your terminal into an interactive AI chat space.

---

##  Features 🔥

-  **Powered by Gemini 2.0** (via Google Generative Language API)
-  Beautiful CLI output using `rich`
-  Smartly formatted and animated responses
-  Code block rendering with syntax highlighting
-  Copy-to-clipboard functionality for code
-  Conversations with memory (scroll-friendly)
-  Portable single-file script — plug and play

---

##  Workflow Diagram 🧠

```mermaid
graph TD
A[User Prompt] --> B[Terminal-Pal]
B --> C[Gemini API Request]
C --> D[Receive Text + Code]
D --> E[Render with rich]
E --> F[Show in Terminal]
F --> G{Code Block Present?}
G -- Yes --> H[Offer to Copy Code]
G -- No --> I[Wait for Next Prompt]
```

---

##  Folder Structure 📁

```shell
terminal-pal/
├── terminal_pal.py         # Main app script
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
```

---

##  Installation & Setup 🛠️

###  Step 1: Install Requirements 📦

```bash
pip install rich requests pyperclip
```

> Ensure Python 3.8+ is installed.

###  Step 2: Add Your Gemini API Key 🔐

Edit `terminal_pal.py` and replace:

```python
API_KEY = "PUT_YOUR_API_KEY_HERE"
```

with your API key from [Google AI Studio](https://makersuite.google.com/).

###  Step 3: Run the Script 🚀

```bash
python terminal_pal.py
```

You’re now ready to chat with your AI pal right from your terminal!

---
## 📽️ Execution Demo

[![Watch Execution Video](https://imgur.com/YOUR_PREVIEW_IMG.png)](https://github.com/user-attachments/assets/3d4a48c7-b82c-4227-811b-f987059907b0)



And if there's code, you'll get this:
```
Press 'c' to copy the code above or any other key to continue
```

---






##  License 📜

Licensed under the [MIT License](LICENSE).

---

##  Credits ✨

Built by **Vishnupriyan P R**  
If you like this, feel free to ⭐ the repo or fork it!  
Got feature ideas? Hit me up or open an issue!

---

