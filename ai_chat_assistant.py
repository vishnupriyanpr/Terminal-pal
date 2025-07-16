import requests
import time
import re
import pyperclip
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.align import Align
from rich.prompt import Prompt

#-------------------------------------Put your gemini api key here :)------------------------------------------
API_KEY = ""
#-------------------------------------Put your gemini api key here :)------------------------------------------

console = Console()
def get_single_keypress():
    try:
        import msvcrt
        return msvcrt.getch().decode("utf-8")
    except ImportError:
        import termios, tty, sys
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
def render_conversation(conversation, line_delay=0.01):
    """
    conversation: list of dict { "type": "text"/"code", "content": str }
    Prints full conversation without clearing repeatedly.
    Animates only the latest text part line by line.
    """
    console.clear()
    # Render all previous parts fully (without animation)
    for i, part in enumerate(conversation[:-1]):
        if part["type"] == "text":
            console.print(
                Align.center(
                    Panel(Text(part["content"], style="bold white"), title="🤖 Gemini says", border_style="bright_magenta", padding=(1, 4), width=100)
                )
            )
        else:  # code block
            syntax = Syntax(part["content"], "python", theme="monokai", line_numbers=True, indent_guides=True)
            console.print(
                Align.center(
                    Panel(syntax, title="💻 Code", border_style="cyan", padding=(1, 2), width=100)
                )
            )

    # Animate only the latest part (which can be multiple: text + code blocks)
    latest = conversation[-1]

    if latest["type"] == "text":
        lines = latest["content"].splitlines()
        animated_text = Text()
        for line in lines:
            animated_text.append(Text(line, style="bold white") + "\n")
            console.clear()
            # Render full conversation except last text animated with new lines growing
            for part in conversation[:-1]:
                if part["type"] == "text":
                    console.print(
                        Align.center(
                            Panel(Text(part["content"], style="bold white"), title="🤖 Gemini says", border_style="bright_magenta", padding=(1, 4), width=100)
                        )
                    )
                else:
                    syntax = Syntax(part["content"], "python", theme="monokai", line_numbers=True, indent_guides=True)
                    console.print(
                        Align.center(
                            Panel(syntax, title="💻 Code", border_style="cyan", padding=(1, 2), width=100)
                        )
                    )
            # Render animated last text block growing
            console.print(
                Align.center(
                    Panel(animated_text, title="🤖 Gemini says", border_style="bright_magenta", padding=(1, 4), width=100)
                )
            )
            time.sleep(line_delay)

    else:
        # Latest part is code block - just print it normally (no animation)
        syntax = Syntax(latest["content"], "python", theme="monokai", line_numbers=True, indent_guides=True)
        console.print(
            Align.center(
                Panel(syntax, title="💻 Code", border_style="cyan", padding=(1, 2), width=100)
            )
        )

def ask_gemini(prompt_text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    data = {"contents": [{"parts": [{"text": prompt_text}]}]}

    try:
        response = requests.post(url, headers=headers, params=params, json=data, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {e}"

    try:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError):
        return "❌ Gemini response error: Invalid structure."

def main():
    console.clear()
    console.print(
        Align.center(
            Panel(
                "[bold green]                Hello, I'm your Coding Pal ! (Powered by Gemini 2.0) 🚀[/bold green]",
                title="✨ AI Terminal Pal ! ✨",
                border_style="green",
                padding=(1, 4),
                width=100,
            )
        )
    )

    conversation = []

    try:
        while True:
            user_input = Prompt.ask("\n[bold yellow]Your Prompt (type 'exit' to quit)[/bold yellow]")
            if user_input.strip().lower() in ["exit", "quit"]:
                console.print(
                    Align.center(
                        Panel("👋 [bold magenta]Goodbye, Creator![/bold magenta]", border_style="magenta", width=100)
                    )
                )
                break

            console.print("\n[bold yellow]Thinking...[/bold yellow]")
            reply = ask_gemini(user_input)

            # Parse reply into text parts and code blocks
            code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", reply, re.DOTALL)
            text_parts = re.split(r"```(?:python)?\n?.*?```", reply, flags=re.DOTALL)
            # Add all text parts as single text block (combined)
            combined_text = "\n\n".join(part.strip() for part in text_parts if part.strip())
            if combined_text:
                conversation.append({"type": "text", "content": combined_text})
            # Add each code block separately
            for cb in code_blocks:
                conversation.append({"type": "code", "content": cb.strip()})

            render_conversation(conversation)

            # For each new code block, allow copy prompt
            for part in conversation:
                if part["type"] == "code":
                    console.print(Align.center("[bold green]Press 'c' to copy the code above or any other key to continue[/bold green]"))
                    key = get_single_keypress()
                    if key.lower() == "c":
                        pyperclip.copy(part["content"])
                        console.print(Align.center("[bold cyan]✅ Code copied to clipboard![/bold cyan]"))
                        time.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted! Exiting...[/bold red]")

if __name__ == "__main__":
    main()
