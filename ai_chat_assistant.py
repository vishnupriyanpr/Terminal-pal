import requests
from rich.console import Console
from rich.prompt import Prompt

# Replace this with your actual API key
API_KEY = ""

console = Console()

def ask_gemini(prompt_text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": "AIzaSyB5NnM2PjjtyV7puN9DzU7p8_Vc43dUTzs"}
    data = {
        "contents": [
            {
                "parts": [{"text": prompt_text}]
            }
        ]
    }

    response = requests.post(url, headers=headers, params=params, json=data)
    
    if response.status_code == 200:
        result = response.json()
        try:
            return result['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return "No valid response received."
    else:
        return f"Error: {response.status_code} - {response.text}"

def main():
    console.print("[bold green]Welcome to your Gemini 2.0 Flash Chat Assistant! 🚀[/bold green]")
    
    while True:
        user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
        
        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold magenta]Goodbye![/bold magenta]")
            break
        
        console.print("[bold yellow]Thinking...[/bold yellow]")
        reply = ask_gemini(user_input)
        
        console.print(f"\n[bold cyan]Gemini:[/bold cyan] {reply}")

if __name__ == "__main__":
    main()
