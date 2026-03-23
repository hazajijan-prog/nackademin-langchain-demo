# Standardbibliotek

import re        # används för regex (hitta tider, dela text osv)
import json      # används för att tolka JSON-strängar
import ast       # fallback om JSON parsing misslyckas


# MCP / FastMCP setup
from fastmcp import FastMCP
from typing import Annotated
from pydantic import Field

# Logging (för debugging / requests)
from config.custom_logging_config import RequestLoggingMiddleware
from config.logging_config import configure_logging

# Starta logging
configure_logging()

# Skapa MCP-server
mcp = FastMCP("Planner Server")

# Lägg till middleware för att logga requests
mcp.add_middleware(RequestLoggingMiddleware())

# TOOL 1: Extract tasks
@mcp.tool()
def extract_tasks(
    text: Annotated[str, Field(description="Free text input describing tasks for the day")],
) -> list[str]:

    # Dela upp texten i meningar baserat på punkt eller komma
    sentences = re.split(r"[,.]", text)

    tasks = []

    # Loopa igenom varje mening
    for s in sentences:
        s = s.strip()
        
        # Kontrollera om meningen innehåller ett "task-ord"
        if any(word in s.lower() for word in ["möte", "plugga", "hämta", "träna", "handla"]):
            tasks.append(s.capitalize())

    return tasks

# TOOL 2: Extract times
@mcp.tool()
def extract_times(
    text: Annotated[str | list[str] | None, Field(description="Free text input")] = None,
    tasks: Annotated[list[str] | None, Field(description="Optional tasks list")] = None,
) -> list[str]:

    import re
    
    # Om text är en lista → platta ut till en sträng
    if isinstance(text, list):
        flattened = []
        for item in text:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        text = " ".join(flattened)
        
    # Om tasks finns, lägg till dem i texten
    if tasks:
        text = (text or "") + " " + " ".join(tasks)
        
    # Om inget finns, returnera tom lista
    if not text:
        return []
    
    # Hitta tider i format:
    # - kl 12:00
    # - 12:00
    # - 9-10
    return re.findall(r"(kl\s?\d{1,2}:\d{2}|\b\d{1,2}:\d{2}\b|\b\d{1,2}-\d{1,2}\b)", text.lower())

# TOOL 3: Extract durations
@mcp.tool()
def extract_durations(
    text: Annotated[str | list[str] | None, Field(description="Text")] = None,
    tasks: Annotated[list[str] | None, Field(description="Optional tasks")] = None,
    durations: Annotated[list[int] | None, Field(description="Optional durations")] = None, 
) -> list[str]:

    #  Om agenten redan skickar durations, använd dem direkt
    if durations:
        return [f"{d} minuter" for d in durations]
    
    # Hantera lista → sträng
    if isinstance(text, list):
        text = " ".join(text)
        
    # Lägg till tasks i text
    if tasks:
        text = (text or "") + " " + " ".join(tasks)

    if not text:
        return []
    # Hitta t.ex. "4 timmar", "30 minuter"
    return re.findall(r"\d+\s*(timmar|timme|minuter|minut)", text.lower())

# TOOL 4: Create schedule
@mcp.tool()
def create_schedule(
    tasks: Annotated[list[str | None] | str, Field(description="Tasks list")],
    times: Annotated[list[str | None] | str | None, Field(description="Times list")] = None,
) -> list[dict]:
    
    # Om tasks kommer som string → försök konvertera till lista
    if isinstance(tasks, str):
        try:
            tasks = json.loads(tasks)
        except Exception:
            tasks = re.findall(r'[\w\s]+', tasks)
    # Ta bort tomma tasks
    tasks = [t for t in tasks if t and str(t).strip()]

    # Times
    if times is None:
        times = []
        
    # Säkerställ att allt är string
    if isinstance(times, list):
        times = [str(t) if isinstance(t, int) else t for t in times]
        
    # Om times är string → försök konvertera
    if isinstance(times, str):
        try:
            times = json.loads(times)
        except Exception:
            times = re.findall(r'kl\s?\d{1,2}|\d{1,2}:\d{2}', times)

    times = [t if t not in ("null", "None") else None for t in times]
    
    # Ta bort "None" som text
    times = [
    t if isinstance(t, str) and ("kl" in t or ":" in t)
    else None
    for t in times
]

    # Bygg schema
    return [
        {
            "time": times[i] if i < len(times) else None,
            "task": task
        }
        for i, task in enumerate(tasks)
    ]

# TOOL 5: Prioritize tasks (extra)
@mcp.tool()
def prioritize_tasks(
    tasks: Annotated[list[str], Field(description="Tasks list")],
    keywords: Annotated[list[str], Field(description="Keywords for priority")],
) -> list[dict]:
    
    # Enkel scoring baserat på nyckelord
    def score(task: str) -> int:
        return sum(1 for kw in keywords if kw.lower() in task.lower())

    scored = [{"task": t, "priority": score(t)} for t in tasks]
    
    # Sortera högst prioritet först
    return sorted(scored, key=lambda x: -x["priority"])

# TOOL 6: Format schedule (final output)
@mcp.tool()
def format_schedule(
    schedule: Annotated[list[dict] | str, Field(description="Schedule list")],
) -> str:

    import json

    # Om schedule är string, konvertera

    if isinstance(schedule, str):
        try:
            schedule = json.loads(schedule)
        except Exception:
            try:
                schedule = ast.literal_eval(schedule)
            except Exception:
                return "Kunde inte tolka schemat"

    lines = []
    
    # Formatera varje rad
    for entry in schedule:
        task = entry.get("task", "").strip()
        time = entry.get("time")

        if time:
            lines.append(f"{time.replace('kl ', '')} - {task}")
        else:
            lines.append(f"(ingen tid) - {task}")

    # Bygg slutlig output
    output = "Din dag har strukturats på följande sätt:\n\n"

    for i, line in enumerate(lines, 1):
        output += f"{i}. {line}\n"

    output += "\nSpara detta schema för att planera din dag!"

    return output

# Starta MCP-servern
if __name__ == "__main__":
    import asyncio

    asyncio.run(
        mcp.run_http_async(
            host="0.0.0.0",
            port=8003,
            log_level="warning",
        )
    )