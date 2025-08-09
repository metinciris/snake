Snake Game

Otomatik ekran koruması şeklinde çalışan yılan oyunu.
Yeni versiyonlar eklenmektedir. Snake_2.py Grok 4 Heavy modeli.

Python

A modern take on the classic Snake game, featuring an intelligent AI, smooth visuals, and high score tracking. The snake moves as a continuous, tube-like structure with rounded joints and a gradually fading tail, designed to avoid traps and maximize survival.
Features

Intelligent AI: Uses A* pathfinding, BFS, and flood-fill algorithms to navigate to food while avoiding traps and ensuring long-term survival by maintaining access to the tail.
Smooth Visuals: The snake is rendered as a continuous tube with rounded joints, a bright green head, and a tail that fades gradually over the last 10 segments (from RGB(0,255,0) to RGB(0,150,0)).
High Scores: Tracks all-time and daily high scores (Python version only).
Browser Support: Playable in the browser via an HTML version using Pyodide.
Smooth Movement: Runs at 30 FPS for fluid motion without jitter.

Requirements
Python Version

Python 3.13+
Pygame 2.6.1+
Install dependencies:pip install pygame


Code Structure

AI Logic (find_path, a_star, bfs, flood_fill): Ensures the snake takes safe paths to food, prioritizes larger free areas, and maintains access to the tail after eating.
Rendering: Uses pygame.draw.line and pygame.draw.circle to create a smooth, tube-like snake with rounded joints and a fading tail.
High Scores (Python only): Stored in snake_highscores.txt.

Notes

The AI is designed to play indefinitely until the grid is nearly full, avoiding traps and ensuring survival.
The HTML version is slightly simplified (no file I/O) but retains the same gameplay and visuals.
For contributions or issues, please open a pull request or issue on GitHub.
