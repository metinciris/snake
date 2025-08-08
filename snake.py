import pygame
import sys
import random
import time
from datetime import date
from collections import deque
import heapq

# Pygame'i baslat
pygame.init()

# Tam ekran boyutlarini al
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Yilan Ekran Koruyucu")

# Renkler
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)  # Parlak yesil kafa icin
RED = (255, 0, 0)
GRAY = (100, 100, 100)  # Soluk gri icin skorlar

# Oyun ayarlari
BLOCK_SIZE = min(WIDTH, HEIGHT) // 40  # Ekranı daha iyi kullanmak icin kucuk bloklar
SNAKE_WIDTH = int(BLOCK_SIZE * 0.8)  # Yilan kalinligini azalt
FPS = 30  # Daha yumusak hareket icin

# Yonler
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Yuksek skor dosyasi
HIGHSCORE_FILE = "snake_highscores.txt"

def load_highscores():
    try:
        with open(HIGHSCORE_FILE, "r") as f:
            lines = f.readlines()
            all_time = int(lines[0].strip())
            daily_date = lines[1].strip()
            daily_score = int(lines[2].strip())
            today = str(date.today())
            if daily_date != today:
                daily_score = 0
            return all_time, daily_score
    except:
        return 0, 0

def save_highscores(all_time, daily_score):
    today = str(date.today())
    with open(HIGHSCORE_FILE, "w") as f:
        f.write(f"{all_time}\n")
        f.write(f"{today}\n")
        f.write(f"{daily_score}\n")

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def a_star(snake, target, grid_width, grid_height, ignore_tail=False):
    head = snake[0]
    pq = []
    heapq.heappush(pq, (0 + manhattan(head, target), 0, head, []))
    visited = set(snake[:-1] if ignore_tail else snake)

    while pq:
        f, g, pos, path = heapq.heappop(pq)
        if pos == target:
            return path

        for d in [UP, RIGHT, DOWN, LEFT]:
            nx, ny = pos[0] + d[0], pos[1] + d[1]
            if 0 <= nx < grid_width and 0 <= ny < grid_height and (nx, ny) not in visited:
                visited.add((nx, ny))
                new_path = path + [d]
                new_g = g + 1
                h = manhattan((nx, ny), target)
                heapq.heappush(pq, (new_g + h, new_g, (nx, ny), new_path))

    return None

def bfs(snake, target, grid_width, grid_height, ignore_tail=False):
    head = snake[0]
    queue = deque([(head, [])])
    visited = set(snake[:-1] if ignore_tail else snake)

    while queue:
        pos, path = queue.popleft()
        if pos == target:
            return path

        for d in [UP, RIGHT, DOWN, LEFT]:
            nx, ny = pos[0] + d[0], pos[1] + d[1]
            if 0 <= nx < grid_width and 0 <= ny < grid_height and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [d]))

    return None

def flood_fill(snake, start, w, h):
    obstacles = set(snake[1:])  # Exclude head
    visited = set(obstacles)
    queue = deque()
    count = 0

    # Start from neighbors of start
    for d in [UP, DOWN, LEFT, RIGHT]:
        nx, ny = start[0] + d[0], start[1] + d[1]
        if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
            visited.add((nx, ny))
            queue.append((nx, ny))
            count += 1

    while queue:
        pos = queue.popleft()
        for d in [UP, DOWN, LEFT, RIGHT]:
            nx, ny = pos[0] + d[0], pos[1] + d[1]
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
                count += 1

    return count

def find_path(snake, food, grid_width, grid_height):
    head = snake[0]
    directions = [UP, DOWN, LEFT, RIGHT]
    safe_moves_to_food = []
    safe_moves = []

    for d in directions:
        nx, ny = head[0] + d[0], head[1] + d[1]
        if not (0 <= nx < grid_width and 0 <= ny < grid_height and (nx, ny) not in snake):
            continue

        if (nx, ny) == food:
            # Special case: immediate eat
            sim_snake = snake.copy()
            sim_snake.insert(0, (nx, ny))  # Eat, no pop
            area = flood_fill(sim_snake, (nx, ny), grid_width, grid_height)
            tail = sim_snake[-1]
            path_to_tail = bfs(sim_snake, tail, grid_width, grid_height, ignore_tail=True)
            if path_to_tail is not None:
                safe_moves_to_food.append((1, -area, d))
        else:
            # Normal case
            sim_snake = snake.copy()
            sim_snake.insert(0, (nx, ny))
            sim_snake.pop()
            area = flood_fill(sim_snake, (nx, ny), grid_width, grid_height)
            path_to_food = a_star(sim_snake, food, grid_width, grid_height)
            if path_to_food is not None:
                if len(path_to_food) == 0:
                    # Should not happen in normal case
                    continue
                food_sim_snake = sim_snake.copy()
                for move in path_to_food[:-1]:
                    f_head = food_sim_snake[0]
                    f_new_head = (f_head[0] + move[0], f_head[1] + move[1])
                    food_sim_snake.insert(0, f_new_head)
                    food_sim_snake.pop()
                f_head = food_sim_snake[0]
                f_new_head = (f_head[0] + path_to_food[-1][0], f_head[1] + path_to_food[-1][1])
                food_sim_snake.insert(0, f_new_head)
                tail = food_sim_snake[-1]
                path_to_tail = bfs(food_sim_snake, tail, grid_width, grid_height, ignore_tail=True)
                if path_to_tail is not None:
                    safe_moves_to_food.append((len(path_to_food) + 1, -area, d))

        safe_moves.append((-area, d))

    if safe_moves_to_food:
        safe_moves_to_food.sort()
        return safe_moves_to_food[0][2]

    if safe_moves:
        safe_moves.sort()
        return safe_moves[0][1]

    # Fallback
    for d in directions:
        nx, ny = head[0] + d[0], head[1] + d[1]
        if 0 <= nx < grid_width and 0 <= ny < grid_height and (nx, ny) not in snake:
            return d

    return None

def draw_text(text, size, color, x, y, alpha=255):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    if alpha < 255:
        text_surface.set_alpha(alpha)
    text_rect = text_surface.get_rect(center=(x, y))
    SCREEN.blit(text_surface, text_rect)

def show_records(all_time, daily):
    SCREEN.fill(BLACK)
    draw_text("Yuksek Skorlar", 50, WHITE, WIDTH//2, HEIGHT//4)
    draw_text(f"Tum Zamanlar: {all_time}", 40, GREEN, WIDTH//2, HEIGHT//2)
    draw_text(f"Gunluk: {daily}", 40, GREEN, WIDTH//2, HEIGHT//2 + 50)
    pygame.display.flip()
    time.sleep(3)

def main():
    all_time_high, daily_high = load_highscores()
    show_records(all_time_high, daily_high)

    grid_width = WIDTH // BLOCK_SIZE
    grid_height = HEIGHT // BLOCK_SIZE

    while True:
        # Oyun baslat
        snake = [(grid_width // 2, grid_height // 2)]
        direction = RIGHT
        food = (random.randint(0, grid_width - 1), random.randint(0, grid_height - 1))
        score = 0
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

            # AI yon kararini ver
            next_dir = find_path(snake, food, grid_width, grid_height)
            if next_dir:
                direction = next_dir

            # Yilani hareket ettir
            head = snake[0]
            new_head = (head[0] + direction[0], head[1] + direction[1])

            # Carpisma kontrolu
            if (new_head[0] < 0 or new_head[0] >= grid_width or
                new_head[1] < 0 or new_head[1] >= grid_height or
                new_head in snake):
                running = False
                break

            snake.insert(0, new_head)

            # Yemek kontrolu
            if new_head == food:
                score += 1
                food = (random.randint(0, grid_width - 1), random.randint(0, grid_height - 1))
                while food in snake:
                    food = (random.randint(0, grid_width - 1), random.randint(0, grid_height - 1))
            else:
                snake.pop()

            # Cizim
            SCREEN.fill(BLACK)
            
            # Soluk yuksek skorlari ciz (yilanin altinda kalacak sekilde once ciz)
            draw_text(f"Tum Zamanlar: {all_time_high}", 30, GRAY, WIDTH - 150, 20, alpha=128)
            draw_text(f"Gunluk: {daily_high}", 30, GRAY, WIDTH - 150, 50, alpha=128)
            
            # Yilan noktalarini hesapla (merkezler)
            points = [(seg[0] * BLOCK_SIZE + BLOCK_SIZE // 2, seg[1] * BLOCK_SIZE + BLOCK_SIZE // 2) for seg in snake]
            
            # Govde cizgilerini ciz (baglantilar)
            for i in range(len(points) - 1):
                if i >= len(snake) - 10:
                    fade_factor = (len(snake) - i - 1) / 10.0
                    green_value = int(150 + 105 * fade_factor)
                    color = (0, green_value, 0)
                else:
                    color = GREEN
                pygame.draw.line(SCREEN, color, points[i], points[i+1], SNAKE_WIDTH)
            
            # Her nokta icin daire ciz (yuvarlak eklemler icin)
            for i, point in enumerate(points):
                if i == 0:
                    color = GREEN
                elif i >= len(snake) - 10:
                    fade_factor = (len(snake) - i) / 10.0
                    green_value = int(150 + 105 * fade_factor)
                    color = (0, green_value, 0)
                else:
                    color = GREEN
                pygame.draw.circle(SCREEN, color, point, SNAKE_WIDTH // 2)
            
            # Yemegi ciz
            pygame.draw.circle(SCREEN, RED, (food[0]*BLOCK_SIZE + BLOCK_SIZE//2, food[1]*BLOCK_SIZE + BLOCK_SIZE//2), BLOCK_SIZE//2)
            
            # Skoru ciz
            draw_text(f"Skor: {score}", 30, WHITE, WIDTH//2, 20)
            
            pygame.display.flip()

            clock.tick(FPS)

        # Oyun bitti, skorlari guncelle
        if score > all_time_high:
            all_time_high = score
        if score > daily_high:
            daily_high = score
        save_highscores(all_time_high, daily_high)

        # Oyun bitti ekrani
        SCREEN.fill(BLACK)
        draw_text("Oyun Bitti", 50, RED, WIDTH//2, HEIGHT//4)
        draw_text(f"Skor: {score}", 40, WHITE, WIDTH//2, HEIGHT//2)
        pygame.display.flip()
        time.sleep(2)
        show_records(all_time_high, daily_high)

if __name__ == "__main__":
    main()
