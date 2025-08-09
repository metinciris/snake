import pygame
import sys
import random
import time
from datetime import date
from collections import deque
import heapq
import math
import logging

# Setup logging for session scores
logging.basicConfig(filename='snake_scores.log', level=logging.INFO, format='%(asctime)s - Score: %(message)s')

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
GRAY = (200, 200, 200)  # Soluk gri icin skorlar (daha acik renk)
YELLOW = (255, 255, 0)  # Soru isareti icin hafif sari glow

# Oyun ayarlari
BLOCK_SIZE = min(WIDTH, HEIGHT) // 40  # Ekranı daha iyi kullanmak icin kucuk bloklar
SNAKE_WIDTH = int(BLOCK_SIZE * 0.8)  # Yilan kalinligini azalt
FPS = 60  # Hızlı olsun

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

def generate_hamiltonian_cycle(grid_width, grid_height):
    if grid_width % 2 == 1 and grid_height % 2 == 1:
        return None

    cycle = []
    for y in range(grid_height):
        if y % 2 == 0:
            for x in range(grid_width):
                cycle.append((x, y))
        else:
            for x in range(grid_width - 1, -1, -1):
                cycle.append((x, y))
    return cycle

def find_path(snake, food, grid_width, grid_height, h_cycle):
    head = snake[0]
    directions = [UP, DOWN, LEFT, RIGHT]
    safe_moves_to_food = []
    safe_moves_perfect = []
    safe_moves_fallback = []
    total_cells = grid_width * grid_height

    score = len(snake) - 1
    use_advanced_safety = score >= 400  # Activate safety at 400+

    # Early game: go directly to food
    if score < 10:
        path = a_star(snake, food, grid_width, grid_height)
        if path:
            return path[0]

    # Penalize backtracking to avoid oscillation
    current_dir = None
    backtrack_dir = None
    if len(snake) > 1:
        current_dir = snake[1]
        backtrack_dir = (-(head[0] - current_dir[0]), -(head[1] - current_dir[1]))
        directions = [d for d in directions if d != backtrack_dir]

    for d in directions:
        nx, ny = head[0] + d[0], head[1] + d[1]
        if not (0 <= nx < grid_width and 0 <= ny < grid_height and (nx, ny) not in snake):
            continue

        # Penalize moves near left/right boundaries early in the game
        edge_penalty = 0
        if score < 100:  # Apply penalty for first 100 moves
            if nx <= 1 or nx >= grid_width - 1:
                edge_penalty = 10

        is_eat = (nx, ny) == food
        sim_snake = snake.copy()
        sim_snake.insert(0, (nx, ny))
        if not is_eat:
            sim_snake.pop()

        area = flood_fill(sim_snake, (nx, ny), grid_width, grid_height)
        total_empty = total_cells - len(sim_snake)

        path_to_tail = None
        if len(sim_snake) > 1:
            tail = sim_snake[-1]
            path_to_tail = bfs(sim_snake, tail, grid_width, grid_height, ignore_tail=True)

        # Exploration incentive: prefer moves toward food or open areas
        exploration_score = area / total_empty if total_empty > 0 else 0
        path_len = 0 if is_eat else (len(a_star(sim_snake, food, grid_width, grid_height) or [float('inf')]))
        
        # Minimal randomization only at high scores
        random_factor = random.uniform(-0.05, 0.05) if use_advanced_safety else 0
        
        # Prioritize shorter paths, larger areas, exploration, avoid edges
        priority = (path_len + edge_penalty, -area, -exploration_score, random_factor)

        is_perfect = (area == total_empty)
        if path_to_tail is not None:
            if is_perfect or not use_advanced_safety:
                safe_moves_perfect.append((priority, d))
            else:
                safe_moves_fallback.append((priority, d))

        if is_eat and path_to_tail is not None:
            safe_moves_to_food.append((priority, d))

    if safe_moves_to_food:
        safe_moves_to_food.sort()
        return safe_moves_to_food[0][1]

    if safe_moves_perfect:
        safe_moves_perfect.sort()
        return safe_moves_perfect[0][1]

    # Hamiltonian cycle only at high scores
    if use_advanced_safety and h_cycle is not None and not safe_moves_fallback:
        current_index = -1
        for i, p in enumerate(h_cycle):
            if p == head:
                current_index = i
                break

        if current_index != -1:
            for offset in [1, -1]:
                next_index = (current_index + offset) % len(h_cycle)
                next_pos = h_cycle[next_index]
                dx, dy = next_pos[0] - head[0], next_pos[1] - head[1]
                if abs(dx) + abs(dy) == 1 and next_pos not in snake and (backtrack_dir is None or (dx, dy) != backtrack_dir):
                    return (dx, dy)

        # Path to closest free point on cycle
        min_dist = float('inf')
        best_target = None
        for p in h_cycle:
            if p not in snake:
                dist = manhattan(head, p) + random.uniform(-0.5, 0.5)
                if dist < min_dist:
                    min_dist = dist
                    best_target = p

        if best_target:
            path = a_star(snake, best_target, grid_width, grid_height)
            if path:
                return path[0]

    if safe_moves_fallback:
        safe_moves_fallback.sort()
        return safe_moves_fallback[0][1]

    # Survival: path to tail
    if len(snake) > 1:
        tail = snake[-1]
        path_to_tail = a_star(snake, tail, grid_width, grid_height, ignore_tail=True)
        if path_to_tail:
            d = path_to_tail[0]
            nx, ny = head[0] + d[0], head[1] + d[1]
            sim_snake = snake.copy()
            sim_snake.insert(0, (nx, ny))
            sim_snake.pop()
            area = flood_fill(sim_snake, (nx, ny), grid_width, grid_height)
            if area > len(snake) * 2:  # Ensure reasonable area to avoid loop
                return d

    # Any non-colliding move
    for d in [UP, DOWN, LEFT, RIGHT]:
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
    h_cycle = generate_hamiltonian_cycle(grid_width, grid_height)

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
            next_dir = find_path(snake, food, grid_width, grid_height, h_cycle)
            if next_dir is None:
                # Draw current state
                SCREEN.fill(BLACK)
                draw_text(f"Tum Zamanlar: {all_time_high}", 30, GRAY, WIDTH - 150, 20, alpha=128)
                draw_text(f"Gunluk: {daily_high}", 30, GRAY, WIDTH - 150, 50, alpha=128)
                points = [(seg[0] * BLOCK_SIZE + BLOCK_SIZE // 2, seg[1] * BLOCK_SIZE + BLOCK_SIZE // 2) for seg in snake]
                for i in range(len(points) - 1):
                    if i >= len(snake) - 10:
                        fade_factor = (len(snake) - i - 1) / 10.0
                        green_value = int(150 + 105 * fade_factor)
                        color = (0, green_value, 0)
                    else:
                        pulse = 0.7 + 0.3 * math.sin(time.time() * 2 + i * 0.2)
                        green_value = int(255 * pulse)
                        color = (0, green_value, 0)
                    pygame.draw.line(SCREEN, color, points[i], points[i+1], SNAKE_WIDTH)
                for i, point in enumerate(points):
                    if i == 0:
                        color = GREEN
                    elif i >= len(snake) - 10:
                        fade_factor = (len(snake) - i) / 10.0
                        green_value = int(150 + 105 * fade_factor)
                        color = (0, green_value, 0)
                    else:
                        pulse = 0.7 + 0.3 * math.sin(time.time() * 2 + i * 0.2)
                        green_value = int(255 * pulse)
                        color = (0, green_value, 0)
                    pygame.draw.circle(SCREEN, color, point, SNAKE_WIDTH // 2)
                pygame.draw.circle(SCREEN, RED, (food[0]*BLOCK_SIZE + BLOCK_SIZE//2, food[1]*BLOCK_SIZE + BLOCK_SIZE//2), BLOCK_SIZE//2)
                draw_text(f"Skor: {score}", 30, WHITE, WIDTH//2, 20)
                
                # Show pulsing "?" with glow effect
                head_point = points[0]
                pulse_alpha = 128 + 127 * math.sin(time.time() * 5)
                # Glow effect: draw slightly larger, semi-transparent yellow question mark behind
                draw_text("?", 60, YELLOW, head_point[0], head_point[1] - BLOCK_SIZE, alpha=int(pulse_alpha * 0.5))
                draw_text("?", 50, WHITE, head_point[0], head_point[1] - BLOCK_SIZE, alpha=int(pulse_alpha))
                pygame.display.flip()
                
                # Pause for 1 sec then end
                time.sleep(1)
                running = False
                continue

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
            
            # Soluk yuksek skorlari ciz
            draw_text(f"Tum Zamanlar: {all_time_high}", 30, GRAY, WIDTH - 150, 20, alpha=128)
            draw_text(f"Gunluk: {daily_high}", 30, GRAY, WIDTH - 150, 50, alpha=128)
            
            # Yilan noktalarini hesapla
            points = [(seg[0] * BLOCK_SIZE + BLOCK_SIZE // 2, seg[1] * BLOCK_SIZE + BLOCK_SIZE // 2) for seg in snake]
            
            # Govde cizgileri (pulsing effect)
            for i in range(len(points) - 1):
                if i >= len(snake) - 10:
                    fade_factor = (len(snake) - i - 1) / 10.0
                    green_value = int(150 + 105 * fade_factor)
                    color = (0, green_value, 0)
                else:
                    pulse = 0.7 + 0.3 * math.sin(time.time() * 2 + i * 0.2)  # Smoother pulsing
                    green_value = int(255 * pulse)
                    color = (0, green_value, 0)
                pygame.draw.line(SCREEN, color, points[i], points[i+1], SNAKE_WIDTH)
            
            # Yuvarlak eklemler
            for i, point in enumerate(points):
                if i == 0:
                    color = GREEN
                elif i >= len(snake) - 10:
                    fade_factor = (len(snake) - i) / 10.0
                    green_value = int(150 + 105 * fade_factor)
                    color = (0, green_value, 0)
                else:
                    pulse = 0.7 + 0.3 * math.sin(time.time() * 2 + i * 0.2)
                    green_value = int(255 * pulse)
                    color = (0, green_value, 0)
                pygame.draw.circle(SCREEN, color, point, SNAKE_WIDTH // 2)
            
            # Yemegi ciz (pulsing effect)
            pulse = 0.8 + 0.2 * math.sin(time.time() * 3)
            red_value = int(255 * pulse)
            pygame.draw.circle(SCREEN, (red_value, 0, 0), (food[0]*BLOCK_SIZE + BLOCK_SIZE//2, food[1]*BLOCK_SIZE + BLOCK_SIZE//2), BLOCK_SIZE//2)
            
            # Skoru ciz
            draw_text(f"Skor: {score}", 30, WHITE, WIDTH//2, 20)
            
            pygame.display.flip()

            clock.tick(FPS)

        # Oyun bitti, skorlari guncelle
        logging.info(score)  # Log session score
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
