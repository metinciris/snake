import pygame
import sys
import random
import time
from datetime import date
from collections import deque
import heapq
import math

# Pygame baslatiliyor
pygame.init()

# Tam ekran boyutlarini al
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Yilan Ekran Koruyucu")

# Renkler
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)

# Oyun ayarlari
BLOCK_SIZE = min(WIDTH, HEIGHT) // 40
SNAKE_WIDTH = int(BLOCK_SIZE * 0.8)
FPS = 300

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
            all_time_rate = float(lines[1].strip())
            daily_date = lines[2].strip()
            daily_score = int(lines[3].strip())
            daily_rate = float(lines[4].strip())
            today = str(date.today())
            if daily_date != today:
                daily_score = 0
                daily_rate = 0.0
            return all_time, all_time_rate, daily_score, daily_rate
    except:
        return 0, 0.0, 0, 0.0

def save_highscores(all_time, all_time_rate, daily_score, daily_rate):
    today = str(date.today())
    with open(HIGHSCORE_FILE, "w") as f:
        f.write(f"{all_time}\n")
        f.write(f"{all_time_rate}\n")
        f.write(f"{today}\n")
        f.write(f"{daily_score}\n")
        f.write(f"{daily_rate}\n")

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def a_star(snake, target, grid_width, grid_height, ignore_tail=False, max_steps=300):
    head = snake[0]
    pq = []
    heapq.heappush(pq, (0 + manhattan(head, target), 0, head, []))
    visited = set(snake[:-1] if ignore_tail else snake)
    steps = 0

    while pq and steps < max_steps:
        steps += 1
        _, g, pos, path = heapq.heappop(pq)
        if pos == target:
            return path

        for d in [UP, RIGHT, DOWN, LEFT]:
            nx, ny = pos[0] + d[0], pos[1] + d[1]
            if 2 <= nx < grid_width - 2 and 2 <= ny < grid_height - 2 and (nx, ny) not in visited:  # Leave 2 cells gap from edges
                visited.add((nx, ny))
                new_path = path + [d]
                new_g = g + 1
                h = manhattan((nx, ny), target)
                heapq.heappush(pq, (new_g + h, new_g, (nx, ny), new_path))
            elif 0 <= nx < grid_width and 0 <= ny < grid_height and (nx, ny) not in visited:  # Allow edge if necessary, but with higher cost
                visited.add((nx, ny))
                new_path = path + [d]
                new_g = g + 2  # Higher cost for edge moves
                h = manhattan((nx, ny), target)
                heapq.heappush(pq, (new_g + h, new_g, (nx, ny), new_path))

    return None

def bfs(snake, target, grid_width, grid_height, ignore_tail=False, max_steps=300):
    head = snake[0]
    queue = deque([(head, [])])
    visited = set(snake[:-1] if ignore_tail else snake)
    steps = 0

    while queue and steps < max_steps:
        steps += 1
        pos, path = queue.popleft()
        if pos == target:
            return path

        for d in [UP, RIGHT, DOWN, LEFT]:
            nx, ny = pos[0] + d[0], pos[1] + d[1]
            if 2 <= nx < grid_width - 2 and 2 <= ny < grid_height - 2 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [d]))
            elif 0 <= nx < grid_width and 0 <= ny < grid_height and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [d]))  # Allow edge but prefer inner

    return None

def flood_fill(snake, start, w, h):
    obstacles = set(snake[1:])
    visited = set(obstacles)
    queue = deque()
    count = 0

    if 0 <= start[0] < w and 0 <= start[1] < h and start not in visited:
        visited.add(start)
        queue.append(start)
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

def place_food(snake, grid_width, grid_height):
    total_cells = grid_width * grid_height
    safety_threshold = 0.90
    attempts = 0
    max_attempts = 10

    while attempts < max_attempts:
        attempts += 1
        food = (random.randint(2, grid_width - 3), random.randint(2, grid_height - 3))  # Place food away from edges
        if food in snake:
            continue

        path_to_food = a_star(snake, food, grid_width, grid_height)
        if not path_to_food:
            continue

        sim_snake = snake.copy()
        sim_snake.insert(0, food)
        area = flood_fill(sim_snake, food, grid_width, grid_height)
        total_empty = total_cells - len(sim_snake)
        path_to_tail = bfs(sim_snake, sim_snake[-1], grid_width, grid_height, ignore_tail=True)

        if path_to_tail and area >= total_empty * safety_threshold:
            print(f"Placed food at {food}, Area: {area}/{total_empty}")
            return food

    best_food = None
    best_area = -1
    for _ in range(5):
        food = (random.randint(0, grid_width - 1), random.randint(0, grid_height - 1))
        if food in snake:
            continue
        sim_snake = snake.copy()
        sim_snake.insert(0, food)
        area = flood_fill(sim_snake, food, grid_width, grid_height)
        if area > best_area and a_star(snake, food, grid_width, grid_height):
            best_area = area
            best_food = food
    print(f"Fallback food at {best_food}, Area: {best_area}")
    food = best_food if best_food else None
    while food is None or food in snake:
        food = (random.randint(2, grid_width - 3), random.randint(2, grid_height - 3))
    return food

def find_path(snake, food, grid_width, grid_height):
    head = snake[0]
    directions = [UP, DOWN, LEFT, RIGHT]
    score = len(snake) - 1
    safety_threshold = min(0.85, 0.40 + (score / 150) * 0.45)  # 0.40 to 0.85
    total_cells = grid_width * grid_height

    # Early game: direct food path
    if score < 10:
        food_path = a_star(snake, food, grid_width, grid_height, max_steps=300)
        if food_path:
            print(f"Score: {score}, Early game: Moving to food with direction {food_path[0]}")
            return food_path[0], False, True
        for d in directions:
            nx, ny = head[0] + d[0], head[1] + d[1]
            if 0 <= nx < grid_width and 0 <= ny < grid_height and (nx, ny) not in snake:
                print(f"Score: {score}, Early game: No food path, choosing valid move {d}")
                return d, False, False

    # Food path if safe
    food_path = a_star(snake, food, grid_width, grid_height, max_steps=300)
    food_path_found = bool(food_path)

    backtrack_dir = None
    if len(snake) > 1:
        current_dir = (snake[0][0] - snake[1][0], snake[0][1] - snake[1][1])
        backtrack_dir = (-current_dir[0], -current_dir[1])
        directions = [d for d in directions if d != backtrack_dir]

    if food_path:
        sim_snake = snake.copy()
        new_head = (head[0] + food_path[0][0], head[1] + food_path[0][1])
        sim_snake.insert(0, new_head)
        area = flood_fill(sim_snake, new_head, grid_width, grid_height)
        total_empty = total_cells - len(sim_snake)
        if area >= total_empty * safety_threshold:
            print(f"Score: {score}, Direct move to food: {food_path[0]}, Area: {area}/{total_empty}")
            return food_path[0], False, True

    # Safe moves, prioritizing food distance
    safe_moves = []
    for d in directions:
        nx, ny = head[0] + d[0], head[1] + d[1]
        if not (0 <= nx < grid_width and 0 <= ny < grid_height and (nx, ny) not in snake):
            continue
        sim_snake = snake.copy()
        sim_snake.insert(0, (nx, ny))
        if (nx, ny) != food:
            sim_snake.pop()
        area = flood_fill(sim_snake, (nx, ny), grid_width, grid_height)
        total_empty = total_cells - len(sim_snake)
        food_dist = manhattan((nx, ny), food)
        if area >= total_empty * safety_threshold:
            safe_moves.append((food_dist, -area, random.uniform(-0.05, 0.05), d))

    if safe_moves:
        safe_moves.sort()  # Prioritize low food_dist
        print(f"Score: {score}, Chose safe move: {safe_moves[0][3]}, Food distance: {safe_moves[0][0]}")
        return safe_moves[0][3], False, food_path_found

    # Tail fallback: target 3 segments before tail end to avoid collision
    if len(snake) > 3:
        tail_target = snake[-3]  # 3 segments before tail end
        path_to_tail = a_star(snake, tail_target, grid_width, grid_height, ignore_tail=True, max_steps=300)
        if path_to_tail:
            next_pos = (head[0] + path_to_tail[0][0], head[1] + path_to_tail[0][1])
            if next_pos != snake[-1]:  # Avoid moving to current tail position
                print(f"Score: {score}, Following tail (3rd segment) with direction {path_to_tail[0]}")
                return path_to_tail[0], False, food_path_found

    # Last resort: any valid move
    for d in directions:
        nx, ny = head[0] + d[0], head[1] + d[1]
        if 0 <= nx < grid_width and 0 <= ny < grid_height and (nx, ny) not in snake:
            print(f"Score: {score}, Last-resort move: {d}")
            return d, False, food_path_found

    print(f"Score: {score}, No valid moves, entering thinking mode")
    return None, True, food_path_found

def draw_text(text, size, color, x, y, alpha=255):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    if alpha < 255:
        text_surface.set_alpha(alpha)
    text_rect = text_surface.get_rect(center=(x, y))
    SCREEN.blit(text_surface, text_rect)

def show_records(all_time, all_time_rate, daily, daily_rate):
    SCREEN.fill(BLACK)
    draw_text("Yuksek Skorlar", 50, WHITE, WIDTH//2, HEIGHT//4)
    draw_text(f"Tum Zamanlar: {all_time} ({all_time_rate:.2f}/s)", 40, GREEN, WIDTH//2, HEIGHT//2)
    draw_text(f"Gunluk: {daily} ({daily_rate:.2f}/s)", 40, GREEN, WIDTH//2, HEIGHT//2 + 50)
    pygame.display.flip()
    time.sleep(2)

def main():
    all_time_high, all_time_rate, daily_high, daily_rate = load_highscores()
    show_records(all_time_high, all_time_rate, daily_high, daily_rate)

    grid_width = WIDTH // BLOCK_SIZE
    grid_height = HEIGHT // BLOCK_SIZE

    while True:
        snake = [(grid_width // 2, grid_height // 2)]
        direction = RIGHT
        food = place_food(snake, grid_width, grid_height)
        score = 0
        clock = pygame.time.Clock()
        thinking = False
        think_start = 0
        food_timer_start = time.time()  # Start food timer
        start_time = time.time()
        move_timer = 0
        base_move_interval = 0.015
        move_interval = base_move_interval
        speed_variation = 0
        just_ate = False

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

            current_time = time.time()
            speed_variation = random.uniform(-0.005, 0.005)
            move_interval = max(0.005, base_move_interval + speed_variation - (score / 2000))

            # Check 30-second food timer
            if current_time - food_timer_start > 30:
                print(f"Game over: Failed to eat food within 30 seconds")
                running = False
                break

            # Force exit thinking after 0.1s
            if current_time - think_start > 0.1 and thinking:
                thinking = False
                just_ate = False
                print(f"Score: {score}, Forced exit from thinking after 0.1s")

            if current_time > move_timer:
                if just_ate and not thinking:
                    thinking = True
                    think_start = current_time
                    move_interval = 0.1
                    print(f"Score: {score}, Just ate, starting to think at {current_time}")

                next_dir, is_thinking, food_path_found = find_path(snake, food, grid_width, grid_height)

                if is_thinking or (thinking and current_time - think_start <= 0.1):
                    thinking = True
                    move_interval = 0.1
                else:
                    thinking = False
                    just_ate = False
                    move_interval = 0.005 if food_path_found else base_move_interval
                    print(f"Score: {score}, Exited thinking mode, move_interval: {move_interval}")

                    direction = next_dir if next_dir else direction
                    head = snake[0]
                    new_head = (head[0] + direction[0], head[1] + direction[1])

                    if (new_head[0] < 0 or new_head[0] >= grid_width or
                        new_head[1] < 0 or new_head[1] >= grid_height or
                        new_head in snake):
                        print(f"Game over: Head at {new_head}, Snake: {snake}")
                        running = False
                        break

                    snake.insert(0, new_head)
                    print(f"Score: {score}, Moved to {new_head}, Direction: {direction}")

                    if new_head == food:
                        score += 1
                        just_ate = True
                        food = place_food(snake, grid_width, grid_height)
                        food_timer_start = current_time  # Reset food timer
                        print(f"Score: {score}, Ate food, new food at {food}")
                    else:
                        snake.pop()
                        just_ate = False

                    move_timer = current_time + move_interval

            SCREEN.fill(BLACK)
            draw_text(f"Tum Zamanlar: {all_time_high}", 30, GRAY, WIDTH - 150, 20, alpha=128)
            draw_text(f"Gunluk: {daily_high}", 30, GRAY, WIDTH - 150, 50, alpha=128)
            
            points = [(seg[0] * BLOCK_SIZE + BLOCK_SIZE // 2, seg[1] * BLOCK_SIZE + BLOCK_SIZE // 2) for seg in snake]
            pulse = 0.7 + 0.3 * math.sin(current_time * 2)
            
            for i in range(len(points) - 1):
                green_value = int(150 + 105 * (len(snake) - i - 1) / 10.0) if i >= len(snake) - 10 else int(255 * pulse)
                color = (0, green_value, 0)
                pygame.draw.line(SCREEN, color, points[i], points[i+1], SNAKE_WIDTH)
            
            # Head color: red in last 5 seconds, else green
            for i, point in enumerate(points):
                green_value = int(150 + 105 * (len(snake) - i) / 10.0) if i >= len(snake) - 10 else int(255 * pulse)
                if i == 0 and current_time - food_timer_start > 25:
                    color = (int(255 * pulse), 0, 0)  # Red with pulse
                else:
                    color = GREEN if i == 0 else (0, green_value, 0)
                pygame.draw.circle(SCREEN, color, point, SNAKE_WIDTH // 2)
            
            pulse_food = 0.8 + 0.2 * math.sin(current_time * 3)
            red_value = int(255 * pulse_food)
            pygame.draw.circle(SCREEN, (red_value, 0, 0), (food[0]*BLOCK_SIZE + BLOCK_SIZE//2, food[1]*BLOCK_SIZE + BLOCK_SIZE//2), BLOCK_SIZE//2)
            
            draw_text(f"Skor: {score}", 30, WHITE, WIDTH//2, 20)
            
            # Hunger warning in last 10 seconds, below score, flashing
            if current_time - food_timer_start > 20:
                pulse_alpha = 128 + 127 * math.sin(current_time * 5)
                draw_text("Açlık Uyarısı", 40, YELLOW, WIDTH//2, 60, alpha=int(pulse_alpha))
            
            if thinking:
                head_point = points[0]
                pulse_alpha = 128 + 127 * math.sin(current_time * 5)
                draw_text("?", 60, YELLOW, head_point[0], head_point[1] - BLOCK_SIZE, alpha=int(pulse_alpha * 0.5))
                draw_text("?", 50, WHITE, head_point[0], head_point[1] - BLOCK_SIZE, alpha=int(pulse_alpha))
            
            pygame.display.flip()
            clock.tick(FPS)

        end_time = time.time()
        game_time = end_time - start_time
        current_rate = score / game_time if game_time > 0 else 0
        if score > all_time_high:
            all_time_high = score
            all_time_rate = current_rate
        if score > daily_high:
            daily_high = score
            daily_rate = current_rate
        save_highscores(all_time_high, all_time_rate, daily_high, daily_rate)

        SCREEN.fill(BLACK)
        draw_text("Oyun Bitti", 50, RED, WIDTH//2, HEIGHT//4)
        reason = "Açlıktan Öldü" if current_time - food_timer_start > 30 else "Collision"
        draw_text(f"Skor: {score} ({current_rate:.2f}/s) - {reason}", 40, WHITE, WIDTH//2, HEIGHT//2)
        pygame.display.flip()
        time.sleep(2)
        show_records(all_time_high, all_time_rate, daily_high, daily_rate)

if __name__ == "__main__":
    main()
