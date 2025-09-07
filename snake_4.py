
"""
Snake Screensaver — Smart v3
----------------------------
- **Köşe osilasyonu KESİN çözüm**: Corner-escape modu (min_wall_dist<=1 ise duvardan uzaklaştıran hamle zorlanır).
- **Tersine dönme (reverse) yasak** mecbur kalınmadıkça.
- **Osilasyon cooldown**: ABAB saptanınca aynı eksene 0.7s boyunca ağır ceza.
- **Skorlu aday seçimi**: food yakınlığı + boş alan + kuyruk erişimi + duvardan uzaklık + yön ataleti.
- **Food timeout dinamik**: tahta boyu ve hızla orantılı; default 30s yerine 60–150s aralığı.
"""

import sys
import time
import math
import random
from datetime import date
from collections import deque
import heapq
import pygame

# ----------------------- Genel Ayarlar -----------------------
DEBUG = False

# Ekran & çizim
RENDER_FPS = 60
PRETTY_SNAKE = True

# Hareket zamanlaması (oyun mantığı)
MOVE_BASE_INTERVAL = 0.05
MOVE_SPEEDUP_PER_SCORE = 0.0008
MOVE_MIN_INTERVAL = 0.020

# Pathfinding / Güvenlik
MAX_ASTAR_STEPS = 600
SAFETY_BASE = 0.42
SAFETY_MAX = 0.88

# Osilasyon parametreleri
OSC_COOLDOWN_SEC = 0.7

# Renkler
BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
RED    = (255, 0, 0)
GRAY   = (100, 100, 100)
YELLOW = (255, 255, 0)

# Yönler
UP    = (0, -1)
DOWN  = (0, 1)
LEFT  = (-1, 0)
RIGHT = (1, 0)
DIRS  = (UP, RIGHT, DOWN, LEFT)

# Skor dosyası
HIGHSCORE_FILE = "snake_highscores.txt"

# ----------------------- Yardımcılar -----------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def safety_threshold(length_minus_one):
    return min(SAFETY_MAX, SAFETY_BASE + min(length_minus_one, 150) / 150.0 * (SAFETY_MAX - SAFETY_BASE))

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
    except Exception:
        return 0, 0.0, 0, 0.0

def save_highscores(all_time, all_time_rate, daily_score, daily_rate):
    today = str(date.today())
    with open(HIGHSCORE_FILE, "w") as f:
        f.write(f"{all_time}\n{all_time_rate}\n{today}\n{daily_score}\n{daily_rate}\n")

# Font önbelleği
pygame.font.init()
_FONT_CACHE = {}
def get_font(size):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.Font(None, size)
    return _FONT_CACHE[size]

def draw_text(screen, text, size, color, x, y, alpha=255, center=True):
    font = get_font(size)
    surface = font.render(text, True, color)
    if alpha < 255:
        surface.set_alpha(alpha)
    rect = surface.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    screen.blit(surface, rect)

# ----------------------- Pathfinding -----------------------
def reconstruct_path(came_from, start, goal):
    cur = goal
    out = deque()
    while cur != start:
        prev, d = came_from[cur]
        out.appendleft(d)
        cur = prev
    return list(out)

def a_star(head, target, blocked_set, w, h, ignore_tail_cell=None, max_steps=MAX_ASTAR_STEPS):
    if head == target:
        return []
    blocked = set(blocked_set)
    blocked.discard(head)
    if ignore_tail_cell is not None:
        blocked.discard(ignore_tail_cell)

    openq = []
    heapq.heappush(openq, (manhattan(head, target), 0, head))
    came_from = {}
    g = {head: 0}
    visited = set()
    steps = 0

    while openq and steps < max_steps:
        _, gcur, pos = heapq.heappop(openq)
        if pos in visited:
            continue
        visited.add(pos)
        if pos == target:
            return reconstruct_path(came_from, head, target)
        x, y = pos
        for d in DIRS:
            nx, ny = x + d[0], y + d[1]
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            np = (nx, ny)
            if np in blocked and np != target:
                continue
            ng = gcur + 1
            if ng < g.get(np, 1e12):
                g[np] = ng
                came_from[np] = (pos, d)
                # Kenar cezası ve köşe cezası
                edge_penalty = 0.25 if (nx == 0 or ny == 0 or nx == w - 1 or ny == h - 1) else 0.0
                corner_pen  = 0.25 if ((nx in (0, w-1)) and (ny in (0, h-1))) else 0.0
                f = ng + manhattan(np, target) + edge_penalty + corner_pen
                heapq.heappush(openq, (f, ng, np))
        steps += 1
    return None

def is_reachable(start, goal, blocked_set, w, h, ignore_cell=None, max_expansions=2500):
    if start == goal:
        return True
    blocked = set(blocked_set)
    if ignore_cell is not None:
        blocked.discard(ignore_cell)
    q = deque([start])
    seen = {start}
    expansions = 0
    while q and expansions < max_expansions:
        x, y = q.popleft()
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            np = (nx, ny)
            if np in blocked:
                continue
            if np == goal:
                return True
            if np not in seen:
                seen.add(np)
                q.append(np)
        expansions += 1
    return False

def flood_fill_limited(start, blocked_set, w, h, limit):
    if start in blocked_set or not (0 <= start[0] < w and 0 <= start[1] < h):
        return 0
    q = deque([start])
    seen = {start}
    cnt = 0
    while q:
        x, y = q.popleft()
        cnt += 1
        if cnt >= limit:
            return cnt
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                np = (nx, ny)
                if np not in seen and np not in blocked_set:
                    seen.add(np)
                    q.append(np)
    return cnt

def min_wall_dist(p, w, h):
    x, y = p
    return min(x, y, w - 1 - x, h - 1 - y)

# ----------------------- Food yerleştirme -----------------------
def place_food(snake, snake_set, w, h):
    head = snake[0]
    tail = snake[-1]

    # Dinamik hedef: head'in 10..60 adım ilerisindeki bölgeye odaklan (izlemeyi zevkli kılar)
    window = max(10, (w*h)//120)  # tahta büyüdükçe artır
    candidates = []
    tries = 0
    while len(candidates) < 50 and tries < 400:
        tries += 1
        fx = random.randint(1, w - 2)
        fy = random.randint(1, h - 2)
        p = (fx, fy)
        if p in snake_set:
            continue
        # uzaklık ve duvardan uzaklık puanı
        dist = manhattan(head, p)
        wall = min_wall_dist(p, w, h)
        candidates.append((-(wall) + random.random()*0.1, dist, p))
    # duvardan uzak olanları öne al, sonra mesafe
    candidates.sort()

    blocked = snake_set
    for _, _, p in candidates:
        if a_star(head, p, blocked, w, h, ignore_tail_cell=tail) is not None:
            return p

    # Fallback
    while True:
        p = (random.randint(0, w - 1), random.randint(0, h - 1))
        if p not in snake_set:
            return p

# ----------------------- Akıllı hamle seçici -----------------------
class PlannerState:
    def __init__(self):
        self.path = deque()
        self.kind = None
        self.for_food = None
        self.invalidations = 0
        self.last_heads = deque(maxlen=8)
        self.last_dir = None
        self.osc_axis = None   # 'h' or 'v'
        self.osc_until = 0.0   # time threshold

def detect_oscillation(state, now):
    if len(state.last_heads) >= 4:
        a1 = state.last_heads[-1]
        a2 = state.last_heads[-2]
        a3 = state.last_heads[-3]
        a4 = state.last_heads[-4]
        if a1 == a3 and a2 == a4:
            # eksen bul
            axis = 'v' if a1[0] == a2[0] else 'h'
            state.osc_axis = axis
            state.osc_until = now + OSC_COOLDOWN_SEC
            return True
    return False

def pick_move(snake, snake_set, food, w, h, state, now):
    head = snake[0]
    tail = snake[-1]
    length = len(snake)
    score = length - 1
    total_cells = w * h

    # Plan uygunsa uygula
    if state.path:
        d = state.path[0]
        nx, ny = head[0] + d[0], head[1] + d[1]
        new_head = (nx, ny)
        eating = (new_head == food)
        collide = (new_head in snake_set) and not (not eating and new_head == tail)
        if (0 <= nx < w and 0 <= ny < h) and not collide:
            state.path.popleft()
            state.last_dir = d
            return d, (state.kind == 'food')
        else:
            state.path.clear(); state.kind = None; state.invalidations += 1

    # Yiyeceğe A* planı
    blocked = set(snake_set); blocked.discard(head)
    path_to_food = a_star(head, food, blocked, w, h, ignore_tail_cell=tail, max_steps=MAX_ASTAR_STEPS)
    if path_to_food:
        first = path_to_food[0]
        nx, ny = head[0] + first[0], head[1] + first[1]
        new_head = (nx, ny)
        eating = (new_head == food)
        blocked_future = set(snake_set); blocked_future.add(new_head)
        if not eating: blocked_future.discard(tail)
        total_empty_future = total_cells - (len(snake) + (1 if eating else 0))
        min_safe = max(1, int(math.ceil(total_empty_future * safety_threshold(score))))
        area = flood_fill_limited(new_head, blocked_future, w, h, limit=min_safe)
        tail_reach = is_reachable(new_head, tail, blocked_future, w, h, ignore_cell=None, max_expansions=2600)
        if area >= min_safe and tail_reach:
            state.path = deque(path_to_food); state.kind = 'food'; state.for_food = food
            state.path.popleft(); state.last_dir = first
            return first, True

    # Yerel adaylar — reverse yasak (mecbur kalmadıkça)
    prev_dir = None
    if len(snake) > 1:
        prev_dir = (head[0] - snake[1][0], head[1] - snake[1][1])
    reverse_dir = (-prev_dir[0], -prev_dir[1]) if prev_dir else None

    candidates = []
    for d in DIRS:
        nx, ny = head[0] + d[0], head[1] + d[1]
        if not (0 <= nx < w and 0 <= ny < h):
            continue
        new_head = (nx, ny)
        eating = (new_head == food)
        collide = (new_head in snake_set) and not (not eating and new_head == tail)
        if collide:
            continue

        blocked_future = set(snake_set)
        blocked_future.add(new_head)
        if not eating:
            blocked_future.discard(tail)

        total_empty_future = w * h - (len(snake) + (1 if eating else 0))
        min_safe = max(1, int(math.ceil(total_empty_future * safety_threshold(score))))
        area = flood_fill_limited(new_head, blocked_future, w, h, limit=min_safe)
        tail_reach = is_reachable(new_head, tail, blocked_future, w, h, ignore_cell=None, max_expansions=1600)

        if tail_reach and area >= min_safe:
            food_dist = manhattan(new_head, food)
            wall = min_wall_dist(new_head, w, h)
            axis = 'v' if d in (UP, DOWN) else 'h'
            same_dir_bonus = 0.0
            if state.last_dir is not None and d == state.last_dir:
                same_dir_bonus = -0.3  # küçük bonus (daha az ceza)
            osc_pen = 0.0
            if now < state.osc_until and state.osc_axis == axis:
                osc_pen = 0.6  # aynı eksene ağır ceza
            rev_pen = 0.0
            if reverse_dir and d == reverse_dir:
                rev_pen = 1.0  # ters yöne büyük ceza
            # skor anahtarı: (food yakınlığı, -alan, duvardan uzaklığı ters, osilasyon/ters ceza, rasgele küçük jitter)
            key = (food_dist, -area, -wall, osc_pen + rev_pen + same_dir_bonus, random.random()*0.02, d)
            candidates.append(key)

    # Corner-escape: köşe/duvara yapışıkken duvardan uzaklaştıranları seç
    corner_mode = (min_wall_dist(head, w, h) <= 1)
    if candidates:
        if corner_mode:
            better = [c for c in candidates if min_wall_dist((head[0]+c[-1][0], head[1]+c[-1][1]), w, h) > min_wall_dist(head, w, h)]
            if better:
                candidates = better
        # reverse filtre (son çare değilse)
        if reverse_dir:
            non_rev = [c for c in candidates if c[-1] != reverse_dir]
            if non_rev:
                candidates = non_rev
        candidates.sort()
        best = candidates[0][-1]
        state.last_dir = best
        return best, False

    # Kuyruğa plan (dolaşma)
    path_to_tail = a_star(head, tail, blocked, w, h, ignore_tail_cell=tail, max_steps=MAX_ASTAR_STEPS)
    if path_to_tail:
        first = path_to_tail[0]
        state.path = deque(path_to_tail)
        state.kind = 'tail'
        state.path.popleft()
        state.last_dir = first
        return first, False

    # Son çare — ilk geçerli
    for d in DIRS:
        nx, ny = head[0] + d[0], head[1] + d[1]
        new_head = (nx, ny)
        eating = (new_head == food)
        if (0 <= nx < w and 0 <= ny < h) and (new_head not in snake_set or (not eating and new_head == tail)):
            state.last_dir = d
            return d, False

    return None, False

# ----------------------- Oyun -----------------------
def main():
    pygame.init()
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Yılan Ekran Koruyucu — Akıllı v3")

    BLOCK_SIZE = max(8, min(WIDTH, HEIGHT) // 40)
    SNAKE_WIDTH = int(BLOCK_SIZE * 0.8)

    all_time_high, all_time_rate, daily_high, daily_rate = load_highscores()

    def show_records():
        SCREEN.fill(BLACK)
        draw_text(SCREEN, "Yüksek Skorlar", 50, WHITE, WIDTH // 2, HEIGHT // 4)
        draw_text(SCREEN, f"Tüm Zamanlar: {all_time_high} ({all_time_rate:.2f}/s)", 38, GREEN, WIDTH // 2, HEIGHT // 2)
        draw_text(SCREEN, f"Günlük: {daily_high} ({daily_rate:.2f}/s)", 38, GREEN, WIDTH // 2, HEIGHT // 2 + 48)
        pygame.display.flip()
        time.sleep(0.9)

    show_records()

    grid_w = WIDTH // BLOCK_SIZE
    grid_h = HEIGHT // BLOCK_SIZE

    clock = pygame.time.Clock()
    MOVE_EVENT = pygame.USEREVENT + 1

    while True:
        snake = [(grid_w // 2, grid_h // 2)]
        snake_set = set(snake)
        direction = RIGHT
        food = place_food(snake, snake_set, grid_w, grid_h)
        score = 0
        start_time = time.time()

        # Dinamik food timeout (izlenebilirlik için uzun)
        tps_est = 1.0 / max(MOVE_MIN_INTERVAL, MOVE_BASE_INTERVAL)
        approx_cells = grid_w * grid_h
        FOOD_TIMEOUT = int(min(150, max(60, approx_cells / (tps_est * 1.4))))

        food_timer_start = time.time()

        state = PlannerState()
        state.last_heads.append(snake[0])

        move_interval = MOVE_BASE_INTERVAL
        pygame.time.set_timer(MOVE_EVENT, int(move_interval * 1000))

        running = True
        reason = "Bilinmiyor"
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.type == MOVE_EVENT:
                    # Hız güncelle
                    target_interval = max(MOVE_MIN_INTERVAL, MOVE_BASE_INTERVAL - score * MOVE_SPEEDUP_PER_SCORE)
                    if abs(target_interval - move_interval) > 0.001:
                        move_interval = target_interval
                        pygame.time.set_timer(MOVE_EVENT, int(move_interval * 1000))

                    # Açlık
                    now = time.time()
                    if now - food_timer_start > FOOD_TIMEOUT:
                        running = False
                        reason = "Açlıktan Öldü"
                        break

                    # Osilasyon tespiti
                    detect_oscillation(state, now)

                    # Hamle
                    d, using_food_plan = pick_move(snake, snake_set, food, grid_w, grid_h, state, now)
                    if d is None:
                        running = False
                        reason = "Hamle Yok"
                        break

                    direction = d
                    head = snake[0]
                    new_head = (head[0] + direction[0], head[1] + direction[1])
                    eating = (new_head == food)
                    tail = snake[-1]

                    collide = (new_head in snake_set) and not (not eating and new_head == tail)
                    if not (0 <= new_head[0] < grid_w and 0 <= new_head[1] < grid_h) or collide:
                        running = False
                        reason = "Çarpışma"
                        break

                    # Uygula
                    snake.insert(0, new_head)
                    snake_set.add(new_head)
                    state.last_heads.append(new_head)

                    if eating:
                        score += 1
                        food = place_food(snake, snake_set, grid_w, grid_h)
                        food_timer_start = now
                    else:
                        tail_cell = snake.pop()
                        snake_set.discard(tail_cell)

            # --- Çizim ---
            SCREEN.fill(BLACK)

            draw_text(SCREEN, f"Tüm Zamanlar: {all_time_high}", 28, GRAY, WIDTH - 180, 22, alpha=160)
            draw_text(SCREEN, f"Günlük: {daily_high}", 28, GRAY, WIDTH - 180, 48, alpha=160)
            draw_text(SCREEN, f"Skor: {score}", 32, WHITE, WIDTH // 2, 22)

            now = time.time()
            since_food = now - food_timer_start
            if since_food > (FOOD_TIMEOUT - 10):
                pulse_alpha = int(128 + 127 * math.sin(now * 5))
                draw_text(SCREEN, "Açlık Uyarısı", 40, YELLOW, WIDTH // 2, 60, alpha=pulse_alpha)

            # Yılan
            points = [(seg[0] * BLOCK_SIZE + BLOCK_SIZE // 2, seg[1] * BLOCK_SIZE + BLOCK_SIZE // 2) for seg in snake]
            pulse = 0.7 + 0.3 * math.sin(now * 2)

            if PRETTY_SNAKE and len(points) > 1:
                for i in range(len(points) - 1):
                    green_value = int(150 + 105 * max(0, (len(snake) - i - 1)) / 10.0) if i >= len(snake) - 10 else int(255 * pulse)
                    color = (0, green_value, 0)
                    pygame.draw.line(SCREEN, color, points[i], points[i + 1], SNAKE_WIDTH)
            for i, p in enumerate(points):
                if i == 0 and since_food > (FOOD_TIMEOUT - 5):
                    color = (int(255 * pulse), 0, 0)
                else:
                    green_value = int(150 + 105 * max(0, (len(snake) - i)) / 10.0) if i >= len(snake) - 10 else int(255 * pulse)
                    color = GREEN if i == 0 else (0, green_value, 0)
                pygame.draw.circle(SCREEN, color, p, SNAKE_WIDTH // 2)

            # Food
            fx = food[0] * BLOCK_SIZE + BLOCK_SIZE // 2
            fy = food[1] * BLOCK_SIZE + BLOCK_SIZE // 2
            pulse_food = 0.8 + 0.2 * math.sin(now * 3)
            red_value = int(255 * pulse_food)
            pygame.draw.circle(SCREEN, (red_value, 0, 0), (fx, fy), BLOCK_SIZE // 2)

            pygame.display.flip()
            clock.tick(RENDER_FPS)

        # Oyun sonu
        end_time = time.time()
        game_time = max(1e-9, end_time - start_time)
        current_rate = score / game_time
        if score > all_time_high:
            all_time_high = score
            all_time_rate = current_rate
        if score > daily_high:
            daily_high = score
            daily_rate = current_rate
        save_highscores(all_time_high, all_time_rate, daily_high, daily_rate)

        SCREEN.fill(BLACK)
        draw_text(SCREEN, "Oyun Bitti", 52, RED, WIDTH // 2, HEIGHT // 4)
        draw_text(SCREEN, f"Skor: {score} ({current_rate:.2f}/s) - {reason}", 38, WHITE, WIDTH // 2, HEIGHT // 2)
        pygame.display.flip()
        time.sleep(1.0)

        # Kayıt ekranı
        SCREEN.fill(BLACK)
        draw_text(SCREEN, "Yüksek Skorlar", 50, WHITE, WIDTH // 2, HEIGHT // 4)
        draw_text(SCREEN, f"Tüm Zamanlar: {all_time_high} ({all_time_rate:.2f}/s)", 38, GREEN, WIDTH // 2, HEIGHT // 2)
        draw_text(SCREEN, f"Günlük: {daily_high} ({daily_rate:.2f}/s)", 38, GREEN, WIDTH // 2, HEIGHT // 2 + 48)
        pygame.display.flip()
        time.sleep(0.9)

if __name__ == "__main__":
    main()
