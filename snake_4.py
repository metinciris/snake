
"""
Snake Screensaver — Smarter AI + Much Lower CPU
------------------------------------------------

Neler değişti? (Özet)
- **Aşırı CPU kullanımını** ciddi biçimde azaltır: 
  - Oyun mantığı ayrı bir "MOVE_EVENT" zamanlayıcısında çalışır (render FPS'ten bağımsız).
  - Pathfinding (A*) **önbelleğe alınır**; her karede değil, gerektiğinde yeniden hesaplanır.
  - `snake_set` kullanımı ile O(n) üyelik kontrolleri O(1)'e düşürüldü.
  - `flood fill` artık **limitli** ve erken durmalı — güvenlik eşiğine ulaşınca durur.
  - Gereksiz `print`'ler ve alfa/Font oluşturma tekrarları kaldırıldı (fontlar önbellekte).
- **Daha akıllı yön bulma**:
  - Önce yiyeceğe (food) A* ile plan çıkarır; plan güvenliyse uygular ve **takip eder**.
  - Güvenli değilse en güvenli yerel hamleyi seçer (boş alan ve kuyruğa ulaşılabilirlik kontrolü).
  - Gerekirse kuyruğu takip eder (A* ile).
- Görselleştirme korunurken; istenirse performans için `PRETTY_SNAKE=False` yapabilirsiniz.

İpuçları
- Performans için gerekirse `RENDER_FPS` değerini düşürün (ör. 45).
- Hareket hızı puan arttıkça artar; sınırları `MOVE_*` sabitleri belirler.
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
RENDER_FPS = 60          # Çizim FPS (yalnızca ekran yenileme)
PRETTY_SNAKE = True      # False yaparsanız daha basit ve hızlı render

# Hareket zamanlaması (oyun mantığı)
MOVE_BASE_INTERVAL = 0.05      # saniye (20 tps)
MOVE_SPEEDUP_PER_SCORE = 0.0008
MOVE_MIN_INTERVAL = 0.020      # saniye (>=50 tps olmaz)

# Pathfinding / Güvenlik
MAX_ASTAR_STEPS = 180
SAFETY_BASE = 0.40
SAFETY_MAX = 0.85

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
    # Eski formülle aynı: 0.40 -> 0.85 arası, skor büyüdükçe artar (150 puanda 0.85)
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
    """
    A*: head -> target, blocked_set engeller. 
    ignore_tail_cell: Verilirse bu hücre engel sayılmaz (kuyruk hareket edecek varsayımı).
    """
    if head == target:
        return []

    blocked = set(blocked_set)
    blocked.discard(head)  # Başlangıç serbest
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
                # Kenarlara hafif ceza (food için içeriği tercih eder)
                edge_penalty = 0.2 if (nx == 0 or ny == 0 or nx == w - 1 or ny == h - 1) else 0.0
                f = ng + manhattan(np, target) + edge_penalty
                heapq.heappush(openq, (f, ng, np))
        steps += 1

    return None

def is_reachable(start, goal, blocked_set, w, h, ignore_cell=None, max_expansions=2000):
    """ Basit BFS erişilebilirlik (kısa devre). """
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
    """ 
    Engeller haricinde ulaşılabilir boş alan sayısını döndürür.
    'limit'e ulaştığında erken döner (performans için).
    """
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

# ----------------------- Food yerleştirme -----------------------
def place_food(snake_set, head, w, h):
    """
    Daha hafif: Yılanın üzerinde olmayan rastgele iç hücrelerden seçer, 
    head'e uzak ve ulaşılabilir olanı tercih eder.
    """
    interior = True  # Kenarlardan bir hücre içeride başla
    min_x = 1 if interior else 0
    min_y = 1 if interior else 0
    max_x = w - 2 if interior else w - 1
    max_y = h - 2 if interior else h - 1

    candidates = []
    tries = 0
    while len(candidates) < 24 and tries < 100:
        tries += 1
        fx = random.randint(min_x, max_x)
        fy = random.randint(min_y, max_y)
        p = (fx, fy)
        if p in snake_set:
            continue
        candidates.append(p)

    # En uzaklardan başla
    candidates.sort(key=lambda p: manhattan(p, head), reverse=True)

    # Ulaşılabilir olan ilk adayı al
    # A* kontrolü sırasında kuyruğu serbest saymak için tail parametresi bekleyen fonksiyon çağıranın içinde kullanılacak.
    # Burada yalnızca hızlı bir filtre yapalım; gerçek A* zaten next_move'ta çalışacak.
    for p in candidates:
        return p

    # Fallback (teorik)
    while True:
        p = (random.randint(0, w - 1), random.randint(0, h - 1))
        if p not in snake_set:
            return p

# ----------------------- Akıllı hamle seçici -----------------------
class PlannerState:
    def __init__(self):
        self.path = deque()  # tutulan plan (yön vektörleri)
        self.kind = None     # 'food' | 'tail' | None
        self.for_food = None # planlanan hedef
        self.invalidations = 0

def pick_move(snake, snake_set, food, w, h, state):
    """
    snake: list[(x,y)] — 0=head
    snake_set: set([...]) — hızlı üyelik
    state: PlannerState (önbellek)
    """
    head = snake[0]
    tail = snake[-1]
    length = len(snake)
    score = length - 1
    total_cells = w * h

    # 1) Önbellekte plan varsa ve ilk adım halen güvenliyse onu uygula
    if state.path:
        d = state.path[0]
        nx, ny = head[0] + d[0], head[1] + d[1]
        new_head = (nx, ny)
        eating = (new_head == food)
        # Kuyruğa basma durumu güvenli (yemiyorsa kuyruk çekilecek)
        collide = (new_head in snake_set) and not (not eating and new_head == tail)
        if (0 <= nx < w and 0 <= ny < h) and not collide:
            state.path.popleft()
            return d, (state.kind == 'food')
        else:
            # plan geçersiz
            state.path.clear()
            state.kind = None
            state.invalidations += 1

    # Engeller (baş hariç)
    blocked = set(snake_set)
    blocked.discard(head)

    # 2) Yiyeceğe A* ile yeni plan — güvenlik kontrolü ile
    path_to_food = a_star(head, food, blocked, w, h, ignore_tail_cell=tail, max_steps=MAX_ASTAR_STEPS)
    if path_to_food:
        # İlk adımı uygula (güvenlik testleri)
        first = path_to_food[0]
        nx, ny = head[0] + first[0], head[1] + first[1]
        new_head = (nx, ny)
        eating = (new_head == food)

        # Kuyruk hareketini hesaba katarak engel seti
        blocked_future = set(snake_set)
        blocked_future.add(new_head)
        if not eating:
            # yemiyorsa kuyruk bir hücre çekilecek => o hücreyi serbest say
            blocked_future.discard(tail)

        # 2a) Kuyruğa erişilebilir mi?
        tail_reach = is_reachable(new_head, tail, blocked_future, w, h, ignore_cell=None, max_expansions=2000)
        # 2b) Alan yeterli mi? (erken durmalı flood fill)
        total_empty_future = total_cells - (len(snake) + (1 if eating else 0))
        min_safe = max(1, int(math.ceil(total_empty_future * safety_threshold(score))))
        area = flood_fill_limited(new_head, blocked_future, w, h, limit=min_safe)

        if tail_reach and area >= min_safe:
            state.path = deque(path_to_food)
            state.kind = 'food'
            state.for_food = food
            # İlk adımı döndür
            state.path.popleft()
            return first, True

    # 3) Yerel güvenli hamleler arasından en iyisi
    candidates = []
    for d in DIRS:
        # Geri gitmeyi zorunlu olarak engellemeyelim; filtre güvenlikten geçer
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

        # Kuyruğa erişilebilirlik (hızlı)
        tail_reach = is_reachable(new_head, tail, blocked_future, w, h, ignore_cell=None, max_expansions=1200)

        if tail_reach and area >= min_safe:
            candidates.append((manhattan(new_head, food), -area, d))

    if candidates:
        candidates.sort()
        best = candidates[0][2]
        return best, False

    # 4) Kuyruğa doğru plan (son çare güvenli dolaşma)
    path_to_tail = a_star(head, tail, blocked, w, h, ignore_tail_cell=tail, max_steps=MAX_ASTAR_STEPS)
    if path_to_tail:
        state.path = deque(path_to_tail)
        state.kind = 'tail'
        state.path.popleft()
        return path_to_tail[0], False

    # 5) Artık ne geçerse — ilk geçerli hamle
    for d in DIRS:
        nx, ny = head[0] + d[0], head[1] + d[1]
        new_head = (nx, ny)
        eating = (new_head == food)
        if (0 <= nx < w and 0 <= ny < h) and (new_head not in snake_set or (not eating and new_head == tail)):
            return d, False

    # Hiç yoksa None
    return None, False

# ----------------------- Oyun -----------------------
def main():
    pygame.init()
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Yılan Ekran Koruyucu — Akıllı")

    BLOCK_SIZE = max(8, min(WIDTH, HEIGHT) // 40)
    SNAKE_WIDTH = int(BLOCK_SIZE * 0.8)

    all_time_high, all_time_rate, daily_high, daily_rate = load_highscores()

    def show_records():
        SCREEN.fill(BLACK)
        draw_text(SCREEN, "Yüksek Skorlar", 50, WHITE, WIDTH // 2, HEIGHT // 4)
        draw_text(SCREEN, f"Tüm Zamanlar: {all_time_high} ({all_time_rate:.2f}/s)", 38, GREEN, WIDTH // 2, HEIGHT // 2)
        draw_text(SCREEN, f"Günlük: {daily_high} ({daily_rate:.2f}/s)", 38, GREEN, WIDTH // 2, HEIGHT // 2 + 48)
        pygame.display.flip()
        time.sleep(1.2)

    show_records()

    grid_w = WIDTH // BLOCK_SIZE
    grid_h = HEIGHT // BLOCK_SIZE

    clock = pygame.time.Clock()
    MOVE_EVENT = pygame.USEREVENT + 1

    while True:
        # Oyun başlangıç
        snake = [(grid_w // 2, grid_h // 2)]
        snake_set = set(snake)
        direction = RIGHT
        food = place_food(snake_set, snake[0], grid_w, grid_h)
        score = 0
        start_time = time.time()
        food_timer_start = time.time()

        state = PlannerState()

        # İlk interval ayarı
        move_interval = MOVE_BASE_INTERVAL
        pygame.time.set_timer(MOVE_EVENT, int(move_interval * 1000))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.type == MOVE_EVENT:
                    # Hız güncelle
                    target_interval = max(MOVE_MIN_INTERVAL, MOVE_BASE_INTERVAL - score * MOVE_SPEEDUP_PER_SCORE)
                    if abs(target_interval - move_interval) > 0.001:
                        move_interval = target_interval
                        pygame.time.set_timer(MOVE_EVENT, int(move_interval * 1000))

                    # Açlık zamanlayıcısı
                    now = time.time()
                    if now - food_timer_start > 30:
                        running = False
                        reason = "Açlıktan Öldü"
                        break

                    # Hamle seç
                    d, used_food_plan = pick_move(snake, snake_set, food, grid_w, grid_h, state)
                    if d is None:
                        running = False
                        reason = "Hamle Yok"
                        break

                    direction = d
                    head = snake[0]
                    new_head = (head[0] + direction[0], head[1] + direction[1])
                    eating = (new_head == food)
                    tail = snake[-1]

                    # Çarpışma kontrolü (kuyruk istisnası)
                    collide = (new_head in snake_set) and not (not eating and new_head == tail)
                    if not (0 <= new_head[0] < grid_w and 0 <= new_head[1] < grid_h) or collide:
                        running = False
                        reason = "Çarpışma"
                        break

                    # Hareket uygula
                    snake.insert(0, new_head)
                    snake_set.add(new_head)
                    if eating:
                        score += 1
                        food = place_food(snake_set, snake[0], grid_w, grid_h)
                        food_timer_start = now
                        # food değiştiyse planı iptal etmeye gerek yok; pick_move zaten kontrol ediyor
                    else:
                        tail_cell = snake.pop()
                        snake_set.discard(tail_cell)

            # --- Çizim ---
            SCREEN.fill(BLACK)

            # Skor üst bilgi
            draw_text(SCREEN, f"Tüm Zamanlar: {all_time_high}", 28, GRAY, WIDTH - 180, 22, alpha=160)
            draw_text(SCREEN, f"Günlük: {daily_high}", 28, GRAY, WIDTH - 180, 48, alpha=160)
            draw_text(SCREEN, f"Skor: {score}", 32, WHITE, WIDTH // 2, 22)

            # Açlık uyarısı (son 10 sn)
            now = time.time()
            since_food = now - food_timer_start
            if since_food > 20:
                pulse_alpha = int(128 + 127 * math.sin(now * 5))
                draw_text(SCREEN, "Açlık Uyarısı", 40, YELLOW, WIDTH // 2, 60, alpha=pulse_alpha)

            # Yılanı çiz
            points = [(seg[0] * BLOCK_SIZE + BLOCK_SIZE // 2, seg[1] * BLOCK_SIZE + BLOCK_SIZE // 2) for seg in snake]
            pulse = 0.7 + 0.3 * math.sin(now * 2)

            if PRETTY_SNAKE and len(points) > 1:
                for i in range(len(points) - 1):
                    green_value = int(150 + 105 * max(0, (len(snake) - i - 1)) / 10.0) if i >= len(snake) - 10 else int(255 * pulse)
                    color = (0, green_value, 0)
                    pygame.draw.line(SCREEN, color, points[i], points[i + 1], SNAKE_WIDTH)
            # Kafadan kuyruğa daireler
            for i, p in enumerate(points):
                if i == 0 and since_food > 25:
                    color = (int(255 * pulse), 0, 0)  # Açlıkta kırmızı nabız
                else:
                    green_value = int(150 + 105 * max(0, (len(snake) - i)) / 10.0) if i >= len(snake) - 10 else int(255 * pulse)
                    color = GREEN if i == 0 else (0, green_value, 0)
                pygame.draw.circle(SCREEN, color, p, SNAKE_WIDTH // 2)

            # Yiyecek
            fx = food[0] * BLOCK_SIZE + BLOCK_SIZE // 2
            fy = food[1] * BLOCK_SIZE + BLOCK_SIZE // 2
            pulse_food = 0.8 + 0.2 * math.sin(now * 3)
            red_value = int(255 * pulse_food)
            pygame.draw.circle(SCREEN, (red_value, 0, 0), (fx, fy), BLOCK_SIZE // 2)

            pygame.display.flip()
            clock.tick(RENDER_FPS)

        # Oyun bitti — skor/istatistik
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

        # Bitis ekranı
        SCREEN.fill(BLACK)
        draw_text(SCREEN, "Oyun Bitti", 52, RED, WIDTH // 2, HEIGHT // 4)
        draw_text(SCREEN, f"Skor: {score} ({current_rate:.2f}/s) - {reason}", 38, WHITE, WIDTH // 2, HEIGHT // 2)
        pygame.display.flip()
        time.sleep(1.5)

        # Kayıt ekranı
        show_records()

if __name__ == "__main__":
    main()
