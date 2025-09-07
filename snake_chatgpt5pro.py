#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snake Screensaver — v7 (corner-fix, greedy fallback, robust A*)
- Fixes top-right "iki kare yukarı-aşağı" osilasyonu.
- A* düzeltilmiş (visited=closed set; push'ta değil pop'ta işaretleme).
- Fallback sabit [UP,RIGHT,DOWN,LEFT] yerine "yemeğe doğru" açgözlü (greedy).
- Uzunluk 1 iken bile "hemen ters yöne dönme" engeli var.
- son-kayit.txt'ye ayrıntılı log yazar.
- ESC veya herhangi bir tuş / tık -> kaydedip çıkar.
"""
import pygame, sys, random, time, math, heapq
from datetime import date, datetime
from collections import deque

# ----------------- Ayarlar -----------------
pygame.init()
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Yilan Ekran Koruyucu v7")

BLACK=(0,0,0); WHITE=(255,255,255); GREEN=(0,255,0); RED=(255,0,0); GRAY=(100,100,100); YELLOW=(255,255,0)
BLOCK_SIZE = max(8, min(WIDTH, HEIGHT)//40)
SNAKE_WIDTH = max(3, int(BLOCK_SIZE*0.8))
FPS = 300

UP=(0,-1); DOWN=(0,1); LEFT=(-1,0); RIGHT=(1,0)
HIGHSCORE_FILE="snake_highscores.txt"
LOG_FILE="son-kayit.txt"

# ----------------- Yardımcılar -----------------
def log(msg):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            ts=datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] {msg}\n")
    except Exception as e:
        # Log yazılamazsa sessiz geç
        pass

def draw_text(text, size, color, x, y, alpha=255):
    font = pygame.font.Font(None, size)
    surf = font.render(text, True, color)
    if alpha<255: surf.set_alpha(alpha)
    rect = surf.get_rect(center=(x,y))
    SCREEN.blit(surf, rect)

def load_highscores():
    try:
        with open(HIGHSCORE_FILE, "r", encoding="utf-8") as f:
            lines=f.readlines()
            all_time=int(lines[0].strip())
            all_time_rate=float(lines[1].strip())
            daily_date=lines[2].strip()
            daily_score=int(lines[3].strip())
            daily_rate=float(lines[4].strip())
            if daily_date != str(date.today()):
                daily_score=0; daily_rate=0.0
            return all_time, all_time_rate, daily_score, daily_rate
    except:
        return 0,0.0,0,0.0

def save_highscores(all_time, all_time_rate, daily_score, daily_rate):
    with open(HIGHSCORE_FILE, "w", encoding="utf-8") as f:
        f.write(f"{all_time}\n{all_time_rate}\n{date.today()}\n{daily_score}\n{daily_rate}\n")

def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

def neighbors(pos):
    x,y=pos
    return [(x,y-1),(x+1,y),(x,y+1),(x-1,y)]  # U,R,D,L

def a_star(snake, target, w, h, ignore_tail=False, max_steps=10000):
    """Standart A*: closed set pop'ta tutulur; push'ta visited olmaz.
       'blocked' = yılan gövdesi (tail ignore true ise kuyruğun son karesi boş sayılır).
       Duvara çarpma yasak. Target food/kuyruk olabilir (engel değil).
       Kenara yakın olmayı az da olsa cezalandır (penalty) — ama yasak değil.
    """
    head = snake[0]
    if head == target:
        return []

    blocked = set(snake[:-1] if ignore_tail else snake)
    # Target, blocked içinde olabilir (örn. kuyruk), hedefte durmaya izin ver
    if target in blocked:
        blocked.remove(target)

    def in_bounds(p):
        x,y=p; return 0<=x<w and 0<=y<h

    def step_cost(n):
        x,y=n
        # duvara 1 kare yakınsa +0.2 ceza (yılanın kenarda takılmasını azaltır)
        wall_near = (x==0 or y==0 or x==w-1 or y==h-1)
        return 1.2 if wall_near else 1.0

    openq=[]
    heapq.heappush(openq, (manhattan(head,target), 0.0, head, []))
    closed=set()
    steps=0
    while openq and steps<max_steps:
        steps+=1
        f,g,pos,path = heapq.heappop(openq)
        if pos in closed: 
            continue
        closed.add(pos)
        if pos==target:
            return path
        for nx,ny in neighbors(pos):
            n=(nx,ny)
            if not in_bounds(n): 
                continue
            if n in blocked or n in closed:
                continue
            ng = g + step_cost(n)
            # Basit konsisten heuristik
            hcost = manhattan(n, target)
            heapq.heappush(openq, (ng+hcost, ng, n, path + [(nx-pos[0], ny-pos[1])]))
    return None

def greedy_dirs_toward(src, dst):
    """Yemeğe doğru yön sıralaması üretir."""
    dx=dst[0]-src[0]; dy=dst[1]-src[1]
    order=[]
    if abs(dx)>=abs(dy):
        if dx>0: order.append(RIGHT)
        elif dx<0: order.append(LEFT)
        if dy>0: order.append(DOWN)
        elif dy<0: order.append(UP)
    else:
        if dy>0: order.append(DOWN)
        elif dy<0: order.append(UP)
        if dx>0: order.append(RIGHT)
        elif dx<0: order.append(LEFT)
    # kalanları ekle
    for d in (UP,DOWN,LEFT,RIGHT):
        if d not in order: order.append(d)
    return order

def flood_fill_free(snake, start, w, h):
    obstacles=set(snake[1:])  # baş hariç gövde
    if start in obstacles: return 0
    q=deque([start]); visited=set([start]); cnt=0
    while q:
        x,y=q.popleft(); cnt+=1
        for nx,ny in neighbors((x,y)):
            if 0<=nx<w and 0<=ny<h and (nx,ny) not in obstacles and (nx,ny) not in visited:
                visited.add((nx,ny)); q.append((nx,ny))
    return cnt

def place_food(snake, w, h):
    # basit ve güvenli: rastgele dene; A* ile ulaşılabilirliği ve alanı kontrol et
    tries=0; total=w*h
    while tries<200:
        tries+=1
        fx,fy = random.randint(1,w-2), random.randint(1,h-2)
        if (fx,fy) in snake: 
            continue
        path=a_star(snake, (fx,fy), w, h, ignore_tail=False)
        if not path: 
            continue
        sim=snake[:]; sim.insert(0, (sim[0][0]+path[0][0], sim[0][1]+path[0][1]))
        area=flood_fill_free(sim, sim[0], w, h)
        empty = total - len(sim)
        if area >= max(5, 0.50*empty):  # güvenlik eşiği
            log(f"FOOD ok at {(fx,fy)} area={area}/{empty} tries={tries}")
            return (fx,fy)
    # son çare
    while True:
        fx,fy = random.randint(0,w-1), random.randint(0,h-1)
        if (fx,fy) not in snake:
            log(f"FOOD fallback at {(fx,fy)} after {tries} tries")
            return (fx,fy)

def find_next_dir(snake, food, w, h, prev_dir):
    head=snake[0]
    # 1) Global plan: A*
    path = a_star(snake, food, w, h, ignore_tail=False, max_steps=10000)
    if path:
        return path[0], "ASTAR"

    # 2) Güvenli yerel hamleler: yemeğe greedy sırada
    order = greedy_dirs_toward(head, food)
    best=None
    total=w*h
    for d in order:
        nx,ny=head[0]+d[0], head[1]+d[1]
        if not (0<=nx<w and 0<=ny<h): 
            continue
        if (nx,ny) in snake:
            continue
        sim=snake[:]
        sim.insert(0,(nx,ny))
        if (nx,ny)!=food: sim.pop()  # yemeden ilerledik
        area = flood_fill_free(sim, (nx,ny), w, h)
        empty = total - len(sim)
        if area >= max(5, 0.35*empty):
            # ters yöne dönmeyi mümkünse engelle
            if prev_dir and (d[0]==-prev_dir[0] and d[1]==-prev_dir[1]):
                # başka aday var mı? varsa ertele
                best = best or None
                continue
            return d, "SAFE-GREEDY"
        # en azından bir aday sakla (son çare)
        if best is None:
            best=(d, "GREEDY")

    if best:
        # ters-dönüş guard'ı burada da deneriz
        d,tag=best
        if prev_dir and (d[0]==-prev_dir[0] and d[1]==-prev_dir[1]):
            # farklı ilk legal yön bul
            for alt in order:
                if alt==d: continue
                nx,ny=head[0]+alt[0], head[1]+alt[1]
                if 0<=nx<w and 0<=ny<h and (nx,ny) not in snake:
                    return alt, "ALT-GREEDY"
        return d, tag

    # 3) Tam son çare: herhangi bir legal
    for d in order:
        nx,ny=head[0]+d[0], head[1]+d[1]
        if 0<=nx<w and 0<=ny<h and (nx,ny) not in snake:
            return d, "ANY"
    return None, "STUCK"

def show_records(all_time, all_time_rate, daily, daily_rate):
    SCREEN.fill(BLACK)
    draw_text("Yüksek Skorlar", 56, WHITE, WIDTH//2, HEIGHT//4)
    draw_text(f"Tüm Zamanlar: {all_time}  ({all_time_rate:.2f}/s)", 40, GREEN, WIDTH//2, HEIGHT//2)
    draw_text(f"Günlük: {daily}  ({daily_rate:.2f}/s)", 40, GREEN, WIDTH//2, HEIGHT//2+50)
    pygame.display.flip()
    time.sleep(1.0)

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("== Snake Screensaver v7 LOG ==\n")
        f.write(f"datetime: {datetime.now().isoformat()}\n")
        f.write(f"grid={WIDTH//BLOCK_SIZE}x{HEIGHT//BLOCK_SIZE} block={BLOCK_SIZE}\n")
        f.write("-"*60 + "\n")

    all_time_high, all_time_rate, daily_high, daily_rate = load_highscores()
    show_records(all_time_high, all_time_rate, daily_high, daily_rate)

    w = WIDTH//BLOCK_SIZE
    h = HEIGHT//BLOCK_SIZE

    while True:
        snake=[(w//2, h//2)]
        food = place_food(snake, w, h)
        score=0
        start_t=time.time(); last_move=time.time()
        move_interval=0.015
        prev_dir=None
        clock=pygame.time.Clock()

        log(f"GAME start head={snake[0]} food={food} grid={w}x{h}")

        running=True
        while running:
            # Çıkış: ESC / herhangi tuş / mouse tık/teker
            for e in pygame.event.get():
                if e.type==pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                if e.type==pygame.KEYDOWN:
                    log("EXIT keydown"); save_highscores(all_time_high, all_time_rate, daily_high, daily_rate)
                    pygame.quit(); sys.exit(0)
                if e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEWHEEL):
                    log("EXIT mouse"); save_highscores(all_time_high, all_time_rate, daily_high, daily_rate)
                    pygame.quit(); sys.exit(0)

            now=time.time()
            # hız dinamik
            if now-last_move >= move_interval:
                # yön seç
                d, why = find_next_dir(snake, food, w, h, prev_dir)
                if d is None:
                    log("STUCK no-move -> game over")
                    running=False
                    break

                head=snake[0]
                new_head=(head[0]+d[0], head[1]+d[1])

                # çarpışma
                if not (0<=new_head[0]<w and 0<=new_head[1]<h) or (new_head in snake):
                    log(f"HIT new_head={new_head} score={score}")
                    running=False
                    break

                snake.insert(0, new_head)
                ate = (new_head==food)
                if ate:
                    score+=1
                    food = place_food(snake, w, h)
                    move_interval = max(0.006, 0.015 - score/3000.0)  # yavaş yavaş hızlan
                    log(f"EAT score={score} new food={food} why={why}")
                else:
                    snake.pop()

                prev_dir = d
                last_move = now

                # çiz
                SCREEN.fill(BLACK)
                # skorlar köşe
                draw_text(f"Tüm Zamanlar: {all_time_high}", 28, GRAY, WIDTH-160, 20, alpha=160)
                draw_text(f"Günlük: {daily_high}", 28, GRAY, WIDTH-140, 45, alpha=160)

                pts=[(c[0]*BLOCK_SIZE+BLOCK_SIZE//2, c[1]*BLOCK_SIZE+BLOCK_SIZE//2) for c in snake]
                pulse = 0.7 + 0.3*math.sin(now*2.0)
                for i in range(len(pts)-1):
                    gv = int(150 + 105 * max(0, (len(snake)-i-1))/10.0) if i>=len(snake)-10 else int(255*pulse)
                    color=(0, gv, 0)
                    pygame.draw.line(SCREEN, color, pts[i], pts[i+1], SNAKE_WIDTH)
                for i,p in enumerate(pts):
                    gv = int(150 + 105 * max(0,(len(snake)-i))/10.0) if i>=len(snake)-10 else int(255*pulse)
                    color = GREEN if i>0 else (int(255*pulse),0,0) if False else GREEN
                    pygame.draw.circle(SCREEN, color, p, SNAKE_WIDTH//2)
                # food
                pf=0.8+0.2*math.sin(now*3.0); rv=int(255*pf)
                pygame.draw.circle(SCREEN, (rv,0,0), (food[0]*BLOCK_SIZE+BLOCK_SIZE//2, food[1]*BLOCK_SIZE+BLOCK_SIZE//2), BLOCK_SIZE//2)

                draw_text(f"Skor: {score}  ({why})", 30, WHITE, WIDTH//2, 22)

                pygame.display.flip()
                clock.tick(FPS)

        # oyun bitti -> skorları güncelle, ekran ver
        end=time.time(); dur=end-start_t
        rate = score/dur if dur>0 else 0.0
        if score>all_time_high: all_time_high, all_time_rate = score, rate
        if score>daily_high: daily_high, daily_rate = score, rate
        save_highscores(all_time_high, all_time_rate, daily_high, daily_rate)

        SCREEN.fill(BLACK)
        draw_text("Oyun Bitti", 56, RED, WIDTH//2, HEIGHT//3)
        draw_text(f"Skor: {score}  ({rate:.2f}/s)", 40, WHITE, WIDTH//2, HEIGHT//2)
        pygame.display.flip()
        time.sleep(1.2)
        # yeni tur
        # (screensaver mantığı: kullanıcı giriş yaparsa yukarıda zaten çıkıyor)
        
if __name__=="__main__":
    try:
        main()
    except SystemExit:
        pass
    except Exception as e:
        log(f"FATAL: {e}")
