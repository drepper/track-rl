from PIL import Image
import random
import numpy as np
from math import log
import sys

trackname = "track1.png"

start = (330, 156)
startspeed = (1, 0)

epsilon = 0.025
alpha = 0.25
gamma = 0.95
Q0 = 0.1

dist = []

def norm(v):
    return v / abs(v) if v != 0 else 0

def getpixel(im, data, x, y):
    return data[y*im.width+x]

def setpixel(im, data, x, y, c):
    data[y*im.width+x] = c

def getfinal(im, data, start, startspeed):
    x, y = start
    speedx, speedy = startspeed
    color = getpixel(im, data, x, y)
    res = { start: True }
    assert abs(speedx) + abs(speedy) == 1
    if speedx == 0:
        i = 1
        while x + i < im.width and getpixel(im, data, x + i, y) == color:
            res[(x + 1,y)] = True
            i += 1
        i = 1
        while x >= i and getpixel(im, data, x - i, y) == color:
            res[(x - 1,y)] = True
            i += 1
    else:
        i = 1
        while y + i < im.height and getpixel(im, data, x, y + i) == color:
            res[(x,y + i)] = True
            i += 1
        i = 1
        while y >= i and getpixel(im, data, x, y - i) == color:
            res[(x,y - i)] = True
            i += 1

    return res

def getdist(x, y):
    return dist[x][y]

def setdist(x, y, v):
    dist[x][y] = v

def allowed(track, data, x, y, dx, dy):
    if getpixel(track, data, x+dx, y+dy) != 0:
        return False
    if abs(dx) == 1 and abs(dy) == 1:
        if getpixel(track, data, x+dx, y) != 0 and getpixel(track, data, x, y+dy) != 0:
            return False
    return True

def absspeed(speed):
    return abs(speed[0]) + abs(speed[1])

def possible_speeds(speed):
    res= []
    speedx, speedy = speed
    l = absspeed(speed)
    if l > 1:
        if speedx != 0:
            res.append((speedx - int(speedx / abs(speedx)), speedy))
        if speedy != 0:
            res.append((speedx, speedy - int(speedy / abs(speedy))))
    res.append(speed)
    if speedx != 0:
        if speedy != 0:
            res.append((speedx - int(speedx / abs(speedx)), speedy + int(speedy / abs(speedy))))
        else:
            res.append((speedx - int(speedx / abs(speedx)), 1))
            res.append((speedx - int(speedx / abs(speedx)), -1))
    if speedy != 0:
        if speedx != 0:
            res.append((speedx + int(speedx / abs(speedx)), speedy - int(speedy / abs(speedy))))
        else:
            res.append((1, speedy - int(speedy / abs(speedy))))
            res.append((-1, speedy - int(speedy / abs(speedy))))
    if speedx != 0:
        res.append((speedx + int(speedx / abs(speedx)), speedy))
    else:
        res.append((1, speedy))
        res.append((-1, speedy))
    if speedy != 0:
        res.append((speedx, speedy + int(speedy / abs(speedy))))
    else:
        res.append((speedx, 1))
        res.append((speedx, -1))
    return res

def getr(dist, x, y, vx, vy):
    d1 = dist[x][y]
    d2 = dist[x+vx][y+vy]
    va = absspeed((vx,vy))
    if d1 < 0 or d2 < 0 or d1 - d2 > va:
        return None

    if d1 < d2:
        return (d1 - d2) - (va >> 1) / (d1 - d2)
    else:
        return (1 + d1 - d2) * va

def known_speeds(score, speeds):
    return [ s for s in speeds if s in score ]

def dump_score(nr, track, score, final, extra=None):
    width = track.width
    height = track.height
    with Image.new('RGB', (width, height)) as img:
        d = track.convert('RGB').getdata()
        gen = [ (np.uint8(0),np.uint8(0),np.uint8(0)) for _ in range(width * height) ]
        x = 0
        y = 0

        maxscore = 0.0
        for sy in score[tbbl:tbbr]:
            for sc in sy[tbbt:tbbb]:
                if len(sc) > 0:
                    maxscore = max(maxscore, sc[max(sc, key=sc.get)])

        maxscore = log(maxscore)
        if maxscore == 0.0:
            return
        for r,g,b in d:
            if (x,y) in final:
                col = (np.uint8(255),np.uint8(255),np.uint8(0))
            elif len(score[x][y]) > 0:
                v = score[x][y][max(score[x][y], key=score[x][y].get)]
                if v <= 0:
                    col = (np.uint8(0),np.uint8(0),np.uint8(0))
                else:
                    v = log(v) / maxscore
                    col = (np.uint8(255*v),np.uint8(255*v),np.uint8(255*v))
            else:
                col = (np.uint8(r),np.uint8(g),np.uint8(b))
            gen[x+y*width] = col
            x += 1
            if x == track.width:
                x = 0
                y += 1

        if extra:
            for co in extra:
                gen[co[0]+co[1]*width] = (np.uint8(255),np.uint8(0),np.uint8(0))

        img.putdata(gen)
        img.save("track1-sol-{:07d}.png".format(nr))

def best_path(score, start, v):
    res = [ start ]
    prevco = start
    co = (start[0]+v[0], start[1]+v[1])
    d = getdist(*co)
    prevd = d+1

    while prevd - d <= absspeed(v):
        res.append(co)
        speeds = possible_speeds(v)
        s = score[co[0]][co[1]]
        known = known_speeds(s, speeds)
        if len(known) == 0:
            break
        bestmove = known[0]
        bestscore = s[bestmove]
        for k in known[1:]:
            if s[k] > bestscore:
                bestmove = k
                bestscore = s[k]
        v = bestmove
        prevco = co
        prevd = d
        co = (co[0]+bestmove[0],co[1]+bestmove[1])
        d = getdist(*co)
    return res

with Image.open(trackname) as track:
    data = track.getdata()

    dist = [[-1]*track.height for i in range(track.width)]
    score = [[{} for j in range(track.height)] for i in range(track.width)]

    final = getfinal(track, data, start, startspeed)

    for co in final:
        setdist(*co, 0)

    dir = (int(norm(startspeed[0])), int(norm(startspeed[1])))

    candidates = [ (x-dir[0], y-dir[1]) for x,y in final if getpixel(track, data, x-dir[0], y-dir[1]) == 0 ]
    tracklen = 1

    while len(candidates) > 0:
        nextc = []

        for ca in candidates:
            if getpixel(track, data, *ca) == 0 and dist[ca[0]][ca[1]] < 0:
                setdist(*ca, tracklen)

                for dx,dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                    co = (ca[0]+dx, ca[1]+dy)

                    if dist[co[0]][co[1]] < 0 and not co in candidates and allowed(track, data, *ca, dx, dy):
                        nextc.append(co)
        candidates = list(set(nextc))
        tracklen += 1

    # bounding box for track
    tbbl = -1
    tbbr = -1
    tbbt = -1
    tbbb = -1
    for i in range(track.width):
        if tbbl < 0 and max(dist[i]) >= 0:
            tbbl = i
        if tbbr < 0 and max(dist[track.width-1-i]) >= 0:
            tbbr = track.width-1-i
    for i in range(track.height):
        for j in range(track.width):
            if tbbt < 0 and dist[j][i] >= 0:
                tbbt = i
            if tbbb < 0 and dist[j][track.height-1-i] >= 0:
                tbbb = track.height-1-i
    tbbw = tbbr - tbbl + 1
    tbbh = tbbb - tbbt + 1

    totalrounds = 4000000
    for rounds in range(totalrounds):
        co = (start[0] + startspeed[0], start[1] + startspeed[1])
        speed = startspeed
        prevdist = getdist(*co)
        path = [ ]
        while True:
            curdist = getdist(*co)
            if curdist <= 0 or prevdist - curdist > absspeed(speed):
                break
            # print("distance: {}".format(curdist))
            path.append((speed, co))

            nextspeeds = possible_speeds(speed)
            if random.random() < epsilon:
                nextspeed = random.sample(nextspeeds, 1)[0]
            else:
                # Pick according to current score
                known = known_speeds(score[co[0]][co[1]], nextspeeds)

                bestscore = -1
                bestspeeds = []
                for s in known:
                    thisscore = score[co[0]][co[1]][s] if s in score[co[0]][co[1]] else 0.0
                    if thisscore == bestscore:
                        bestspeeds.append(s)
                    elif thisscore > bestscore:
                        bestscore = thisscore
                        bestspeeds = [ s ]
                if bestscore < 0:
                    if len(known) == len(nextspeeds):
                        # All speeds bad
                        for s in nextspeeds:
                            score[co[0]][co[1]][s] = -100
                        break
                    nextspeed = random.sample(set(nextspeeds) - set(score[co[0]][co[1]].keys()), 1)[0]
                else:
                    nextspeed = random.sample(bestspeeds, 1)[0]

            newco = (co[0] + nextspeed[0], co[1] + nextspeed[1])

            r = getr(dist, *co, *nextspeed)
            if not r:
                # crash
                score[co[0]][co[1]][nextspeed] = -100
                break

            nextscores = score[newco[0]][newco[1]]
            speedspp = possible_speeds(nextspeed)
            known = known_speeds(nextscores, speedspp)
            if len(known) == 0:
                nextbestscore = Q0
            else:
                knownscores = [nextscores[k] for k in known]
                nextbestscore =  max(knownscores)
                if nextbestscore < Q0 and len(speedspp) > len(known):
                    nextbestscore = Q0

            if nextspeed in score[co[0]][co[1]]:
                score[co[0]][co[1]][nextspeed] = (1 - alpha) * score[co[0]][co[1]][nextspeed] + alpha * (r + gamma * nextbestscore - score[co[0]][co[1]][nextspeed])
            else:
                score[co[0]][co[1]][nextspeed] = (1 - alpha) * Q0 + alpha * (r + gamma * nextbestscore)

            co = newco
            speed = nextspeed
            prevdist = curdist

        # print("path = {}".format(path))
        print("stop at {}".format(co))

        if rounds % 10000 == 0:
            dump_score(rounds, track, score, final)


    dump_score(totalrounds, track, score, final, best_path(score, start, startspeed))
