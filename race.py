from PIL import Image
import random
import numpy as np
from math import log
import argparse
from tqdm import tqdm
import sys

if sys.version_info < (3,0):
    raise "must use python 3.x"

linecol = (255,255,0)
startcol = (255,0,0)
speedcol = (0,0,255)
emptycol = (0,0,0)

def norm(v):
    return v / abs(v) if v != 0 else 0

def absspeed(speed):
    return abs(speed[0]) + abs(speed[1])

def getpixel(im, data, x, y):
    return data[y*im.width+x]

def setpixel(im, data, x, y, c):
    data[y*im.width+x] = c

def colidxs(im):
    emptyidx = -1
    lineidx = -1
    startidx = -1
    speedidx = -1
    p = im.getpalette()
    for i in range(int(len(p) / 3)):
        if emptyidx == -1 and (p[3*i] == emptycol[0] and p[3*i+1] == emptycol[1] and p[3*i+2] == emptycol[2]):
            emptyidx = i
        if lineidx == -1 and (p[3*i] == linecol[0] and p[3*i+1] == linecol[1] and p[3*i+2] == linecol[2]):
            lineidx = i
        if startidx == -1 and (p[3*i] == startcol[0] and p[3*i+1] == startcol[1] and p[3*i+2] == startcol[2]):
            startidx = i
        if speedidx == -1 and (p[3*i] == speedcol[0] and p[3*i+1] == speedcol[1] and p[3*i+2] == speedcol[2]):
            speedidx = i
    if emptyidx == -1 or lineidx == -1 or startidx == -1 or speedidx == -1:
        raise RuntimeError("cannot find colors for start") from None
    return (emptyidx, lineidx, startidx, speedidx)

def findpixel(im, data, idx):
    for y in range(im.height):
        for x in range(im.width):
            if getpixel(im, data, x, y) == idx:
                return (x,y)
    raise RuntimeError("no starting/speed point found") from None

def findstart(im, data, startidx, speedidx):
    x,y = findpixel(im, data, startidx)
    dx,dy = findpixel(im, data, speedidx)

    return (x,y, dx-x,dy-y)

def getfinal(im, data):
    emptyidx, lineidx, startidx, speedidx = colidxs(im)
    x,y,vx,vy = findstart(im, data, startidx, speedidx)
    res = { (x,y): True }

    if vy != 0:
        i = 1
        while x + i < im.width and getpixel(im, data, x + i, y) == lineidx:
            res[(x + 1,y)] = True
            i += 1
        i = 1
        while x >= i and getpixel(im, data, x - i, y) == lineidx:
            res[(x - 1,y)] = True
            i += 1
    elif vx != 0:
        i = 1
        while y + i < im.height and getpixel(im, data, x, y + i) == lineidx:
            res[(x,y + i)] = True
            i += 1
        i = 1
        while y >= i and getpixel(im, data, x, y - i) == lineidx:
            res[(x,y - i)] = True
            i += 1
    else:
        raise RuntimeError("invalid speed") from None

    return (res, (x,y), (vx,vy), emptyidx, speedidx)

def allowed(track, data, x, y, dx, dy, allowedidx):
    if not getpixel(track, data, x+dx, y+dy) in allowedidx:
        return False
    if abs(dx) == 1 and abs(dy) == 1:
        if not getpixel(track, data, x+dx, y) in allowedidx and not getpixel(track, data, x, y+dy) in allowedidx:
            return False
    return True

def possible_speeds(speed):
    speedx, speedy = speed
    dspeedy = 1 if speedy > 0 else -1
    if speedx != 0:
        dspeedx = 1 if speedx > 0 else -1
        if speedy != 0:
            return [speed,
                    (speedx - dspeedx, speedy + dspeedy),
                    (speedx + dspeedx, speedy - dspeedy),
                    (speedx, speedy + dspeedy),
                    (speedx - dspeedx, speedy),
                    (speedx, speedy - dspeedy),
                    (speedx + dspeedx, speedy)]
        res = [speed,
               (speedx - dspeedx, 1),
               (speedx - dspeedx, -1),
               (speedx, 1),
               (speedx, -1),
               (speedx + dspeedx, 0)]
        if abs(speedx) > 1:
            res.append((speedx - dspeedx, 0))            
        return res
    res = [speed,
           (1, speedy - dspeedy),
           (-1, speedy - dspeedy),
           (0, speedy + dspeedy),
           (1, speedy),
           (-1, speedy)]
    if abs(speedy) > 1:
        res.append((0, speedy - dspeedy))
    return res

def getr(d1, d2, va, Qgoal):
    if d1 < 0 or d2 < 0 or d1 - d2 > va:
        return None

    if d1 < d2:
        if d1 <= va and d2 > 5 * va:
            return Qgoal
        return (d1 - d2) - (va >> 1) / (d1 - d2)
    else:
        return (1 + d1 - d2) * va

def known_speeds(score, speeds):
    return [ s for s in speeds if s in score ]

def dump_score(nr, trackname, track, tbbl, tbbr, tbbt, tbbb, score, final, extra=None):
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

        if maxscore == 0.0:
            return
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
        img.save("{}-sol-{:07d}.png".format(trackname, nr))

def best_path(dist, score, start, v, tracklen):
    res = [ start ]
    prevco = start
    co = (start[0]+v[0], start[1]+v[1])
    d = dist[co[0]][co[1]]
    prevd = d+1

    while (prevd > absspeed(v) or d < 10 * tracklen) and len(res) < 2 * tracklen:
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
        d = dist[co[0]][co[1]]
    return res

def compute(trackname, epsilon, alpha, gamma, Q0, Qfail, Qgoal, totalrounds):
    trackfname = trackname+".png"
    with Image.open(trackfname) as track:
        data = track.getdata()

        dist = [[-1]*track.height for i in range(track.width)]
        score = [[{} for j in range(track.height)] for i in range(track.width)]

        final, start, startspeed, emptyidx, speedidx = getfinal(track, data)
        allowedidx = [emptyidx, speedidx]

        dir = (int(norm(startspeed[0])), int(norm(startspeed[1])))

        candidates = []
        for co in final:
            dist[co[0]][co[1]] = 0
            if getpixel(track, data, co[0]-dir[0], co[1]-dir[1]) in allowedidx:
                candidates.append((co[0]-dir[0], co[1]-dir[1]))

        tracklen = 1

        while len(candidates) > 0:
            nextc = []

            for ca in candidates:
                if getpixel(track, data, *ca) in allowedidx and dist[ca[0]][ca[1]] < 0:
                    dist[ca[0]][ca[1]] = tracklen

                    for dx,dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                        co = (ca[0]+dx, ca[1]+dy)

                        if dist[co[0]][co[1]] < 0 and not co in candidates and allowed(track, data, *ca, dx, dy, allowedidx):
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

        tq = tqdm(total=totalrounds, desc=trackname)
        for rounds in range(totalrounds):
            co = (start[0] + startspeed[0], start[1] + startspeed[1])
            speed = startspeed
            aspeed = absspeed(speed)
            nextspeeds = possible_speeds(speed)
            curdist = dist[co[0]][co[1]]
            scores = score[co[0]][co[1]]
            speedspp = possible_speeds(speed)
            known = known_speeds(scores, nextspeeds)
            bestscore = Qfail
            bestspeeds = []
            for s in known:
                thisscore = scores[s]
                if thisscore == bestscore:
                    bestspeeds.append(s)
                elif thisscore > bestscore:
                    bestscore = thisscore
                    bestspeeds = [ s ]
            if bestscore < Q0 and len(known) != len(speedspp):
                bestscore = Q0
                bestspeeds = set(nextspeeds) - set(known)
            path = [ ]
            reached_goal = False

            while True:
                path.append((speed, co))

                if random.random() < epsilon:
                    nextspeed = random.sample(nextspeeds, 1)[0]
                else:
                    # Pick according to current score
                    nextspeed = random.sample(bestspeeds, 1)[0]

                nextco = (co[0] + nextspeed[0], co[1] + nextspeed[1])
                nextdist = dist[nextco[0]][nextco[1]]
                nextaspeed = absspeed(nextspeed)

                r = getr(curdist, nextdist, nextaspeed, Qgoal)
                if not r:
                    # crash
                    ##print("crash at ({},{})".format(co[0]+nextspeed[0],co[1]+nextspeed[1]))
                    score[co[0]][co[1]][nextspeed] = Qfail
                    break

                if curdist <= aspeed and nextdist > 10 * aspeed:
                    nextbestscore = Qgoal
                    reached_goal = True
                else:
                    scores = score[nextco[0]][nextco[1]]
                    speedspp = possible_speeds(nextspeed)
                    known = known_speeds(scores, speedspp)
                    bestscore = Qfail
                    bestspeeds = []
                    for s in known:
                        thisscore = scores[s]
                        if thisscore == bestscore:
                            bestspeeds.append(s)
                        elif thisscore > bestscore:
                            bestscore = thisscore
                            bestspeeds = [ s ]
                    if bestscore < Q0 and len(known) != len(speedspp):
                        bestscore = Q0
                        bestspeeds = set(speedspp) - set(known)

                oldval = score[co[0]][co[1]][nextspeed] if nextspeed in score[co[0]][co[1]] else Q0
                score[co[0]][co[1]][nextspeed] = (1 - alpha) * oldval + alpha * (r + gamma * bestscore)

                if reached_goal or len(path) > 5 * tracklen:
                    break

                co = nextco
                speed = nextspeed
                aspeed = nextaspeed
                nextspeeds = speedspp
                curdist = nextdist

            #print("stop at {}".format(co))

            tq.update()
            if rounds % 10000 == 0:
                dump_score(rounds, trackname, track, tbbl, tbbr, tbbt, tbbb, score, final, best_path(dist, score, start, startspeed, tracklen))


        tq.close()
        dump_score(totalrounds, trackname, track, tbbl, tbbr, tbbt, tbbb, score, final, best_path(dist, score, start, startspeed, tracklen))

if __name__ == '__main__':
    trackname = "track2"

    parser = argparse.ArgumentParser(description='Racing with Reinforcement Learning.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fnames', metavar='FNAME', type=str, nargs='+', help='PNG of track')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.01, help='Fraction of random steps')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.1, help='Learning Rate')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.95, help='Discount Factor')
    parser.add_argument('--Q0', dest='Q0', type=float, default=0.0, help='Initial Q function value')
    parser.add_argument('--Qfail', dest='Qfail', type=float, default=-10000, help='Q function value for crash')
    parser.add_argument('--Qgoal', dest='Qgoal', type=float, default=1000, help='Q function value for reaching goal')
    parser.add_argument('--N', dest='nrounds', type=int, default=4000000, help='Number of learning runs')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='Seed random number generator')
    args=parser.parse_args()

    for trackname in args.fnames:
        if args.seed:
            random.seed(args.seed)
        compute(trackname, args.epsilon, args.alpha, args.gamma, args.Q0, args.Qfail, args.Qgoal, args.nrounds)
