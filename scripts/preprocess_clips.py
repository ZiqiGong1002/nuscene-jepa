"""nuScenes mini -> V-JEPA format: CAM_FRONT frames -> 24-frame h264 clips + CSV index.
Frame order comes from the sample_data prev/next chain (keyframes + sweeps, 12Hz).
Hard assertions: timestamps strictly increasing, median gap ~83ms."""
import os, subprocess, tempfile, csv, statistics
from nuscenes.nuscenes import NuScenes

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
ROOT = os.path.join(DATA, 'nuscenes')
OUT  = os.path.join(DATA, 'clips')
CSVP = os.path.join(DATA, 'nuscenes_mini_clips.csv')
CHUNK = 24   # frames per clip = 2s @ 12Hz
os.makedirs(OUT, exist_ok=True)

def ts_of(path):  # filename ends with __<microsecond timestamp>.jpg
    return int(path.rsplit('__', 1)[1].split('.')[0])

nusc = NuScenes(version='v1.0-mini', dataroot=ROOT, verbose=False)
rows = []
for scene in nusc.scene:
    sample = nusc.get('sample', scene['first_sample_token'])
    sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    while sd['prev'] != '':
        sd = nusc.get('sample_data', sd['prev'])
    frames = []
    while True:
        frames.append(os.path.join(ROOT, sd['filename']))
        if sd['next'] == '':
            break
        sd = nusc.get('sample_data', sd['next'])

    # ---- hard verification: strict temporal order, ~83ms gaps ----
    ts = [ts_of(f) for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:])), f"{scene['name']}: timestamps not increasing"
    gaps = [(b - a) / 1000.0 for a, b in zip(ts, ts[1:])]  # ms
    med = statistics.median(gaps)
    assert 60 < med < 110, f"{scene['name']}: median gap {med:.1f}ms, expected ~83ms"

    n_chunks = len(frames) // CHUNK
    print(f"{scene['name']}: {len(frames)} frames, median gap {med:.1f}ms -> {n_chunks} clips")
    for ci in range(n_chunks):
        chunk = frames[ci*CHUNK:(ci+1)*CHUNK]
        outp = os.path.join(OUT, f"{scene['name']}_c{ci:02d}.mp4")
        if os.path.exists(outp):
            rows.append(outp); continue
        with tempfile.TemporaryDirectory() as td:
            for i, f in enumerate(chunk):
                os.symlink(f, os.path.join(td, f"{i:05d}.jpg"))
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12",
                 "-i", os.path.join(td, "%05d.jpg"),
                 "-vf", "scale=-2:256", "-c:v", "libx264", "-crf", "23",
                 "-pix_fmt", "yuv420p", outp], capture_output=True, text=True)
            if r.returncode != 0:
                print("FFMPEG FAIL", outp, r.stderr[:300]); raise SystemExit(1)
        rows.append(outp)

with open(CSVP, "w", newline="") as f:
    w = csv.writer(f, delimiter=" ")
    for p in rows:
        w.writerow([p, 0])
print(f"\nTOTAL: {len(rows)} clips -> {CSVP}")
