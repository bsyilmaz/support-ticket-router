"""Record the demo video: a real-time screen recording of the running Gradio app (new inputs are
typed live by a script), followed by the architecture / results / conclusion slides, with the
narration audio on top.

    python make_video.py narration.wav            # app must be running on http://127.0.0.1:7860

Needs: pip install playwright && playwright install --with-deps chromium; ffmpeg; pdftoppm;
docs/presentation.pdf (python make_slides.py).  Output: docs/demo_video.mp4
"""
import glob
import os
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "http://127.0.0.1:7860")
AUDIO = sys.argv[1]
AUDIO_SKIP = float(os.environ.get("AUDIO_SKIP", "2.40"))   # leading seconds of the narration to drop
W, H = 1280, 720

# (start second on the narration timeline, message). Times follow the pauses in the narration.
MESSAGES = [
    (14.0, "my card got declined at the grocery store"),
    (26.5, "plese tranfer 300 dolars to my savngs acount"),
    (35.9, "my dishwasher won't drain, model DW-6534"),
    (45.0, "hello, how are you?"),
]
DEMO_END = 47.7
SLIDES = [(4, 58.3), (7, 76.5), (8, None)]   # (slide number, end second); None = until audio ends

SCROLL = """(txt) => { const els = [...document.querySelectorAll('p, label, span, h3')];
  const el = els.find(e => e.textContent.includes(txt)); if (el) el.scrollIntoView({behavior: 'smooth', block: 'center'}); }"""
TOP = "() => { window.scrollTo({top: 0, behavior: 'smooth'}); document.querySelectorAll('*').forEach(e => { if (e.scrollTop > 0) e.scrollTo({top: 0, behavior: 'smooth'}); }); }"


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def record():
    os.makedirs("video_tmp", exist_ok=True)
    for f in glob.glob("video_tmp/*.webm"):
        os.remove(f)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("PW_CHANNEL") or None)
        ctx = browser.new_context(viewport={"width": W, "height": H},
                                  record_video_dir="video_tmp", record_video_size={"width": W, "height": H})
        t_ctx = time.time()
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        box = page.locator("textarea").first
        box.wait_for()
        page.wait_for_timeout(1500)
        t0 = time.time()                      # narration second 0

        def until(sec):
            page.wait_for_timeout(max(0, (t0 + sec - time.time()) * 1000))

        def send(msg):
            page.evaluate(TOP)
            box.click()
            box.fill("")
            box.type(msg, delay=45)
            page.wait_for_timeout(350)
            page.get_by_role("button", name="Route message").click()

        t, m = MESSAGES[0]
        until(t); send(m)
        until(22.9); page.evaluate(SCROLL, "All four models side by side")
        t, m = MESSAGES[1]
        until(t); send(m)
        until(30.2); page.evaluate(SCROLL, "After spelling correction")
        t, m = MESSAGES[2]
        until(t); send(m)
        until(38.9); page.evaluate(SCROLL, "Model codes (regex)")
        until(41.2); page.evaluate(TOP)
        t, m = MESSAGES[3]
        until(t); send(m)
        until(DEMO_END + 0.6)
        offset = t0 - t_ctx
        ctx.close()
        browser.close()
    return glob.glob("video_tmp/*.webm")[0], offset


def main():
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", AUDIO]).decode()) - AUDIO_SKIP
    webm, offset = record()
    print(f"recorded {webm}, page-ready offset {offset:.2f}s, narration {dur:.1f}s", flush=True)
    run(["pdftoppm", "-r", "96", "-png", "docs/presentation.pdf", "video_tmp/slide"])
    inputs = ["-ss", f"{offset:.2f}", "-t", f"{DEMO_END:.2f}", "-i", webm]
    filt = [f"[0:v]fps=25,scale={W}:{H},setsar=1,format=yuv420p[v0]"]
    start = DEMO_END
    for i, (n, end) in enumerate(SLIDES, 1):
        end = dur + 0.3 if end is None else end
        png = sorted(glob.glob(f"video_tmp/slide-*{n:02d}.png") or glob.glob(f"video_tmp/slide-{n}.png"))[0]
        inputs += ["-loop", "1", "-t", f"{end - start:.2f}", "-i", png]
        filt.append(f"[{i}:v]fps=25,scale={W}:{H},setsar=1,format=yuv420p[v{i}]")
        start = end
    k = len(SLIDES) + 1
    filt.append("".join(f"[v{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=0[v]")
    filt.append(f"[{k}:a]atrim=start={AUDIO_SKIP},asetpts=PTS-STARTPTS[a]")
    out = "docs/demo_video.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-i", AUDIO, "-filter_complex", ";".join(filt),
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", out])
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
