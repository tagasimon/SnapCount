# SnapCount

A one-screen iPhone app: take or choose a photo of your stock, tap **Count**, and get an AI estimate of how many items are in it, with a confidence level and notes on what limits accuracy.

```
iPhone app (SwiftUI)  ──photo + item name──▶  Backend (FastAPI)  ──▶  Claude vision API
                      ◀──count, confidence──                     ◀──
```

The AI key lives only on the backend. Photos are processed in memory and never stored.

## What's in the folder

```
snapcount/
├── backend/                 FastAPI server
│   ├── app/main.py          POST /count, GET /health, upload validation
│   ├── app/detectors.py     Free models: OWLv2, Grounding DINO, OmDet-Turbo
│   ├── app/consensus.py     How Auto combines the models' counts
│   ├── app/providers.py     Claude (paid) and mock providers
│   ├── app/prompt.py        Counting instructions and structured-output schema
│   ├── app/config.py        Settings from environment variables
│   ├── tests/               28 automated tests
│   ├── Dockerfile
│   └── .env.example
├── ios/
│   ├── SnapCount.xcodeproj  Open this in Xcode
│   ├── SnapCount/           Swift source, assets, Info.plist
│   │   └── Config.swift     ← set your backend URL here
│   └── project.yml          Fallback (XcodeGen) if the project won't open
└── eval/                    Measure accuracy on your own real photos
```

---

## Counting models

The app has a model picker at the top right. Each model is told the item name and returns
a dot on every item it counted, so you can check the result on the photo.

| Model | Cost | Notes |
|---|---|---|
| **Auto** (default) | free | Runs the three free models and shows the middle count, plus each model's answer. Confidence comes from how well they agree. |
| OWLv2 (Google) | free | Best of the three on logs, cans and books in our test. |
| Grounding DINO (IDEA Research) | free | Best single model overall in our test; good on bricks and boxes. |
| OmDet-Turbo | free | Fastest (about 2–7 s on a CPU). |
| Stacked items | free | For nested stacks (plastic chairs, plates, buckets). Counts one edge per item instead of whole objects. Auto uses it automatically for nestable items when the detectors only see 1–3 objects. |
| Claude | paid | Only offered if `ANTHROPIC_API_KEY` is set. No dots. |

The free models are Apache-2.0, run on your own machine or VPS, and download once
(about 1.7 GB) on first start. They're published as safetensors, so loading them
doesn't run any downloaded code.

### How accurate are they? (measured, small sample)

Tested on 35 photos from FSC-147, a public counting benchmark with hand-counted dots (chairs, logs,
bricks, boxes, cement/rice bags, bottles, cans, cups, eggs, plates, books, shoes, pots, apples;
10–199 items each). Thresholds were tuned on 18 of them and scored on the other 17:

| | Median error | Within 10% of the true count |
|---|---|---|
| OWLv2 | 33% | 6 / 17 |
| Grounding DINO | 30% | 8 / 17 |
| OmDet-Turbo | 48% | 5 / 17 |
| **Auto** | **30%** | **8 / 17** |

- **When Auto said "high confidence", it was exactly right (2 of 2); "medium" was 6% off (1 of 1).**
  Everything else was flagged "low", so trust the badge, not the number, on low results.
- Good at: eggs, apples, plates, bricks, books, shoes, flower pots.
- Weak at: big log piles, stacked bags of cement and rice, tall shelves of books.
  On those, often only one model gets close, and Auto can't tell which one.
- 17 photos is a small sample. Run `eval/run_eval.py` on your own stock photos before relying on it.

Re-run the benchmark with `python eval/benchmark_fsc147.py --images <folder> --labels <csv>`.

## 1. Run the backend

You need Python 3.11+. No API key is needed for the free models.

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # includes PyTorch; a few minutes the first time
cp .env.example .env                     # optional: edit settings
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The first start downloads the models (about 1.7 GB), then loads them; wait for
`OmDet-Turbo ready` in the log. Check `http://localhost:8000/health`.

On an Apple Silicon Mac the models use the GPU automatically (MPS). Auto takes about
45–60 s per photo on a 2-core CPU server; a Mac or a GPU is much faster.

**Testing the app without models:** `SNAPCOUNT_MODELS=mock uvicorn app.main:app --host 0.0.0.0 --port 8000`
always answers 12.

Quick test from the terminal:

```bash
curl -F image=@some-photo.jpg -F item="plastic chairs" -F model=auto http://localhost:8000/count
curl http://localhost:8000/models
```

### Environment variables

| Variable | Default | What it does |
|---|---|---|
| `SNAPCOUNT_MODELS` | `owlv2,grounding_dino,omdet_turbo,claude` | Models the server offers (`claude` only appears if its key is set; `mock` for testing) |
| `AUTO_MODELS` | `owlv2,grounding_dino,omdet_turbo` | Models Auto runs and combines |
| `PRELOAD_MODELS` | `1` | Load the free models at startup so the first count isn't slow |
| `DETECT_TILED` | `0` | `1` also scans zoomed-in tiles: ~5× slower, sometimes better on dense piles |
| `OWLV2_THRESHOLD` etc. | tuned | Per-model score cut-off. Lower counts more, with more false hits |
| `TORCH_DEVICE` | auto | Force `cpu`, `mps` or `cuda` |
| `ANTHROPIC_API_KEY` | – | Only for the paid Claude model |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Claude model to use |
| `APP_TOKEN` | empty | Shared secret. If set, the app must send the same value. Set this once the server is public |
| `MAX_UPLOAD_MB` | `10` | Upload limit |
| `MAX_IMAGE_SIDE` | `1600` | Server shrinks images to this before counting |

### Run the tests

```bash
cd backend
pip install pytest
python -m pytest -q
```

### Deploy (cheapest practical option)

Your existing VPS with Coolify is the cheapest route. With the three free models loaded the
server needs **about 4 GB of RAM** (8 GB is comfortable) and ~6 GB of disk for the image and
model files. No GPU is needed, but counts are slower without one.

1. Push this folder to a Git repo.
2. In Coolify: **New Resource → your repo → Build pack: Dockerfile**, base directory `/backend`, port `8000`.
3. Add the environment variables (at least `APP_TOKEN`), and a persistent volume mounted at
   `/models` so the model files aren't downloaded again on every deploy.
4. Give it a domain, e.g. `snapcount.elastictech.biz`. Coolify issues the HTTPS certificate.
5. Check `https://your-domain/health`.

The deployed address **must be HTTPS**; iPhones block plain `http://` to internet addresses.

---

## 2. Run the app on your iPhone

You need a Mac with **Xcode 16 or newer**, and an iPhone on iOS 17+. A free Apple ID is enough for your own phone.

1. Open `ios/SnapCount.xcodeproj` in Xcode.
2. Edit `ios/SnapCount/Config.swift`:
   - **Testing at home/office:** `backendURL` = your computer's local IP, e.g. `http://192.168.1.20:8000` (find it in System Settings → Wi-Fi → Details). iPhone and computer must be on the same Wi-Fi, and the backend must be started with `--host 0.0.0.0`.
   - **Deployed:** `https://snapcount.yourdomain`.
   - `appToken` = the same value as `APP_TOKEN` on the server (leave empty if you didn't set one).
3. Click the **SnapCount** project → target **SnapCount** → **Signing & Capabilities**:
   - Tick *Automatically manage signing*.
   - **Team:** add your Apple ID (Xcode → Settings → Accounts) and select it.
   - If Xcode says the bundle identifier is taken, change `biz.elastictech.snapcount` to something unique.
4. Connect the iPhone by cable, unlock it, tap **Trust**.
5. On the iPhone, turn on **Settings → Privacy & Security → Developer Mode** (it restarts).
6. In Xcode pick your iPhone at the top, press **▶ Run**.
7. First launch only: on the iPhone open **Settings → General → VPN & Device Management**, tap your Apple ID, **Trust**.
8. When the app first talks to a local server, iOS asks to allow **local network** access. Tap Allow, then tap **Try Again** (the first attempt can fail while the prompt is showing).

With a free Apple ID the app stops opening after 7 days; just press Run in Xcode again.

The camera doesn't work in the iOS Simulator. Use **Choose Photo** there, or test on a real phone.

**If the project won't open** (very old or very new Xcode): `brew install xcodegen`, then `cd ios && xcodegen`, and open the regenerated project.

### Sharing with your team: TestFlight

Needs the paid Apple Developer Program (US$99/year).

1. Create the app in App Store Connect with the same bundle identifier.
2. In Xcode: select **Any iOS Device**, then **Product → Archive**.
3. In the Organizer: **Distribute App → App Store Connect → Upload**.
4. In App Store Connect → TestFlight, add testers by email. They install via the TestFlight app.

The app must point at the deployed HTTPS backend for testers.

---

## 3. Measure accuracy on real photos

Don't trust the counts until you've measured them on your own stock.

1. Photograph real inventory and count each photo yourself, carefully.
2. Put the photos in `eval/photos/`.
3. Add a row per photo to `eval/labels.csv` (see `eval/labels.example.csv`). Set `should_warn` to `yes` when the photo is hard and the app ought to flag it.
4. With the backend running:
   ```bash
   python eval/run_eval.py --server http://localhost:8000 --model auto
   ```
5. Results go to `eval/results.csv`: true count, AI count, absolute and % error, confidence, and whether the warning was right. The terminal shows mean/median error, broken down by confidence level.

Cover: separated items, nested chairs, partly hidden chairs, log piles (ends facing you, and from the side), similar colours, poor light, and photos with other objects that shouldn't be counted. Aim for 20+ photos per item type before drawing conclusions.

---

## How the counting works

**Free models (OWLv2, Grounding DINO, OmDet-Turbo)**
- The item name is turned into a singular prompt ("tree logs" → "tree log"). Asked for a plural,
  detectors tend to box the whole pile as one thing.
- Each model returns a box per item it finds. The server drops low-scoring boxes, boxes covering
  most of the photo (the whole pile), duplicates, and boxes that sit inside a bigger box
  (a chair's seat inside the chair).
- The count is the number of boxes left. Each box's centre becomes a dot in the app.
- A single model's confidence is never "high", because it can't see hidden items.

**Stacks (chairs, plates, buckets)**
- In a stack each item hides the one below, so detectors see one object. The stack counter
  instead counts the repeated edge each item shows (one seat lip or rim per item).
- It reads 16 vertical strips through the stack. Only strips at the stack's main spacing vote,
  so backrest slats (closer together) and logos don't add extra counts.
- Deep gaps under each edge are counted; the faint ones above the top item and the floor below
  are ignored, except one for the bottom item.
- **Count one side:** for an exact count, tap *Count one side of the stack* and drag a line down
  the seat fronts. Only the edges crossed by your line are counted.

**Logs, pipes and poles**
- Only roughly square boxes (the cut ends) are counted, not the long sides.
- If a model boxes the whole pile as one or two items, the pile box is replaced by the items
  found inside it.

**Checking and correcting**
- Every counted item gets a numbered dot. Tap a wrong dot to remove it; tap a missed item to add
  one. The card shows "Corrected by you" with the AI's original number and an Undo button.

**Auto**
- Runs the three free models on the same photo (in parallel).
- Shows the middle count and that model's dots. Tap **Details** to see every model's count and
  switch which model's dots are shown.
- Confidence: models within 10% of each other = high, within 30% = medium, otherwise low.
  If any model found nothing, it's never shown as high.

**Claude (optional, paid)**
- Counts section by section through a fixed answer format; the server adds up the sections.
- Returns no positions, so no dots. About US$0.01 per count with Claude Sonnet 5.5.

## Known limitations

- **Accuracy is only measured on a small public sample** (see the table above), not on your stock. Run `eval/run_eval.py` on your own photos.
- **The free models are slow on a plain CPU server** (Auto ≈ 45–60 s per photo on 2 cores).
- **Hidden items can't be counted.** Chairs nested inside a stack, logs behind other logs, boxes at the back of a pallet: the AI only sees the surface. Low confidence and a retake tip are shown, but the number may still be low.
- **Large dense piles** (hundreds of small items) are where vision models make the biggest mistakes. Photograph smaller sections and add them up yourself.
- **One photo per count.** There's no multi-photo merging, so nothing prevents double-counting across photos.
- **Confidence is a rule of thumb** (how well the models agree), checked on only a few photos so far.
- Needs a connection to your server; nothing runs on the phone yet.

## Next steps (in priority order)

1. Run the eval on 50–100 real photos of your main item types, with `model` set to each free model and Auto, and compare.
2. For the one or two item types that matter most (e.g. log ends), train a small custom detector on ~200 labelled photos. It will beat the general models by a wide margin and can run on the iPhone.
3. Add a +/– adjuster and a "Save" button so workers can correct and record the confirmed count.
4. Multi-photo counting for large stock: one photo per section, summed, with a reminder not to overlap.
5. Simple history and CSV export.
