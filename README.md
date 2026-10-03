# Hello!

I asked claude to make this readme file if you would like to duplicate this locally. As I discussed in my report, the front end and live DB parts of the project were written using/alongside claude, while the algorithms, data handling, basically the DAEN stuff was me.

You will also notice that this repo is 300,000 lines long. That scared me too. It's mostly from the CSV files, which I chose to include from my results if anyone wants to see them.


# Movie Swipe

A movie recommender built on MovieLens 32M and LensKit. A new user picks a few favorite movies, then swipes through cards ("Seen it" with a 1–5 star rating, or "Haven't seen" with Add to my list / Not interested), and gets explained recommendations with a when-to-watch suggestion.

## Requirements

- Python 3.12.5 or newer (tested on 3.14)
- About 2 GB of free disk space for the MovieLens 32M download

## Quick start: run the app

The built artifacts in `data/artifacts/deploy/` are committed, so the app runs without downloading MovieLens.

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app/main.py --web           # serves at http://localhost:8550
```

Answers are saved to a local SQLite file, `data/app.db`.

## Full reproduction

### 1. Install the offline tools (LensKit, plotting, tests)

```bash
pip install -r requirements-offline.txt
python -m pytest tests             # 95 tests
```

### 2. Download MovieLens 32M

```bash
mkdir -p data/raw && cd data/raw
curl -LO https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip && cd ../..
```

### 3. Build the artifacts

```bash
python offline/build_artifacts.py --data data/raw/ml-32m --mode deploy   # used by the app
python offline/build_artifacts.py --data data/raw/ml-32m --mode eval     # used by the simulation (20% of users held out)
```

### 4. Add posters and runtimes (optional)

Needs a free TMDB API key from themoviedb.org. Rebuilt artifacts have no posters until this runs.

```bash
export TMDB_API_KEY=<your key>     # Windows PowerShell: $env:TMDB_API_KEY = "<your key>"
python offline/fetch_posters.py
```

### 5. Run the simulation

```bash
# Main results: 500 held-out users, written to results/ml-32m/
python offline/simulate.py --data data/raw/ml-32m --artifacts data/artifacts/eval

# The two rejected evaluation designs, for comparison
python offline/simulate.py --data data/raw/ml-32m --artifacts data/artifacts/eval --test-cards reveal --out results/ml-32m-reveal
python offline/simulate.py --data data/raw/ml-32m --artifacts data/artifacts/eval --test-cards exclude --out results/ml-32m-exclude

# One weight-tuning run on separate validation users (see results/tuning/ for all runs)
python offline/simulate.py --data data/raw/ml-32m --artifacts data/artifacts/eval --seed 7 --users 300 --set DAMPING=5 --set K_SHRINK=100 --out results/tuning/d5-k100
```

### 6. Analyze the real-user data

The anonymized tester data is committed in `results/real-users/snapshot/`, so this needs no database access:

```bash
python offline/analyze_real_users.py       # writes summary.json and figures to results/real-users/
```

## Deploying with Firestore

The deployed app stores data in Google Cloud Firestore instead of SQLite. Create a Firebase project with a Firestore database, download a service-account key, and keep it outside the repo.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
python offline/check_firestore.py          # checks every storage operation, then deletes its test data

STORAGE_BACKEND=firestore python app/main.py --web
```

The included `Dockerfile` and `render.yaml` deploy the app to Render's free tier. The key goes in the `FIREBASE_CREDENTIALS` secret (the key file's contents, not its path). To run the container locally:

```bash
docker build -t movie-swipe .
docker run -p 10000:10000 -e FIREBASE_CREDENTIALS="$(cat /path/to/key.json)" movie-swipe
```

## Data credits

Ratings data from MovieLens 32M (GroupLens Research, University of Minnesota). Posters and runtimes from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.
