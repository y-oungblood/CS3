"""All tunable numbers for the recommender."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"
RESULTS_DIR = ROOT / "results"

# Artifacts
MIN_RATINGS = 50          # minimum ratings for a movie to be kept
K_NEIGHBORS = 50          # neighbors stored per movie
TOP_TAGS = 3              # tags kept per movie for explanations
EVAL_TEST_FRACTION = 0.2  # share of users held out in eval mode
SEED = 42

# Scoring
DAMPING = 5.0          # tuned on validation users (seed 7); see results/tuning
K_SHRINK = 100         # strong pull to the global mean; seeds would otherwise inflate it
W_INTERESTED = 0.3
CANDIDATE_POOL = 200
FALLBACK_POPULAR = 1000
MIN_POSITIVE_CANDIDATES = 10

# Adventurousness dial
LAMBDA_MAX = 0.6

# Card selection
ELICIT_POOL = 2000
PHASE1_ANSWERS = 10
WILDCARD_RATE = 0.3
AB_TEST = True

# UX
GATE_ANSWERS = 10
CHECKINS = (10, 25)
PROGRESS_FULL = 25

# Simulation
N_SIM_USERS = 500
SIM_SWIPES = 50

# When to watch
SHORT_RUNTIME = 100
LONG_RUNTIME = 130
