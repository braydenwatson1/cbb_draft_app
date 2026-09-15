"""
CBB Legend Draft & Tournament Simulator
----------------------------------------
Requirements:
    pip install streamlit pandas numpy google-genai

Expects raw, headerless Barttorvik player CSVs named like:
    cbb_players_2010.csv ... cbb_players_2026.csv
in the same directory as this script (glob pattern: cbb_players_20*.csv).

Column order is HARD-CODED below (verified against real 2010 and 2022 rows
by cross-checking ratios like FTM/FTA == FT%, OBPM+DBPM == BPM, etc).
"""

import glob
import random

import numpy as np
import pandas as pd
import streamlit as st

try:
    from google import genai
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False

MODEL_NAME = "gemini-3.6-flash"
CSV_GLOB = "cbb_players_20*.csv*"
HARD_MIN_GP = 8  # baseline sanity floor -- strips tiny-sample garbage rows
CPU_PICK_DELAY = "4s"
BPA_TOP_N = 4
SEARCH_MIN_CHARS = 2
SEARCH_MAX_MATCHES = 40
BPA_RANK_COLS = {
    "Overall": "overall_z",
    "Offense": "obpm",
    "Defense": "dbpm",
}

# ---------------------------------------------------------------------------
# Hard-coded Barttorvik column schema (headerless CSVs, 67 fields)
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "player", "team", "conf", "gp", "min_pct", "ortg", "usg", "efg", "ts",           # 1-9
    "orb_pct", "drb_pct", "ast_pct", "to_pct",                                       # 10-13
    "ftm", "fta", "ft_pct", "twom", "twoa", "two_pct",                               # 14-19
    "threem", "threea", "three_pct",                                                 # 20-22
    "blk_pct", "stl_pct", "ftr",                                                     # 23-25
    "yr", "height", "num",                                                           # 26-28
    "porpag", "adjoe", "rtg", "year", "pid", "hometown", "rec_rank",                 # 29-35
    "ast_tov", "rim_makes", "rim_att", "mid_makes", "mid_att",                       # 36-40
    "rim_pct", "mid_pct", "dunks_made", "dunks_att", "dunk_pct", "draft_pick",       # 41-46
    "drtg", "adrtg", "dporpag", "stops",                                             # 47-50
    "bpm", "obpm", "dbpm", "gbpm", "mpg", "ogbpm", "dgbpm",                          # 51-57
    "oreb_g", "dreb_g", "treb_g", "ast_g", "stl_g", "blk_g", "pts_g",                # 58-64
    "role", "unknown_66", "birthdate",                                              # 65-67
]
assert len(COLUMN_NAMES) == 67

NUMERIC_COLS = [c for c in COLUMN_NAMES if c not in
                ("player", "team", "conf", "yr", "height", "hometown", "role", "birthdate")]

# Stat category -> column used for hidden CPU archetype scoring
CATEGORY_COLUMNS = {
    "scoring": "pts_g",
    "three_pt": "three_pct",
    "rebounding": "treb_g",
    "playmaking": "ast_g",
    "steals": "stl_g",
    "blocks": "blk_g",
    "usage": "usg",
    "efficiency": "bpm",
}

# ---------------------------------------------------------------------------
# Archetypes (kept 100% hidden from the UI during the draft)
# ---------------------------------------------------------------------------
ARCHETYPES = {
    "3-Point / Pace & Space": {"three_pt": 3.0, "scoring": 1.0, "usage": 0.5},
    "Lockdown Defense": {"steals": 2.0, "blocks": 2.0, "rebounding": 1.0},
    "Best Player Available (BPA)": {
        "scoring": 1.0, "rebounding": 1.0, "playmaking": 1.0,
        "steals": 1.0, "blocks": 1.0, "usage": 1.0, "efficiency": 1.0,
    },
    "Post-Up / Inside Grind": {"rebounding": 3.0, "blocks": 1.5, "scoring": 1.0},
    "Playmaking Maestro": {"playmaking": 3.0, "usage": 1.0, "scoring": 0.5},
    "High-Usage Stars": {"usage": 3.0, "scoring": 2.0, "efficiency": 1.0},
    "Balanced Roster Build": {
        "scoring": 1.0, "rebounding": 1.0, "playmaking": 1.0,
        "steals": 1.0, "blocks": 1.0,
    },
    "Glue Guys & Grit": {"steals": 1.5, "efficiency": 1.5, "playmaking": 1.0},
}

FUN_TEAM_ADJ = ["Ironclad", "Rowdy", "Downtown", "Blue-Collar", "Fast-Break",
                "Old-School", "Next-Gen", "Highlight-Reel", "Grit-and-Grind",
                "Hardwood", "Small-Ball", "Paint-Beast", "Sharpshooting",
                "Underdog", "Dynasty", "Wildcard", "Backcourt", "Frontcourt",
                "Buzzer-Beater", "Full-Court", "Triple-Threat", "Clutch",
                "Rebuilding", "Championship", "Cinderella", "Powerhouse",
                "Scrappy", "Elite", "Legacy", "Rising", "Veteran", "Rookie"]

# ---------------------------------------------------------------------------
# Deterministic position mapping (no AI calls -- instant, free, reliable)
# ---------------------------------------------------------------------------
ROLE_POSITIONS = {
    "Pure PG": ["PG", "SG"],
    "Scoring PG": ["PG", "SG"],
    "Combo G": ["SG", "PG", "SF"],
    "Wing G": ["SG", "SF", "PG"],
    "Wing F": ["SF", "PF", "SG"],
    "Stretch 4": ["PF", "SF", "C"],
    "PF/C": ["PF", "C", "SF"],
    "C": ["C", "PF"],
}
DEFAULT_POSITIONS = ["SF", "SG", "PF", "PG", "C"]


def parse_height_inches(h):
    if not isinstance(h, str) or "-" not in h:
        return None
    try:
        feet, inches = h.split("-")
        return int(feet) * 12 + int(inches)
    except (ValueError, TypeError):
        return None


def height_based_positions(h):
    inches = parse_height_inches(h)
    if inches is None:
        return DEFAULT_POSITIONS
    if inches >= 83:
        return ["C", "PF"]
    if inches >= 80:
        return ["PF", "C", "SF"]
    if inches >= 77:
        return ["SF", "PF", "SG"]
    if inches >= 74:
        return ["SG", "SF", "PG"]
    return ["PG", "SG"]


def get_position_prefs(role, height=None):
    if isinstance(role, str) and role in ROLE_POSITIONS:
        return ROLE_POSITIONS[role]
    return height_based_positions(height)


LEGAL_BY_PRIMARY = {
    "PG": {"PG", "SG", "SF"},
    "SG": {"PG", "SG", "SF"},
    "SF": {"SG", "SF", "PF"},
    "PF": {"SF", "PF", "C"},
    "C": {"PF", "C"},
}


def legal_starter_slots(role, height=None):
    """Guards can slide to forwards, forwards can slide to bigs.
    Guards and true bigs do not cross (no C at PG, no PG at C)."""
    prefs = get_position_prefs(role, height)
    primary = prefs[0] if prefs else "SF"
    legal = LEGAL_BY_PRIMARY.get(primary, {"SG", "SF", "PF"})
    return [p for p in ["PG", "SG", "SF", "PF", "C"] if p in legal]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data():
    files = sorted(glob.glob(CSV_GLOB))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(
                f, header=None, names=COLUMN_NAMES,
                quotechar='"', skipinitialspace=True, low_memory=False,
            )
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(), []
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, files


@st.cache_data(show_spinner=False)
def build_master_pool(raw_df):
    """Clean types, apply baseline sanity filter, build display names + ratings."""
    df = raw_df.copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["player"] = df["player"].astype(str).str.strip()
    df = df[df["player"].str.len() > 0]
    df = df[df["player"].str.lower() != "nan"]

    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # baseline sanity floor: strips tiny-sample rows with unreliable/NaN-heavy
    # stats BEFORE we ever compute z-scores, so they can't skew the population
    df = df[df["gp"].fillna(0) >= HARD_MIN_GP].copy()

    df["min_pct_clean"] = df["min_pct"].clip(lower=0, upper=100)
    df["team_strength"] = df.groupby(["team", "year"])["porpag"].transform("mean")

    # One row per exact player name: keep the last season they played.
    # If they appear twice in that last year, keep the higher-GP row.
    df = df.sort_values(["player", "year", "gp"], ascending=[True, False, False])
    df = df.drop_duplicates(subset=["player"], keep="first").copy()
    df = df.reset_index(drop=True)

    prefs = df.apply(lambda r: get_position_prefs(r["role"], r["height"]), axis=1)
    df["primary_pos"] = prefs.map(lambda p: p[0])
    df["pos_label"] = prefs.map(lambda p: "/".join(p))
    df["display_name"] = (
        df["pos_label"] + " " + df["player"] + " (" + df["year"].astype(str) + ")"
    )
    df["short_name"] = df["player"] + " (" + df["year"].astype(str) + ")"

    # category z-scores, used ONLY for hidden CPU archetype-driven picks
    z_cols = []
    for cat, col in CATEGORY_COLUMNS.items():
        if col in df.columns:
            series = df[col]
            mean, std = series.mean(), series.std()
            df[f"z_{cat}"] = ((series - mean) / std).fillna(0.0) if std and std > 0 else 0.0
            z_cols.append(f"z_{cat}")

    # overall rating: driven by Barttorvik's own composite metrics (bpm, porpag)
    # rather than an unweighted average of noisy per-game counting stats
    for col in ["bpm", "porpag", "team_strength"]:
        series = df[col]
        mean, std = series.mean(), series.std()
        df[f"z_{col}"] = ((series - mean) / std).fillna(0.0) if std and std > 0 else 0.0
    df["overall_z"] = df[["z_bpm", "z_porpag"]].mean(axis=1)

    # NBA draft capital: pick 1 >> pick 60 >> undrafted. log(61/pick) keeps lottery
    # talent loud without rewriting the rest of the rating stack.
    pick = pd.to_numeric(df["draft_pick"], errors="coerce")
    draft_raw = pd.Series(0.0, index=df.index)
    drafted = pick.notna() & (pick >= 1)
    draft_raw.loc[drafted] = np.log(61.0 / pick.loc[drafted].clip(upper=60.0))
    dmean, dstd = draft_raw.mean(), draft_raw.std()
    df["z_draft"] = ((draft_raw - dmean) / dstd).fillna(0.0) if dstd and dstd > 0 else 0.0

    meta = {"z_cols": z_cols}
    return df, meta


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "stage": "setup",
        "teams": {},
        "available_pool": [],
        "pick_order": [],
        "current_pick": 0,
        "draft_history": [],
        "num_teams": 10,
        "roster_size": 8,
        "user_team_idx": 0,
        "pool_df": None,
        "tournament_result": None,
        "tournament_bracket": None,
        "api_key_input": "",
        "selected_slot": None,
        "pending_player_id": None,
        "cpu_delay_ready": False,
        "scroll_to_top": False,
        "pick_select_nonce": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_draft():
    for key in ["stage", "teams", "available_pool", "pick_order", "current_pick",
                "draft_history", "pool_df", "tournament_result", "tournament_bracket",
                "selected_slot", "pending_player_id", "cpu_delay_ready", "scroll_to_top",
                "pick_select_nonce", "draft_search", "bpa_mode"]:
        if key in st.session_state:
            del st.session_state[key]
    init_state()


# ---------------------------------------------------------------------------
# Roster slot helpers
# ---------------------------------------------------------------------------
def player_short_name(pool_df, pid, empty="—"):
    if pid is None:
        return empty
    if "short_name" in pool_df.columns:
        return pool_df.loc[pid, "short_name"]
    return pool_df.loc[pid, "display_name"]


def get_slot_value(team, slot_key):
    if slot_key in team["slots"]:
        return team["slots"][slot_key]
    idx = int(slot_key.replace("BENCH", "")) - 1
    return team["bench"][idx]


def set_slot_value(team, slot_key, value):
    if slot_key in team["slots"]:
        team["slots"][slot_key] = value
    else:
        idx = int(slot_key.replace("BENCH", "")) - 1
        team["bench"][idx] = value


def assign_player_to_roster(team_idx, player_id):
    """Place on preferred legal slot first, then any legal starter, then bench.
    Joke teams ignore the guard/big split on purpose."""
    team = st.session_state.teams[team_idx]
    pool_df = st.session_state.pool_df
    role = pool_df.loc[player_id, "role"]
    height = pool_df.loc[player_id, "height"]
    prefs = get_position_prefs(role, height)
    joke = bool(team.get("is_joke"))

    if joke:
        for pos in ["PG", "SG", "SF", "PF", "C"]:
            if team["slots"][pos] is None:
                team["slots"][pos] = player_id
                return pos
        for i in range(len(team["bench"])):
            if team["bench"][i] is None:
                team["bench"][i] = player_id
                return f"Bench {i + 1}"
        return "Overflow"

    legal = legal_starter_slots(role, height)
    for pos in prefs:
        if pos in legal and team["slots"][pos] is None:
            team["slots"][pos] = player_id
            return pos
    for pos in legal:
        if team["slots"][pos] is None:
            team["slots"][pos] = player_id
            return pos
    for i in range(len(team["bench"])):
        if team["bench"][i] is None:
            team["bench"][i] = player_id
            return f"Bench {i + 1}"
    return "Overflow"


def roster_all_ids(team_idx):
    team = st.session_state.teams[team_idx]
    ids = [v for v in team["slots"].values() if v is not None]
    ids += [v for v in team["bench"] if v is not None]
    return ids


def get_open_slots_summary(team_idx):
    team = st.session_state.teams[team_idx]
    parts = [pos for pos in ["PG", "SG", "SF", "PF", "C"] if team["slots"][pos] is None]
    open_bench = sum(1 for b in team["bench"] if b is None)
    if open_bench:
        parts.append(f"Bench x{open_bench}")
    return parts


# ---------------------------------------------------------------------------
# Draft logic
# ---------------------------------------------------------------------------
def build_pick_order(num_teams, rounds):
    order = []
    for rnd in range(rounds):
        team_order = list(range(num_teams)) if rnd % 2 == 0 else list(range(num_teams - 1, -1, -1))
        order.extend(team_order)
    return order


def cpu_make_pick(team_idx):
    pool_df = st.session_state.pool_df
    available = st.session_state.available_pool
    if not available:
        return None
    team = st.session_state.teams[team_idx]
    sub = pool_df.loc[available]
    open_starters = [pos for pos in ["PG", "SG", "SF", "PF", "C"] if team["slots"][pos] is None]
    first_open = open_starters[0] if open_starters else None

    if team.get("is_joke"):
        score = 0.35 * sub["overall_z"].fillna(0.0)
        mismatch = pd.Series(0.0, index=sub.index)
        if first_open:
            prim = sub["primary_pos"].astype(str)
            for pos_key, legal in LEGAL_BY_PRIMARY.items():
                mask = prim == pos_key
                if first_open not in legal:
                    mismatch.loc[mask] = 1.0
        score = score + 2.8 * mismatch
        score = score + np.random.normal(0, 1.4, size=len(score))
        return score.idxmax()

    archetype = team["archetype"]
    weights = ARCHETYPES[archetype]
    score = pd.Series(0.0, index=sub.index)
    matched = False
    for cat, w in weights.items():
        col = f"z_{cat}"
        if col in sub.columns:
            score = score + w * sub[col].fillna(0.0)
            matched = True
    if not matched:
        score = sub["overall_z"].fillna(0.0)

    # Extra talent / pedigree / winning-program juice on top of the archetype.
    score = score + 2.2 * sub["z_porpag"].fillna(0.0)
    if "z_draft" in sub.columns:
        score = score + 1.6 * sub["z_draft"].fillna(0.0)
    if "z_team_strength" in sub.columns:
        score = score + 0.8 * sub["z_team_strength"].fillna(0.0)

    fit = pd.Series(0.0, index=sub.index)
    open_set = set(open_starters)
    prim = sub["primary_pos"].astype(str)
    for pos_key, legal in LEGAL_BY_PRIMARY.items():
        mask = prim == pos_key
        if not mask.any():
            continue
        open_legal = legal & open_set
        if pos_key in open_legal:
            fit.loc[mask] = 2.2
        elif open_legal:
            fit.loc[mask] = 1.4
        elif open_starters:
            fit.loc[mask] = -2.5
    score = score + fit
    noise = np.random.normal(0, 0.45, size=len(score))
    score = score + noise
    return score.idxmax()


def commit_pick(team_idx, player_id, *, from_user=False):
    player_id = int(player_id)
    num_teams = st.session_state.num_teams
    slot = assign_player_to_roster(team_idx, player_id)
    st.session_state.available_pool.remove(player_id)
    rnd = st.session_state.current_pick // num_teams + 1
    st.session_state.draft_history.append({
        "pick": st.session_state.current_pick + 1,
        "round": rnd,
        "team": st.session_state.teams[team_idx]["name"],
        "player": st.session_state.pool_df.loc[player_id, "display_name"],
        "slot": slot,
    })
    st.session_state.current_pick += 1
    st.session_state.pending_player_id = None
    st.session_state.cpu_delay_ready = False
    if from_user:
        st.session_state.scroll_to_top = True


def cpu_is_on_clock():
    current = st.session_state.current_pick
    order = st.session_state.pick_order
    if current >= len(order):
        return False
    return order[current] != st.session_state.user_team_idx


@st.fragment(run_every=CPU_PICK_DELAY)
def cpu_pick_ticker():
    """Wait one interval, then take a single CPU pick so the board updates live."""
    if not cpu_is_on_clock():
        return
    if not st.session_state.cpu_delay_ready:
        st.session_state.cpu_delay_ready = True
        return
    team_idx = st.session_state.pick_order[st.session_state.current_pick]
    pid = cpu_make_pick(team_idx)
    if pid is None:
        return
    commit_pick(team_idx, pid)
    st.rerun()


def queue_player_for_draft(player_id):
    st.session_state.pending_player_id = player_id
    st.session_state.draft_search = ""
    st.session_state.pick_select_nonce = st.session_state.get("pick_select_nonce", 0) + 1


def scroll_draft_to_top():
    st.html(
        """
        <script>
        const roots = [document, window.parent && window.parent.document].filter(Boolean);
        for (const doc of roots) {
          const nodes = [
            doc.querySelector('[data-testid="stMain"]'),
            doc.querySelector('section.main'),
            doc.querySelector('.stApp'),
            doc.scrollingElement,
          ];
          for (const el of nodes) {
            if (el && typeof el.scrollTo === "function") {
              el.scrollTo({top: 0, behavior: "smooth"});
            }
          }
        }
        window.scrollTo({top: 0, behavior: "smooth"});
        </script>
        """,
        unsafe_allow_javascript=True,
        width="content",
    )


def search_available_players(query):
    pool_df = st.session_state.pool_df
    available = st.session_state.available_pool
    q = (query or "").strip().lower()
    if len(q) < SEARCH_MIN_CHARS:
        return []
    sub = pool_df.loc[available]
    hit = sub["player"].astype(str).str.lower().str.contains(q, regex=False)
    matches = sub[hit].sort_values("player")
    return list(matches.index[:SEARCH_MAX_MATCHES])


def best_available_by_position(rank_col, n=BPA_TOP_N):
    pool_df = st.session_state.pool_df
    available = st.session_state.available_pool
    sub = pool_df.loc[available]
    ranked = {}
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        pos_df = sub[sub["primary_pos"] == pos]
        if pos_df.empty or rank_col not in pos_df.columns:
            ranked[pos] = []
            continue
        top = pos_df.sort_values(rank_col, ascending=False, na_position="last").head(n)
        ranked[pos] = list(top.index)
    return ranked


def team_rating(team_idx):
    ids = roster_all_ids(team_idx)
    if not ids:
        return 0.0
    return float(st.session_state.pool_df.loc[ids, "overall_z"].mean())


# ---------------------------------------------------------------------------
# Tournament helpers
# ---------------------------------------------------------------------------
def seed_positions(n):
    seeds = [1]
    while len(seeds) < n:
        m = len(seeds) * 2
        new_seeds = []
        for s in seeds:
            new_seeds.append(s)
            new_seeds.append(m + 1 - s)
        seeds = new_seeds
    return seeds


def build_bracket():
    num_teams = st.session_state.num_teams
    ranked = sorted(range(num_teams), key=lambda t: team_rating(t), reverse=True)
    size = 1
    while size < num_teams:
        size *= 2
    slots = ranked + [None] * (size - num_teams)
    order = seed_positions(size)
    bracket = [slots[s - 1] if s - 1 < len(slots) else None for s in order]
    pairs = [(bracket[i], bracket[i + 1]) for i in range(0, len(bracket), 2)]
    return ranked, pairs


def build_tournament_prompt():
    ranked, pairs = build_bracket()
    st.session_state.tournament_bracket = pairs
    pool_df = st.session_state.pool_df

    lines = []
    lines.append(
        "You are the color commentator and simulation engine for a single-elimination "
        "college basketball legends tournament. Simulate the ENTIRE bracket yourself, "
        "round by round, deciding realistic winners based on the rosters, ratings, and "
        "team strategies described below (add some randomness/upsets for realism, but "
        "generally respect team strength). Output in Markdown."
    )
    lines.append("\n## Teams (strongest to weakest by rating)\n")
    for t in ranked:
        team = st.session_state.teams[t]
        rating = team_rating(t)
        roster_names = [pool_df.loc[pid, "display_name"] for pid in roster_all_ids(t)]
        tag = " (USER-DRAFTED TEAM)" if t == st.session_state.user_team_idx else ""
        joke = " — this GM is secretly a chaos/meme drafter (funny out-of-position and questionable picks)" if team.get("is_joke") else ""
        lines.append(
            f"- **{team['name']}{tag}** — Strategy: *{team['archetype']}*{joke} — "
            f"Overall Rating: {rating:.2f}\n  Roster: {', '.join(roster_names)}"
        )

    lines.append("\n## Round 1 Matchups (higher seed had a bye if opponent is 'BYE')\n")
    for a, b in pairs:
        a_name = st.session_state.teams[a]["name"] if a is not None else "BYE"
        b_name = st.session_state.teams[b]["name"] if b is not None else "BYE"
        lines.append(f"- {a_name} vs {b_name}")

    worst_team = min(range(st.session_state.num_teams), key=lambda t: team_rating(t))
    worst_name = st.session_state.teams[worst_team]["name"]

    lines.append(
        "\n## Instructions for your response\n"
        "1. Simulate every round match-by-match with a final score for each game and a "
        "2-3 sentence recap that references each team's hidden strategy now that the "
        "draft is over (explain *why* their archetype won or lost that matchup).\n"
        "2. Use clear Markdown headers for each round (## Round of X, ## Semifinals, ## Championship).\n"
        "3. Advance byes automatically to the next round without a 'game'.\n"
        "4. At the end, crown the champion with a fun trophy section (## \U0001F3C6 Champion).\n"
        f"5. Finish with a savage but good-natured roast of **{worst_name}**'s roster "
        "construction and strategy in a section titled '## \U0001F525 Worst Roster Roast'."
    )
    return "\n".join(lines)


def run_tournament_simulation(api_key):
    if not GENAI_SDK_AVAILABLE:
        return None, "The `google-genai` package isn't installed. Run `pip install google-genai`."
    if not api_key:
        return None, "Please enter your Google API key first."
    try:
        client = genai.Client(api_key=api_key)
        prompt = build_tournament_prompt()
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        text = getattr(response, "text", None) or str(response)
        return text, None
    except Exception as e:
        return None, f"Error calling Gemini API: {e}"


def inject_layout_styles():
    st.html(
        """
        <style>
        section[data-testid="stSidebar"] { display: none !important; }
        div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
        div.stButton > button {
          white-space: normal !important;
          height: auto !important;
          min-height: 2.6rem;
          text-align: left !important;
          line-height: 1.25 !important;
          text-overflow: unset !important;
          overflow: visible !important;
        }
        div.stButton > button p {
          white-space: normal !important;
          overflow: visible !important;
          text-overflow: unset !important;
        }
        </style>
        """
    )


def render_new_draft_control():
    if st.button("New draft", key="new_draft_top"):
        reset_draft()
        st.rerun()


def render_setup():
    st.title("College basketball legend draft")
    st.caption("Set the league, trim the historical pool, then draft against the CPU.")

    raw_df, files = load_raw_data()
    if raw_df.empty:
        st.error(
            f"No CSV files matching `{CSV_GLOB}` were found in the app's root directory. "
            "Add files like `cbb_players_2024.csv` and reload."
        )
        st.stop()

    pool_df, meta = build_master_pool(raw_df)
    all_years = sorted(pool_df["year"].unique().tolist())
    year_lo, year_hi = int(all_years[0]), int(all_years[-1])

    st.badge(
        f"{len(pool_df):,} players loaded from {len(files)} seasons",
        icon=":material/check_circle:",
        color="green",
    )
    st.space("small")

    with st.container(border=True):
        st.markdown("##### :material/groups: League")
        c1, c2, c3 = st.columns(3)
        with c1:
            num_teams = st.slider(
                "Teams", 2, 32, st.session_state.get("num_teams", 10)
            )
        with c2:
            roster_size = st.slider(
                "Roster size", 5, 10, st.session_state.get("roster_size", 8)
            )
        with c3:
            if "draft_position" in st.session_state and st.session_state.draft_position > num_teams:
                st.session_state.draft_position = num_teams
            draft_position = st.slider(
                "Your draft slot",
                1,
                num_teams,
                min(st.session_state.get("draft_position", 1), num_teams),
                key="draft_position",
            )

    st.space("small")

    with st.container(border=True):
        st.markdown("##### :material/filter_list: Player pool")
        left, right = st.columns(2, gap="large")
        with left:
            season_mode = st.segmented_control(
                "Seasons",
                options=["All years", "Last 5", "Last 10", "Custom range"],
                default="All years",
                required=True,
                key="season_mode",
            )
            if season_mode == "Last 5":
                years_selected = all_years[-5:]
            elif season_mode == "Last 10":
                years_selected = all_years[-10:]
            elif season_mode == "Custom range":
                start_year, end_year = st.slider(
                    "Season range",
                    min_value=year_lo,
                    max_value=year_hi,
                    value=(year_lo, year_hi),
                    key="season_range",
                )
                years_selected = [y for y in all_years if start_year <= y <= end_year]
            else:
                years_selected = all_years
            if years_selected:
                st.caption(f"{years_selected[0]}–{years_selected[-1]} · {len(years_selected)} season(s)")
            else:
                st.caption("No seasons in this range.")
        with right:
            min_minutes = st.slider(
                "Minimum minutes %",
                0,
                100,
                30,
                key="min_minutes",
            )
            st.caption(
                "Barttorvik **Min%** is the share of team minutes a player used that season. "
                "A 30% floor drops tiny-sample benches and walk-ons so the pool is mostly rotation players."
            )

    preview = pool_df[pool_df["year"].isin(years_selected)]
    if min_minutes > 0:
        preview = preview[preview["min_pct_clean"].fillna(0) >= min_minutes]
    needed = num_teams * roster_size
    season_label = (
        f"{years_selected[0]}–{years_selected[-1]}" if years_selected else "—"
    )

    st.space("small")
    m1, m2, m3 = st.columns(3)
    m1.metric("Players in pool", f"{len(preview):,}", border=True, icon=":material/person:")
    m2.metric("Spots to fill", f"{needed}", border=True, icon=":material/grid_view:")
    m3.metric("Seasons", season_label, border=True, icon=":material/calendar_month:")

    if len(preview) < needed:
        st.warning(
            f"Need at least {needed} players ({num_teams} teams × {roster_size} spots) "
            f"but only {len(preview):,} qualify. Loosen minutes or add seasons."
        )

    st.space("small")
    start_clicked = st.button(
        "Start draft",
        type="primary",
        width="stretch",
        icon=":material/play_arrow:",
    )

    if start_clicked:
        filtered = pool_df[pool_df["year"].isin(years_selected)].copy()
        if min_minutes > 0:
            filtered = filtered[filtered["min_pct_clean"].fillna(0) >= min_minutes]
        filtered = filtered.reset_index(drop=True)

        needed = num_teams * roster_size
        if len(filtered) < needed:
            st.error("Not enough players under these filters to fill every roster. Please adjust and try again.")
            st.stop()

        st.session_state.pool_df = filtered
        st.session_state.num_teams = num_teams
        st.session_state.roster_size = roster_size
        st.session_state.user_team_idx = draft_position - 1
        st.session_state.available_pool = [int(i) for i in filtered.index]
        st.session_state.pick_order = build_pick_order(num_teams, roster_size)
        st.session_state.current_pick = 0
        st.session_state.draft_history = []
        st.session_state.selected_slot = None

        bench_size = roster_size - 5
        teams = {}
        used_names = set()
        joke_idx = None
        if num_teams >= 4:
            cpu_ids = [i for i in range(num_teams) if i != st.session_state.user_team_idx]
            if cpu_ids:
                joke_idx = random.choice(cpu_ids)
        for i in range(num_teams):
            is_joke = i == joke_idx
            if i == st.session_state.user_team_idx:
                name = f"Team {i + 1} (You)"
                archetype = random.choice(list(ARCHETYPES.keys()))
            elif is_joke:
                while True:
                    candidate = f"{random.choice(FUN_TEAM_ADJ)} Team {i + 1}"
                    if candidate not in used_names:
                        used_names.add(candidate)
                        break
                name = candidate
                archetype = "Chaos / Meme Build"
            else:
                while True:
                    candidate = f"{random.choice(FUN_TEAM_ADJ)} Team {i + 1}"
                    if candidate not in used_names:
                        used_names.add(candidate)
                        break
                name = candidate
                archetype = random.choice(list(ARCHETYPES.keys()))
            teams[i] = {
                "name": name,
                "archetype": archetype,
                "is_joke": is_joke,
                "slots": {"PG": None, "SG": None, "SF": None, "PF": None, "C": None},
                "bench": [None] * bench_size,
            }
        st.session_state.teams = teams
        st.session_state.stage = "draft"
        st.rerun()


# ---------------------------------------------------------------------------
# UI: League-wide roster grid (read-only, always visible)
# ---------------------------------------------------------------------------
def render_league_grid():
    pool_df = st.session_state.pool_df
    bench_size = st.session_state.roster_size - 5
    row_labels = ["PG", "SG", "SF", "PF", "C"] + [f"Bench {i + 1}" for i in range(bench_size)]

    data = {}
    for t_idx, team in st.session_state.teams.items():
        values = []
        for pos in ["PG", "SG", "SF", "PF", "C"]:
            pid = team["slots"][pos]
            values.append(player_short_name(pool_df, pid))
        for b in team["bench"]:
            values.append(player_short_name(pool_df, b))
        data[team["name"]] = values

    grid_df = pd.DataFrame(data, index=row_labels)
    st.dataframe(
        grid_df,
        width="stretch",
        height=min(480, 56 + 42 * len(row_labels)),
        row_height=40,
    )


# ---------------------------------------------------------------------------
# UI: Interactive click-to-swap roster (user's team only)
# ---------------------------------------------------------------------------
def handle_slot_click(team, slot_key):
    sel = st.session_state.selected_slot
    if sel is None:
        st.session_state.selected_slot = slot_key
    elif sel == slot_key:
        st.session_state.selected_slot = None
    else:
        a = get_slot_value(team, sel)
        b = get_slot_value(team, slot_key)
        set_slot_value(team, sel, b)
        set_slot_value(team, slot_key, a)
        st.session_state.selected_slot = None


def render_slot_button(team, slot_key, caption):
    pool_df = st.session_state.pool_df
    val = get_slot_value(team, slot_key)
    label = player_short_name(pool_df, val, empty="— Empty —")
    full = pool_df.loc[val, "display_name"] if val is not None else None
    st.caption(f"**{caption}**")
    is_selected = st.session_state.selected_slot == slot_key
    btn_type = "primary" if is_selected else "secondary"
    if st.button(label, key=f"slotbtn_{slot_key}", type=btn_type, width="stretch", help=full):
        handle_slot_click(team, slot_key)
        st.rerun()


def render_interactive_roster(team_idx):
    team = st.session_state.teams[team_idx]
    bench_size = st.session_state.roster_size - 5

    st.markdown("##### Starters")
    st.caption("Click a slot, then click another to swap players between them.")
    cols = st.columns(5)
    for col, pos in zip(cols, ["PG", "SG", "SF", "PF", "C"]):
        with col:
            render_slot_button(team, pos, pos)

    if bench_size > 0:
        st.markdown("##### Bench")
        bcols = st.columns(bench_size)
        for i in range(bench_size):
            with bcols[i]:
                render_slot_button(team, f"BENCH{i + 1}", f"Bench {i + 1}")


def render_bpa_board():
    pool_df = st.session_state.pool_df
    pending = st.session_state.get("pending_player_id")
    st.markdown("#### Best available")
    st.caption("Click a player to load them, then confirm with Draft selected player. Each player appears at their primary position.")
    mode = st.segmented_control(
        "Rank by",
        options=list(BPA_RANK_COLS.keys()),
        default="Overall",
        required=True,
        key="bpa_mode",
    )
    if mode is None:
        mode = "Overall"
    rank_col = BPA_RANK_COLS[mode]
    board = best_available_by_position(rank_col)
    cols = st.columns(5)
    for col, pos in zip(cols, ["PG", "SG", "SF", "PF", "C"]):
        with col:
            st.markdown(f"**{pos}**")
            pids = board[pos]
            if not pids:
                st.caption("No players left")
                continue
            for pid in pids:
                label = pool_df.loc[pid, "display_name"]
                is_pending = pending == pid
                st.button(
                    label,
                    key=f"bpa_{mode}_{pos}_{pid}",
                    type="primary" if is_pending else "secondary",
                    width="stretch",
                    help=label,
                    on_click=queue_player_for_draft,
                    args=(int(pid),),
                )


def render_user_pick_controls(team_on_clock):
    pool_df = st.session_state.pool_df
    available_ids = st.session_state.available_pool
    pending = st.session_state.get("pending_player_id")
    if pending is not None and pending not in available_ids:
        pending = None
        st.session_state.pending_player_id = None

    st.text_input(
        "Search players",
        placeholder="Type a player name…",
        key="draft_search",
    )
    query = st.session_state.get("draft_search", "")
    matches = search_available_players(query)
    nonce = st.session_state.get("pick_select_nonce", 0)
    selected_id = None
    if len((query or "").strip()) < SEARCH_MIN_CHARS:
        st.caption("Type at least 2 letters, or load someone from the board below.")
    elif not matches:
        st.caption("No matches. Try a different name.")
    else:
        selected_id = st.selectbox(
            "Select a player to draft",
            options=matches,
            index=None,
            placeholder="Choose a match…",
            format_func=lambda i: pool_df.loc[i, "display_name"],
            key=f"pick_select_{st.session_state.current_pick}_{nonce}",
        )
        if selected_id is not None:
            pending = selected_id
            st.session_state.pending_player_id = selected_id

    if pending is not None:
        st.info(f"Ready to draft: **{pool_df.loc[pending, 'display_name']}**")
    draft_clicked = st.button(
        "Draft selected player",
        type="primary",
        width="stretch",
        disabled=pending is None,
    )
    if draft_clicked and pending is not None:
        commit_pick(team_on_clock, pending, from_user=True)
        st.rerun()


def render_draft():
    total_picks = len(st.session_state.pick_order)
    current_pick = st.session_state.current_pick

    if current_pick >= total_picks:
        st.session_state.stage = "finished"
        st.rerun()
        return

    if st.session_state.get("scroll_to_top"):
        scroll_draft_to_top()
        st.session_state.scroll_to_top = False

    num_teams = st.session_state.num_teams
    team_on_clock = st.session_state.pick_order[current_pick]
    round_num = current_pick // num_teams + 1
    pick_in_round = current_pick % num_teams + 1
    user_idx = st.session_state.user_team_idx
    title_col, action_col = st.columns([6, 1], vertical_alignment="bottom")
    with title_col:
        st.title("\U0001F3C0 Draft Room")
    with action_col:
        render_new_draft_control()
    st.progress(current_pick / total_picks, text=f"Pick {current_pick + 1} of {total_picks}")
    st.subheader(
        f"Round {round_num}, Pick {pick_in_round} — On the Clock: "
        f"{st.session_state.teams[team_on_clock]['name']}"
    )

    if st.session_state.draft_history:
        last = st.session_state.draft_history[-1]
        st.success(f"Latest pick: {last['team']} selected {last['player']} ({last['slot']})")

    st.markdown("#### \U0001F3E2 League Rosters (live)")
    render_league_grid()

    st.divider()
    left, right = st.columns([1, 1.3])

    with left:
        if team_on_clock == user_idx:
            open_slots = get_open_slots_summary(user_idx)
            if open_slots:
                st.info(f"**You still need:** {', '.join(open_slots)}")
            render_user_pick_controls(team_on_clock)
        else:
            st.info(f"{st.session_state.teams[team_on_clock]['name']} is on the clock…")
            st.caption("CPU picks appear about every 4 seconds so you can watch the board.")

    with right:
        st.markdown("#### \U0001F465 Your Roster")
        render_interactive_roster(user_idx)

    st.divider()
    if team_on_clock == user_idx:
        render_bpa_board()

    st.divider()
    with st.expander("Recent draft history", expanded=True):
        if st.session_state.draft_history:
            hist_df = pd.DataFrame(st.session_state.draft_history[-15:][::-1])
            hist_df = hist_df.rename(columns={
                "pick": "Pick #", "round": "Round", "team": "Team",
                "player": "Player", "slot": "Slot"
            })
            st.dataframe(hist_df, width="stretch", hide_index=True)
        else:
            st.caption("No picks yet.")

    if cpu_is_on_clock():
        cpu_pick_ticker()


# ---------------------------------------------------------------------------
# UI: Finished / Tournament screen
# ---------------------------------------------------------------------------
def render_finished():
    title_col, action_col = st.columns([6, 1], vertical_alignment="bottom")
    with title_col:
        st.title("\U0001F3C1 Draft Complete!")
    with action_col:
        render_new_draft_control()
    pool_df = st.session_state.pool_df
    user_idx = st.session_state.user_team_idx

    ratings = {t: team_rating(t) for t in st.session_state.teams}
    ranked_teams = sorted(ratings.keys(), key=lambda t: ratings[t], reverse=True)

    st.markdown("#### \U0001F3E2 Final League Rosters")
    render_league_grid()

    st.divider()
    st.markdown("### Revealed Strategies")
    for t in ranked_teams:
        team = st.session_state.teams[t]
        tag = " \U0001F31F (You)" if t == user_idx else ""
        joke_tag = " — secret chaos GM" if team.get("is_joke") else ""
        with st.expander(f"{team['name']}{tag} — Rating {ratings[t]:.2f} — Strategy: {team['archetype']}{joke_tag}"):
            for pos in ["PG", "SG", "SF", "PF", "C"]:
                pid = team["slots"][pos]
                name = pool_df.loc[pid, "display_name"] if pid is not None else "— Empty —"
                st.write(f"**{pos}:** {name}")
            for i, pid in enumerate(team["bench"]):
                name = pool_df.loc[pid, "display_name"] if pid is not None else "— Empty —"
                st.write(f"**Bench {i + 1}:** {name}")

    st.divider()
    st.markdown("### \U0001F3C6 AI Tournament Simulation")

    api_key = st.text_input(
        "Google API Key (for Gemini tournament sim)",
        type="password",
        value=st.session_state.get("api_key_input", ""),
        key="api_key_input",
    )

    if not GENAI_SDK_AVAILABLE:
        st.warning("Install the SDK to enable this feature: `pip install google-genai`")

    run_clicked = st.button("\U0001F3AE Run AI Tournament Simulation", type="primary")
    if run_clicked:
        with st.spinner("Simulating the bracket with Gemini..."):
            text, error = run_tournament_simulation(api_key)
        if error:
            st.error(error)
        else:
            st.session_state.tournament_result = text

    if st.session_state.tournament_result:
        st.markdown(st.session_state.tournament_result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="CBB Legend Draft",
        page_icon="\U0001F3C0",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_state()
    inject_layout_styles()

    stage = st.session_state.stage
    if stage == "setup":
        render_setup()
    elif stage == "draft":
        render_draft()
    elif stage == "finished":
        render_finished()
    else:
        st.session_state.stage = "setup"
        st.rerun()


if __name__ == "__main__":
    main()