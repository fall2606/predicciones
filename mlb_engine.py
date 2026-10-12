"""MLB/Kalshi forecasts and append-only prediction settlement ledger."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import math
import os
import re
import time
import unicodedata
import requests
import numpy as np
from scipy.stats import nbinom
from prediction_core import KALSHI_API_BASE, append_immutable_predictions, valid_probability, sports_series, events_for_series, rank_market_candidates, rank_single_candidates, extract_line, single_candidate_rejection_reasons
from mlb_math import score_probability_matrix
from probability_calibration import calibrate_probability

API_MLB = "https://statsapi.mlb.com/api/v1"
API_KALSHI = KALSHI_API_BASE
ROOT = Path(__file__).resolve().parent
PRED_FILE = ROOT / "predicciones_mlb.csv"
RESULT_FILE = ROOT / "resultados_mlb.csv"
TZ = "America/Chicago"
TIMEOUT = 15

MLB_TEAM_CODES = {
    "arizona diamondbacks": "ARI", "atlanta braves": "ATL", "baltimore orioles": "BAL",
    "boston red sox": "BOS", "chicago cubs": "CHC", "chicago white sox": "CWS",
    "cincinnati reds": "CIN", "cleveland guardians": "CLE", "colorado rockies": "COL",
    "detroit tigers": "DET", "houston astros": "HOU", "kansas city royals": "KC",
    "los angeles angels": "LAA", "los angeles dodgers": "LAD", "miami marlins": "MIA",
    "milwaukee brewers": "MIL", "minnesota twins": "MIN", "new york mets": "NYM",
    "new york yankees": "NYY", "athletics": "ATH", "oakland athletics": "ATH",
    "philadelphia phillies": "PHI", "pittsburgh pirates": "PIT", "san diego padres": "SD",
    "san francisco giants": "SF", "seattle mariners": "SEA", "st louis cardinals": "STL",
    "tampa bay rays": "TB", "texas rangers": "TEX", "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}


def codigo_equipo_mlb(nombre):
    return MLB_TEAM_CODES.get(normalizar(nombre), "")

PRED_COLUMNS = [
    "prediction_id", "event_ticker", "ticker", "match_id", "fecha", "local", "visita",
    "categoria", "mercado", "line", "lado", "seleccion", "bloque", "probabilidad_modelo", "precio_captura",
    "probabilidad_implicita", "edge", "estado_modelo", "reglas", "capturado_en", "disponibilidad",
]
RESULT_COLUMNS = ["prediction_id", "estado", "resultado_kalshi", "marcador", "resuelto_en"]
UNMATCHED_REPORT = ROOT / "mlb_eventos_sin_emparejar.csv"
UNMATCHED_COLUMNS = ["event_ticker", "series_ticker", "title", "start_time", "motivo"]
REJECTED_REPORT = ROOT / "mlb_candidatos_descartados.csv"
REJECTED_COLUMNS = ["match_id", "local", "visita", "event_ticker", "ticker", "lado", "categoria", "probabilidad_modelo", "precio", "edge", "motivo"]
CURRENT_REPORT = ROOT / "mlb_probabilidades_actuales.csv"
CURRENT_COLUMNS = ["match_id", "local", "visita", "event_ticker", "ticker", "lado", "categoria", "probabilidad_modelo", "precio_captura", "edge", "prediction_id"]
COVERAGE_REPORT = ROOT / "mlb_cobertura_por_partido.csv"
COVERAGE_COLUMNS = ["match_id", "fecha", "local", "visita", "contratos_modelables", "sencillas_validas", "alta_probabilidad_validas", "a_tener_en_cuenta_validas", "sencillas_guardadas", "alta_probabilidad_guardadas", "a_tener_en_cuenta_guardadas", "motivo_sencillas", "motivo_alta_probabilidad", "motivo_a_tener_en_cuenta"]


def get_json(url, params=None):
    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                last = requests.HTTPError(f"HTTP 429 (límite de peticiones) en {url}", response=r)
                if attempt < 3:
                    time.sleep(1 + attempt * 2)
                    continue
                raise last
            if r.status_code >= 500:
                last = requests.HTTPError(f"HTTP {r.status_code} en {url}", response=r)
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise last
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last = exc
            if attempt == 3:
                break
            time.sleep(1 + attempt * 2)
    raise last


def kalshi(path, **params):
    return get_json(f"{API_KALSHI}/{path}", params)


def normalizar(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def leer_csv(path, columnas):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def append_csv(path, columns, rows):
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writerows(rows)


def cargar_calendario(inicio, fin):
    data = get_json(f"{API_MLB}/schedule", {
        "sportId": 1, "startDate": inicio, "endDate": fin,
        "gameTypes": "R,F,D,L,W", "hydrate": "probablePitcher",
    })
    return [g for d in data.get("dates", []) for g in d.get("games", [])]


def cargar_stats(season):
    data = get_json(f"{API_MLB}/teams/stats", {
        "stats": "season", "group": "hitting", "season": season, "sportIds": 1,
    })
    stats = {}
    for block in data.get("stats", []):
        for split in block.get("splits", []):
            team, stat = split.get("team", {}), split.get("stat", {})
            n, runs = float(stat.get("gamesPlayed") or 0), float(stat.get("runs") or 0)
            if team.get("id") and math.isfinite(n) and math.isfinite(runs) and n > 0 and runs >= 0:
                stats[int(team["id"])] = {"name": team.get("name", ""), "rpg": runs/n, "games": n}
    return stats


def modelo_partido(game, stats):
    h = game.get("teams", {}).get("home", {})
    a = game.get("teams", {}).get("away", {})
    hid, aid = (h.get("team") or {}).get("id"), (a.get("team") or {}).get("id")
    if not hid or not aid:
        return None
    rates = [float(x["rpg"]) for x in stats.values() if math.isfinite(float(x.get("rpg", float("nan")))) and float(x.get("rpg", -1)) >= 0]
    mean = float(np.mean(rates)) if rates else 4.4
    if not math.isfinite(mean) or mean <= 0:
        mean = 4.4
    hs, aw = stats.get(int(hid)), stats.get(int(aid))
    mh = (0.75 * hs["rpg"] + 0.25 * mean) if hs else mean
    ma = (0.75 * aw["rpg"] + 0.25 * mean) if aw else mean
    mh, ma = max(1.8, mh * 1.03), max(1.8, ma * 0.98)
    axis = np.arange(0, 26)
    # Root cause of historical NaN%: extreme-but-finite API rates can underflow every
    # PMF value on the truncated 0..25 axis, making matrix.sum()==0 before normalization.
    # Use the shared bounded-mean/fallback normalizer instead of dividing by zero.
    matrix = score_probability_matrix(axis, mh, ma)
    return {
        "id": str(game.get("gamePk", "")),
        "home": h.get("team", {}).get("name", "Local"),
        "away": a.get("team", {}).get("name", "Visitante"),
        "date": game.get("gameDate", ""),
        "status": (game.get("status") or {}).get("abstractGameState", ""),
        "detailed_status": (game.get("status") or {}).get("detailedState", ""),
        "runs_home": h.get("score"), "runs_away": a.get("score"),
        "mu_home": mh, "mu_away": ma, "axis": axis, "matrix": matrix,
    }


def partidos_proximos():
    now = datetime.now(timezone.utc)
    start = now.date() - timedelta(days=1)
    # Kalshi publica contratos abiertos más allá de los próximos tres días; cargar dos semanas evita falsos no-emparejamientos por horizonte.
    end = now.date() + timedelta(days=14)
    stats = cargar_stats(now.year)
    games = cargar_calendario(start.isoformat(), end.isoformat())
    return [m for g in games if (m := modelo_partido(g, stats))], stats


def clasificar(serie, market):
    text = normalizar(" ".join(str(market.get(k, "")) for k in ("title", "yes_sub_title", "subtitle", "rules_primary")))
    s = normalizar(serie)
    if (any(x in s for x in ("player", "strikeout", "home_run", "hits", "total_bases", "rbi", "pitcher", "kxmlbhr", "kxmlbtb", "kxmlbrbi", "kxmlbsb", "kxmlbouts", "kxmlber", "kxmlbwalk"))
            or any(x in text for x in ("home run", "total bases", "strikeout", "hits", "rbis", "walks", "outs recorded", "earned runs", "stolen bases"))):
        return "Props de jugador"
    if "inningwin" in s or "first inning winner" in text or "win the first inning" in text:
        ticker = str(market.get("ticker") or market.get("market_ticker") or "").upper()
        inning_match = re.search(r"-(\d+)-(?:TIE|[A-Z]{2,3})$", ticker)
        inning_number = int(inning_match.group(1)) if inning_match else 1
        return f"Ganador de la {inning_number}.ª entrada"
    if any(x in text for x in ("first inning", "1st inning", "yrfi", "nrfi")) or "firstinning" in s:
        return "1.ª entrada / YRFI-NRFI"
    if any(x in text for x in ("first 5", "first five", "first 3", "first three", "first 7", "first seven", "f5 ", "f5 total")) or any(x in s for x in ("first5", "first3", "first7", "f5total", "kxmlbf5")):
        return "Entradas parciales"
    # A generic inning total is not a full-game total; this model has no per-inning score distribution.
    if "inning" in s or "inning" in text:
        return "Entrada individual"
    if "team total" in text or "teamtotal" in s:
        return "Total por equipo"
    if "spread" in s or "run line" in text or "runline" in s:
        return "Run line / margen"
    if "total" in s or re.search(r"\b(over|under|more than|less than)\b", text):
        return "Total de carreras"
    if "exact" in text and ("score" in text or "runs" in text):
        return "Marcador exacto"
    if "extra inning" in text or "extra innings" in text:
        return "Entradas extra"
    if any(x in s for x in ("game", "moneyline", "winner")) or "win" in text:
        return "Ganador (moneyline)"
    return "Otros mercados MLB"


def mercado_probabilidad(market, game):
    """Model supported game, team, inning and exact-score contracts; never invent player-prop probabilities."""
    title = str(market.get("title") or market.get("yes_sub_title") or "")
    text = normalizar(" ".join((title, str(market.get("subtitle") or ""), str(market.get("yes_sub_title") or ""))))
    mat, axis = game["matrix"], game["axis"]
    h, a = game["home"], game["away"]
    home, away = normalizar(h) in text, normalizar(a) in text
    ticker = str(market.get("ticker", ""))
    category = clasificar(market.get("series_ticker", ""), market)
    home_code, away_code = codigo_equipo_mlb(h), codigo_equipo_mlb(a)
    contract_code = ticker.rsplit("-", 1)[-1].upper()

    # KXMLBINNINGWIN is a distinct first-inning outcome market. Model the
    # first inning itself; do not reuse the full-game moneyline probability.
    if "BINNINGWIN" in ticker.upper() or "BINNINGWIN" in str(market.get("series_ticker", "")).upper():
        inning_match = re.search(r"-(\d+)-(?:TIE|[A-Z]{2,3})$", ticker)
        inning_number = int(inning_match.group(1)) if inning_match else 1
        if inning_number != 1:
            return None, "sin_modelo_especifico: solo se modela la 1.ª entrada"
        inning_axis = np.arange(0, 8)
        shape = 1 / 0.12
        def inning_pmf(mu):
            mean = max(0.01, float(mu) / 9)
            probs = nbinom.pmf(inning_axis, shape, shape / (shape + mean))
            total = float(probs.sum())
            return probs / total if total > 0 and math.isfinite(total) else np.array([1.0] + [0.0] * 7)
        inning_mat = np.outer(inning_pmf(game["mu_home"]), inning_pmf(game["mu_away"]))
        inning_mat /= inning_mat.sum()
        if contract_code == "TIE":
            p_yes = float(np.trace(inning_mat))
        elif contract_code == home_code:
            p_yes = float(inning_mat[np.tril_indices_from(inning_mat, -1)].sum())
        elif contract_code == away_code:
            p_yes = float(inning_mat[np.triu_indices_from(inning_mat, 1)].sum())
        else:
            return None, "sin_modelo_especifico: código de resultado de 1.ª entrada desconocido"
        return min(max(p_yes, 0.001), 0.999), "modelado_primera_entrada"
    if contract_code in {home_code, away_code}:
        # The ticker's terminal team code is more reliable than a title mentioning both teams.
        home, away = contract_code == home_code, contract_code == away_code

    # Prefer explicit natural-language lines; ticker parsing is a fallback.
    runs = re.search(r"(?:over|under|more than|less than|mas de|menos de)\s*\$?([0-9]+(?:\.[0-9]+)?)", text)
    spread = re.search(r"(?:run line|runline|spread|handicap|hándicap)\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    signed_line = re.search(r"(?<!\w)([+-][0-9]+(?:\.[0-9]+)?)", title)
    line_match = re.search(r"(?:TOTAL|OVER|UNDER|RUNS|SPREAD|F5)[A-Z_-]*([0-9]+(?:P[0-9]+)?)", ticker, re.I)
    line = float(runs.group(1)) if runs else (float(spread.group(1)) if spread else (float(signed_line.group(1)) if signed_line else (float(line_match.group(1).replace("P", ".")) if line_match else None)))
    p_yes = None
    model_state = "modelado"

    # First 3/5/7 innings: scale expected runs to the requested fraction of a game.
    if category == "Entradas parciales":
        if any(token in text for token in ("first 3", "first three")) or any(token in ticker.lower() for token in ("first3", "1st3")):
            fraction = 3 / 9
        elif any(token in text for token in ("first 7", "first seven")) or any(token in ticker.lower() for token in ("first7", "1st7")):
            fraction = 7 / 9
        else:
            fraction = 5 / 9
        partial_axis = np.arange(0, 18)
        dispersion = 0.12
        n = 1 / dispersion
        def pmf(mu):
            return nbinom.pmf(partial_axis, n, n / (n + max(0.05, mu * fraction)))
        mat = np.outer(pmf(game["mu_home"]), pmf(game["mu_away"]))
        mat /= mat.sum()
        axis = partial_axis
        # Normalize category so the common market predicates below can handle it.
        if any(word in text for word in ("moneyline", "winner", "to win", "win the first", "wins the first")):
            category = "Ganador (moneyline)"
        elif "spread" in text or "run line" in text:
            category = "Run line / margen"
        elif "team total" in text:
            category = "Total por equipo"
        else:
            category = "Total de carreras"

    if category == "Ganador (moneyline)":
        tie_probability = float(np.trace(mat))
        if home:
            p_yes = float(mat[np.tril_indices_from(mat, -1)].sum() + 0.5 * tie_probability)
        elif away:
            p_yes = float(mat[np.triu_indices_from(mat, 1)].sum() + 0.5 * tie_probability)
    elif category == "Total de carreras" and line is not None:
        total = axis[:, None] + axis[None, :]
        if "under" in text or "less than" in text:
            p_yes = float(mat[total < line].sum())
        elif "over" in text or "more than" in text:
            p_yes = float(mat[total > line].sum())
    elif category == "Total por equipo" and line is not None and (home or away):
        scores = axis[:, None] if home else axis[None, :]
        if "under" in text or "less than" in text:
            p_yes = float(mat[scores < line].sum())
        elif "over" in text or "more than" in text:
            p_yes = float(mat[scores > line].sum())
    elif category == "Run line / margen" and line is not None and (home or away):
        margin = axis[:, None] - axis[None, :] if home else axis[None, :] - axis[:, None]
        # A signed line is the team's handicap: +1.5 wins if margin > -1.5;
        # -1.5 wins if margin > +1.5. Unsigned titles are interpreted as a favorite line.
        signed_positive = (signed_line is not None and float(signed_line.group(1)) > 0) or "plus" in text
        threshold = -abs(line) if signed_positive else abs(line)
        p_yes = float(mat[margin > threshold].sum())
    elif category == "1.ª entrada / YRFI-NRFI":
        p_any_run = 1 - math.exp(-(game["mu_home"] + game["mu_away"]) / 9)
        p_yes = p_any_run if any(x in text for x in ("yrfi", "yes run", "at least one run", "a run scored")) else 1 - p_any_run
    elif category == "Entradas extra":
        # Extra innings are possible when the game is tied after nine; estimate from the score matrix.
        p_yes = float(np.trace(mat))
        if any(word in text for word in ("no extra innings", "no extra inning", "not go to extra")):
            p_yes = 1 - p_yes
    elif category == "Marcador exacto":
        # Support explicit "X-Y" score contracts when team order can be inferred from the title.
        score = re.search(r"\b([0-9]{1,2})\s*[-:]\s*([0-9]{1,2})\b", title)
        if score:
            x, y = int(score.group(1)), int(score.group(2))
            if home and away:
                p_yes = float(mat[x, y]) if x < mat.shape[0] and y < mat.shape[1] else 0.0
    if p_yes is not None and math.isfinite(p_yes):
        return min(max(float(p_yes), 0.001), 0.999), model_state
    return None, "sin_modelo_especifico"


def series_mlb():
    all_series = sports_series()
    selected = []
    for item in all_series:
        ticker = str(item.get("ticker", "")).upper()
        name = normalizar(item.get("title") or item.get("name") or "")
        tags = " ".join(map(str, item.get("tags") or [])).lower()
        excluded = ("FODT", "FUTURE", "DIVISION", "PLAYOFF", "CHAMPION", "PENNANT", "MVP", "AWARD", "TEAMWINS", "WORLD SERIES", "SERIES", "EXPAND", "INNINGTOTAL")
        if (ticker.startswith(("KXMLB", "KXBASEBALL")) or ("mlb" in name and ("baseball" in tags or "sports" in tags)) or ("major league baseball" in name)) and not any(x in ticker for x in excluded):
            selected.append(ticker)
    return sorted(set(selected))


def eventos_mlb(series=None):
    """Shared paginated Kalshi event/market catalogue for the selected MLB series."""
    return events_for_series(series if series is not None else series_mlb(), statuses=("open",))


MLB_TEAM_ALIASES = {
    "ARI": ("arizona diamondbacks", "diamondbacks", "d backs", "dbacks", "ari"),
    "ATL": ("atlanta braves", "braves", "atl"),
    "BAL": ("baltimore orioles", "orioles", "bal"),
    "BOS": ("boston red sox", "red sox", "bos"),
    "CHC": ("chicago cubs", "cubs", "chc"),
    "CWS": ("chicago white sox", "white sox", "cws", "chw"),
    "CIN": ("cincinnati reds", "reds", "cin"),
    "CLE": ("cleveland guardians", "guardians", "cle"),
    "COL": ("colorado rockies", "rockies", "col"),
    "DET": ("detroit tigers", "tigers", "det"),
    "HOU": ("houston astros", "astros", "hou"),
    "KC": ("kansas city royals", "royals", "kc", "kcr"),
    "LAA": ("los angeles angels", "la angels", "angels", "laa"),
    "LAD": ("los angeles dodgers", "los angeles d", "la dodgers", "dodgers", "lad"),
    "MIA": ("miami marlins", "marlins", "mia"),
    "MIL": ("milwaukee brewers", "milwaukee", "brewers", "mil"),
    "MIN": ("minnesota twins", "twins", "min"),
    "NYM": ("new york mets", "ny mets", "mets", "nym"),
    "NYY": ("new york yankees", "ny yankees", "yankees", "nyy"),
    "ATH": ("athletics", "oakland athletics", "sacramento athletics", "ath"),
    "PHI": ("philadelphia phillies", "phillies", "phi"),
    "PIT": ("pittsburgh pirates", "pirates", "pit"),
    "SD": ("san diego padres", "sd padres", "padres", "sd"),
    "SF": ("san francisco giants", "sf giants", "giants", "sf"),
    "SEA": ("seattle mariners", "mariners", "sea"),
    "STL": ("st louis cardinals", "st louis", "cardinals", "stl"),
    "TB": ("tampa bay rays", "tampa bay", "rays", "tb"),
    "TEX": ("texas rangers", "rangers", "tex"),
    "TOR": ("toronto blue jays", "blue jays", "tor"),
    "WSH": ("washington nationals", "nationals", "nats", "wsh", "was"),
}


def _team_matches_event(name, event_ticker, title):
    code = codigo_equipo_mlb(name)
    # Team codes are embedded in compact Kalshi event tickers such as LADMIL.
    if code and code in event_ticker:
        return True
    aliases = MLB_TEAM_ALIASES.get(code, (normalizar(name),))
    for alias in aliases:
        alias = normalizar(alias)
        if not alias:
            continue
        if len(alias) <= 3:
            if re.search(r"\b" + re.escape(alias) + r"\b", title):
                return True
        elif alias in title:
            return True
    return False


def motivo_evento_sin_partido(event, games):
    title = normalizar(event.get("title") or event.get("sub_title") or "")
    ticker = str(event.get("event_ticker") or "").upper()
    series_raw = str(event.get("series_ticker") or "").upper()
    series = normalizar(series_raw)
    if series_raw.startswith("KXMLBBGAME"):
        return "no_es_MLB/otra_competicion (serie KXMLBBGAME)"
    # MLB futures, awards, player props, and debut-date markets are not single-game markets.
    non_game_prefixes = (
        "KXMLB-", "KXMLBAL-", "KXMLBNL-", "KXMLBALCY", "KXMLBNLCY",
        "KXMLBALMOTY", "KXMLBNLMOTY", "KXMLBALRELOTY", "KXMLBNLRELOTY",
        "KXMLBALROTY", "KXMLBNLROTY", "KXMLBALCPOTY", "KXMLBNLCPOTY",
        "KXMLBALHAARON", "KXMLBNLHAARON", "KXMLBGG", "KXMLBDEBUT",
        "KXMLBCHAMPLEAGUE", "KXMLBHR", "KXMLBTB", "KXMLBRBI",
        "KXMLBSB", "KXMLBOUTS", "KXMLBER", "KXMLBERA", "KXMLBWALK",
        "KXMLBPLAYER", "KXCOACHOUTMLBDATE", "KXMLBEOTY", "KXMLBFASTPITCH",
        "KXMLBMATCHUP", "KXMLBOPENINGDAY", "KXMLBSS", "KXMLBSTAT",
        "KXMLBSTATCOUNT", "KXMLBSEASONGAMES", "KXMLBTEAMSALE",
    )
    if series_raw.startswith(non_game_prefixes) or any(token in title for token in
        ("cy young", "manager of the year", "rookie of the year", "comeback player", "hank aaron", "gold glove", "silver slugger", "executive of the year", "fastest pitch", "debut date", "next team", "champion", "opening day", "perfect game", "grand slam game", "home run game", "games played in the")):
        return "mercado_no_partido/futuro_premio_o_prop"
    matched = sum(
        any(_team_matches_event(team, ticker, title) for team in (game["home"], game["away"]))
        for game in games
    )
    if matched:
        return "equipos_identificados_pero_fecha_o_hora_ambigua"
    if any(token in series or token in title for token in ("game", "moneyline", "spread", "runline", "total", "inning", "f5")):
        return "aparenta_mercado_de_partido_pero_no_hay_alias_suficiente"
    return "evento_no_identificado/otro"


def _event_datetime(event):
    ticker = str(event.get("event_ticker") or "").upper()
    # Kalshi baseball tickers encode scheduled Eastern time, e.g. 26OCT112000.
    match = re.search(r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(\d{4})", ticker)
    if match:
        try:
            local = datetime.strptime("".join(match.groups()), "%y%b%d%H%M").replace(tzinfo=ZoneInfo("America/New_York"))
            return local.astimezone(timezone.utc)
        except ValueError:
            pass
    for field in ("start_time", "expected_expiration_time", "strike_date"):
        raw = str(event.get(field) or "")
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            try:
                return datetime.fromisoformat(raw[:10]).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def encontrar_partido(event, games):
    title = normalizar(event.get("title") or event.get("sub_title") or "")
    event_ticker = str(event.get("event_ticker") or "").upper()
    event_dt = _event_datetime(event)
    ranked = []
    for game in games:
        score = int(_team_matches_event(game["home"], event_ticker, title)) + int(_team_matches_event(game["away"], event_ticker, title))
        if score != 2:
            continue
        try:
            game_dt = datetime.fromisoformat(game["date"].replace("Z", "+00:00"))
            if game_dt.tzinfo is None:
                game_dt = game_dt.replace(tzinfo=timezone.utc)
            game_dt = game_dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue
        if event_dt:
            delta_seconds = abs((game_dt - event_dt).total_seconds())
            if delta_seconds > 12 * 3600:
                continue
            ranked.append((delta_seconds, game))
        else:
            ranked.append((0.0, game))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        return None
    if len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) < 60 and ranked[0][1]["id"] != ranked[1][1]["id"]:
        return None
    return ranked[0][1]

def mercados_liquidados(series):
    """Consulta por serie para reducir llamadas y respetar los límites de Kalshi."""
    finales = {}
    for serie in series:
        cursor = None
        for _ in range(30):
            params = {"limit": 200, "series_ticker": serie, "status": "settled"}
            if cursor:
                params["cursor"] = cursor
            try:
                page = kalshi("markets", **params)
            except requests.RequestException as exc:
                print(f"No se pudieron consultar liquidaciones de {serie}: {exc}")
                break
            for market in page.get("markets", []):
                ticker = market.get("ticker")
                result = str(market.get("result", "")).lower()
                if ticker and result in {"yes", "no"}:
                    finales[ticker] = result
            cursor = page.get("cursor")
            if not cursor:
                break
            time.sleep(0.35)
        time.sleep(0.5)
    return finales


def main():
    now = datetime.now(timezone.utc)
    games, _ = partidos_proximos()
    series = series_mlb()
    events = eventos_mlb(series)
    existing = leer_csv(PRED_FILE, PRED_COLUMNS)
    existing_ids = {x["prediction_id"] for x in existing}
    captured = now.isoformat()
    by_match = {}
    event_count = 0
    unmatched_event_count = 0
    unmatched_reasons = {}
    unmatched_events = []
    per_match_valid_contracts = {}
    unsupported_market_count = 0
    modelable_market_count = 0
    rejected_candidates = []

    # Model every actual, match-specific Kalshi market that this model supports.
    # Unmodelled contracts are deliberately excluded instead of being saved as NaN/blank picks.
    for event in events:
        markets = event.get("markets") or []
        if not markets and event.get("event_ticker"):
            try:
                markets = kalshi("markets", event_ticker=event["event_ticker"], limit=200).get("markets", [])
            except requests.RequestException:
                markets = []
        if not markets:
            continue
        game = encontrar_partido(event, games)
        if not game:
            unmatched_event_count += 1
            reason = motivo_evento_sin_partido(event, games)
            unmatched_reasons[reason] = unmatched_reasons.get(reason, 0) + 1
            unmatched_events.append({
                "event_ticker": event.get("event_ticker", ""),
                "series_ticker": event.get("series_ticker", ""),
                "title": event.get("title") or event.get("sub_title") or "",
                "start_time": event.get("start_time") or event.get("expected_expiration_time") or event.get("strike_date") or "",
                "motivo": reason,
            })
            continue
        if game.get("status") != "Preview":
            continue
        event_count += 1
        for market in markets:
            if market.get("status") not in (None, "active", "open", "unopened"):
                continue
            ticker = str(market.get("ticker") or "")
            if not ticker:
                continue
            market["series_ticker"] = market.get("series_ticker") or event.get("series_ticker", "")
            category = clasificar(market["series_ticker"], market)
            p_yes, model_state = mercado_probabilidad(market, game)
            if valid_probability(p_yes) is not None:
                p_yes, calibration_note = calibrate_probability(
                    p_yes, category, PRED_FILE, RESULT_FILE
                )
                model_state = (str(model_state or "") + " | " + calibration_note).strip(" |")
            if valid_probability(p_yes) is None:
                unsupported_market_count += 1
                rejected_candidates.append({
                    "match_id": game.get("id", ""), "local": game.get("home", ""), "visita": game.get("away", ""),
                    "event_ticker": event.get("event_ticker", ""), "ticker": ticker, "lado": "",
                    "categoria": category, "probabilidad_modelo": "", "precio": "", "edge": "",
                    "motivo": model_state or "sin modelo específico/datos finitos",
                })
                continue
            modelable_market_count += 1
            per_match_valid_contracts[str(game.get("id", ""))] = per_match_valid_contracts.get(str(game.get("id", "")), 0) + 1
            yes_price = _price(market, "yes_ask_dollars")
            no_price = _price(market, "no_ask_dollars")
            kickoff = game.get("date") or event.get("start_time") or event.get("expected_expiration_time") or ""
            for side, price, probability in (("YES", yes_price, p_yes), ("NO", no_price, 1-p_yes)):
                if valid_probability(probability) is None or price is None:
                    # A model probability without an executable entry quote is not
                    # a tradeable recommendation and must not enter the ledger.
                    continue
                cost_buffer = max(0.0, float(os.getenv("MERCADOS_BUFFER_COSTES", "0.01")))
                edge = probability - price - cost_buffer
                row = {
                    "prediction_id": "",
                    "event_ticker": event.get("event_ticker", ""),
                    "ticker": ticker,
                    "match_id": str(game.get("id", "")),
                    "fecha": kickoff,
                    "local": game.get("home", ""),
                    "visita": game.get("away", ""),
                    "categoria": category,
                    "mercado": market.get("yes_sub_title") or market.get("title") or ticker,
                    "line": extract_line(market.get("yes_sub_title") or market.get("title") or ""),
                    "lado": side,
                    "seleccion": side,
                    "bloque": "",
                    "probabilidad_modelo": f"{probability:.8f}",
                    "precio_captura": f"{price:.6f}" if price is not None else "",
                    "disponibilidad": "Disponible en Kalshi" if price is not None else "Sin precio en Kalshi",
                    "probabilidad_implicita": f"{price:.6f}" if price is not None else "",
                    "edge": f"{edge:.6f}" if edge is not None else "",
                    "estado_modelo": model_state,
                    "reglas": market.get("rules_primary") or market.get("rules_secondary") or "",
                    "capturado_en": captured,
                }
                by_match.setdefault(str(game.get("id", "")), []).append(row)

    new_rows = []
    minimum_combo = float(os.getenv("PROBABILIDAD_MINIMA_COMBO", "0.65"))
    for match_id, rows in by_match.items():
        candidates = []
        for row in rows:
            p = valid_probability(row.get("probabilidad_modelo"))
            if p is None:
                continue
            try:
                edge = float(row["edge"]) if row.get("edge") not in (None, "") else None
            except (TypeError, ValueError):
                edge = None
            candidates.append({
                "_row": row,
                "_p": p,
                "_lado": row["lado"],
                "Ticker": row["ticker"],
                "Apuesta": f'{row["mercado"]} · {row["lado"]}',
                "Categoría resumen": row["categoria"],
                "Probabilidad del modelo": f"{p:.1%}",
                "Precio de compra": f'{float(row["precio_captura"]):.0%}' if row["precio_captura"] else "—",
                "Ventaja (edge)": edge,
                "Lado Kalshi": row["lado"],
            })

        # Singles: price cap, positive edge, trivial-line exclusion and family diversification.
        singles = rank_single_candidates(candidates, limit=5)
        rejection_summary = single_candidate_rejection_reasons(candidates)
        print("CANDIDATOS MLB · match_id={} · {} vs {} · sencillas_rechazo_agregado={}".format(match_id, rows[0].get("local", ""), rows[0].get("visita", ""), rejection_summary))
        if not singles:
            reasons = single_candidate_rejection_reasons(candidates)
            print("SENCILLAS SIN CANDIDATAS · {} · {} vs {}: {}".format(match_id, rows[0].get("local", ""), rows[0].get("visita", ""), reasons or {"sin candidatos modelables": len(candidates)}))

        high, value = rank_market_candidates(candidates, min_probability=minimum_combo, high_limit=5, value_limit=10)
        def candidate_key(candidate):
            raw = candidate.get("_row") if isinstance(candidate.get("_row"), dict) else candidate
            return (str(raw.get("ticker") or candidate.get("Ticker") or ""),
                    str(candidate.get("_lado") or raw.get("lado") or ""))
        selected_keys = {
            candidate_key(candidate)
            for block_rows in (singles, high, value)
            for candidate in block_rows
        }
        for candidate in candidates:
            key = candidate_key(candidate)
            if key in selected_keys:
                continue
            raw = candidate.get("_row", {})
            reasons = []
            simple_reasons = single_candidate_rejection_reasons([candidate])
            reasons.extend(name for name, count in simple_reasons.items()
                           if count and name != "candidata que sí cumple")
            p = valid_probability(candidate.get("_p"))
            price = _price(raw, "precio_captura")
            edge = candidate.get("Ventaja (edge)")
            try:
                edge = float(edge) if edge not in (None, "") else None
            except (TypeError, ValueError):
                edge = None
            if p is not None and p < minimum_combo:
                reasons.append("probabilidad inferior al umbral de alta probabilidad")
            if price is None:
                reasons.append("sin precio disponible para el bloque de valor")
            elif edge is None:
                reasons.append("sin edge calculable para el bloque de valor")
            if not reasons or "candidata que sí cumple" in simple_reasons:
                reasons.append("excluida por diversificación de familia/equipo o límite del bloque")
            reason_text = "; ".join(dict.fromkeys(reasons))
            print("CANDIDATO MLB DESCARTADO · match_id={} · ticker={} · lado={} · motivo={}".format(
                match_id, key[0], key[1], reason_text
            ))
            rejected_candidates.append({
                "match_id": match_id, "local": raw.get("local", ""), "visita": raw.get("visita", ""),
                "event_ticker": raw.get("event_ticker", ""), "ticker": key[0], "lado": key[1],
                "categoria": raw.get("categoria", ""), "probabilidad_modelo": raw.get("probabilidad_modelo", ""),
                "precio": raw.get("precio_captura", ""), "edge": raw.get("edge", ""), "motivo": reason_text,
            })
        print("BLOQUES MLB DISPONIBLES · match_id={} · {} vs {} · sencillas_validas={} · alta_probabilidad_validas={} · a_tener_en_cuenta_validas={}".format(
            match_id, rows[0].get("local", ""), rows[0].get("visita", ""),
            len(singles), len(high), len(value)
        ))
        limits = {"SENCILLAS": 5, "ALTA_PROBABILIDAD": 5, "A_TENER_EN_CUENTA": 10}
        existing_match = [item for item in existing if str(item.get("match_id", "")) == str(match_id)]
        block_counts = {name: 0 for name in limits}
        existing_singles = []
        for item in existing_match:
            block_name = str(item.get("bloque", "") or "")
            if block_name not in limits:
                category_name = str(item.get("categoria", ""))
                block_name = next((name for name in limits if category_name.startswith(name + " · ")), "")
            if block_name == "SENCILLAS":
                existing_singles.append(item)
            elif block_name in limits:
                block_counts[block_name] += 1
        # Old, now-filtered singles must not occupy the slots forever; preserve them in the ledger
        # but only count those which still meet the current public display rules.
        block_counts["SENCILLAS"] = len(rank_single_candidates(existing_singles, limit=limits["SENCILLAS"]))

        for block, selected in (("SENCILLAS", singles), ("ALTA_PROBABILIDAD", high), ("A_TENER_EN_CUENTA", value)):
            count = block_counts[block]
            for candidate in selected:
                if count >= limits[block]:
                    break
                row = dict(candidate["_row"])
                side = candidate.get("_lado") or row["lado"]
                ticker = row["ticker"]
                row["prediction_id"] = f"{ticker}|{side}|{block}"
                row["categoria"] = f"{block} · {row['categoria']}"
                row["bloque"] = block
                if row["prediction_id"] in existing_ids:
                    continue
                new_rows.append(row)
                existing_ids.add(row["prediction_id"])
                count += 1

    capture_summary = append_immutable_predictions(
        PRED_FILE, PRED_COLUMNS, new_rows, probability_field="probabilidad_modelo"
    )
    print("Eventos MLB emparejados: {} | eventos sin partido StatsAPI coincidente: {} | contratos sin modelo específico/datos finitos: {} | contratos modelables: {} | capturas aceptadas: {} | descartadas: {}. Los resultados solo los escribe liquidar_mlb.py.".format(event_count, unmatched_event_count, unsupported_market_count, modelable_market_count, capture_summary["accepted"], capture_summary["rejected"]))
    with UNMATCHED_REPORT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNMATCHED_COLUMNS)
        writer.writeheader()
        writer.writerows(unmatched_events)
    print("CLASIFICACIÓN DE EVENTOS MLB SIN PARTIDO: " + repr(unmatched_reasons))
    print(f"INFORME DE EVENTOS SIN EMPAREJAR: {UNMATCHED_REPORT.name} · filas={len(unmatched_events)}")
    with REJECTED_REPORT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REJECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(rejected_candidates)
    print(f"INFORME DE CANDIDATOS DESCARTADOS: {REJECTED_REPORT.name} · filas={len(rejected_candidates)}")
    current_candidates = [row for match_rows in by_match.values() for row in match_rows]
    with CURRENT_REPORT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CURRENT_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in CURRENT_COLUMNS} for row in current_candidates)
    print(f"PROBABILIDADES MLB ACTUALES: {CURRENT_REPORT.name} · filas={len(current_candidates)}")
    coverage_rows = []
    captured_rows = existing + new_rows
    for game in games:
        if game.get("status") != "Preview":
            continue
        match_id = str(game.get("id", ""))
        rows_for_game = [row for row in captured_rows if str(row.get("match_id", "")) == match_id]
        counts = {name: 0 for name in ("SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA")}
        for row in rows_for_game:
            block = str(row.get("bloque", "") or "")
            if block not in counts:
                category_name = str(row.get("categoria", ""))
                block = next((name for name in counts if category_name.startswith(name + " · ")), "")
            if block in counts:
                counts[block] += 1
        modelable = per_match_valid_contracts.get(match_id, 0)
        candidates_for_game = []
        for row in by_match.get(match_id, []):
            p = valid_probability(row.get("probabilidad_modelo"))
            if p is None:
                continue
            try:
                edge = float(row["edge"]) if row.get("edge") not in (None, "") else None
            except (TypeError, ValueError, OverflowError):
                edge = None
            try:
                price = float(row.get("precio_captura")) if row.get("precio_captura") not in (None, "") else None
                if price is not None and price > 1:
                    price /= 100
                if price is not None and not 0 <= price <= 1:
                    price = None
            except (TypeError, ValueError, OverflowError):
                price = None
            candidates_for_game.append({
                "_row": row, "_p": p, "_lado": row.get("lado", ""),
                "Ticker": row.get("ticker", ""), "ticker": row.get("ticker", ""),
                "Apuesta": f'{row.get("mercado", "")} · {row.get("lado", "")}',
                "Probabilidad del modelo": p, "probabilidad_modelo": p,
                "Precio de compra": f"{price:.0%}" if price is not None else "—",
                "precio_captura": price, "Ventaja (edge)": edge, "edge": edge,
                "Lado Kalshi": row.get("lado", ""), "lado": row.get("lado", ""),
                "categoria": row.get("categoria", ""),
            })
        valid_singles = rank_single_candidates(candidates_for_game, limit=10)
        valid_high, valid_value = rank_market_candidates(
            candidates_for_game, min_probability=minimum_combo, high_limit=5, value_limit=10
        )
        rejected_for_game = [r for r in rejected_candidates if str(r.get("match_id", "")) == match_id]
        reason_pool = []
        for rejected in rejected_for_game:
            reason = str(rejected.get("motivo", "")).strip()
            if reason and reason not in reason_pool:
                reason_pool.append(reason)
        def block_reason(block_name, count):
            if count:
                return ""
            if match_id not in by_match:
                return "sin evento Kalshi emparejado en esta ejecución"
            if modelable == 0:
                return "ningún contrato de este partido tiene probabilidad modelable"
            if reason_pool:
                return "ninguna selección de este bloque superó filtros; descartes observados: " + " | ".join(reason_pool[:3])
            return "ningún candidato elegible para este bloque en la ejecución"
        coverage_rows.append({
            "match_id": match_id, "fecha": game.get("date", ""), "local": game.get("home", ""), "visita": game.get("away", ""),
            "contratos_modelables": modelable,
            "sencillas_validas": len(valid_singles), "alta_probabilidad_validas": len(valid_high),
            "a_tener_en_cuenta_validas": len(valid_value),
            "sencillas_guardadas": counts["SENCILLAS"], "alta_probabilidad_guardadas": counts["ALTA_PROBABILIDAD"],
            "a_tener_en_cuenta_guardadas": counts["A_TENER_EN_CUENTA"],
            "motivo_sencillas": block_reason("SENCILLAS", counts["SENCILLAS"]),
            "motivo_alta_probabilidad": block_reason("ALTA_PROBABILIDAD", counts["ALTA_PROBABILIDAD"]),
            "motivo_a_tener_en_cuenta": block_reason("A_TENER_EN_CUENTA", counts["A_TENER_EN_CUENTA"]),
        })
    with COVERAGE_REPORT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COVERAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(coverage_rows)
    print(f"COBERTURA POR PARTIDO: {COVERAGE_REPORT.name} · partidos_preview={len(coverage_rows)}")
    for match_id, rows in sorted(by_match.items()):
        counts = {name: 0 for name in ("SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA")}
        for row in existing + new_rows:
            if str(row.get("match_id", "")) == match_id and str(row.get("bloque", "")) in counts:
                counts[str(row["bloque"])] += 1
        print("COBERTURA MLB · match_id={} · {} vs {} · contratos_modelables={} · capturas_guardadas_por_bloque={}".format(
            match_id, rows[0].get("local", ""), rows[0].get("visita", ""),
            per_match_valid_contracts.get(match_id, 0), counts
        ))


def _price(market, *keys):
    for key in keys:
        value = market.get(key)
        try:
            if value not in (None, ""):
                price = float(value)
                if price > 1:
                    price /= 100
                if 0 < price < 1 and math.isfinite(price):
                    return price
        except (ValueError, TypeError, OverflowError):
            continue
    return None


if __name__ == "__main__":
    main()
