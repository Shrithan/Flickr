from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flashsafe import analyze as analyze_low, Config as ConfigLow, write_outputs, sanitize_video
from flashsafe_v2 import analyze as analyze_high, Config as ConfigHigh, dampen_video, preset_params
from flashsafe import download_youtube
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
REPORT_FOLDER = "data/output/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return "FlashSafe API running"


# ── STEP 1: Analyse only (fast) ─────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    try:
        tolerance = (request.form.get("tolerance") or request.args.get("tolerance") or "low").lower()
        username = (request.form.get("username") or request.args.get("username") or "").strip().lower()
        password = (request.form.get("password") or request.args.get("password") or "")
        if username and FAKE_USERS.get(username) == password:
            db = _load_db()
            user_data = _get_user_data(db, username)
            lum_thresh = sensitivity_to_threshold(user_data["profile"]["sensitivity"], tolerance)
        else:
            user_t = float(request.form.get("threshold") or request.args.get("threshold") or 0.5)
            user_t = max(0.1, min(0.9, user_t))
            lum_thresh = sensitivity_to_threshold(user_t, tolerance)
        if tolerance == "high":
            cfg = ConfigHigh()
            cfg.luminance_segment_threshold = lum_thresh
            result = analyze_high(filepath, cfg)
            cut_intervals = result.get("effect_intervals_sec", [])
        else:
            cfg = ConfigLow()
            cfg.luminance_segment_threshold = lum_thresh
            result = analyze_low(filepath, cfg)
            cut_intervals = result.get("cut_intervals_sec", [])
        return jsonify({
            "triggers":           result.get("possible_triggers", 0),
            "timestamps":         result.get("harmful_segment_end_times_sec", []),
            "cut_intervals":      cut_intervals,
            "tolerance":          tolerance,
            "filename":           file.filename,
            "adaptive_threshold": result.get("adaptive_threshold"),
            "effective_threshold": result.get("effective_threshold"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── STEP 2: Create safe video — handles file upload AND YouTube/video URL ───
@app.route("/create-safe", methods=["POST"])
def create_safe_video():
    tolerance = (request.form.get("tolerance") or "low").lower()
    url  = request.form.get("url", "").strip()
    file = request.files.get("video")

    if url:
        # YouTube or direct video URL — download first
        try:
            print(f"Downloading: {url}")
            filepath  = download_youtube(url, out_dir=UPLOAD_FOLDER)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
        except Exception as e:
            return jsonify({"error": f"Download failed: {str(e)}"}), 500

    elif file and file.filename:
        filepath  = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        base_name = os.path.splitext(file.filename)[0]

    else:
        return jsonify({"error": "Provide a video file or a URL"}), 400

    try:
        safe_filename = "safe_" + base_name + ".mp4"
        safe_path     = os.path.join(OUTPUT_FOLDER, safe_filename)

        username = (request.form.get("username") or "").strip().lower()
        password = (request.form.get("password") or "")
        if username and FAKE_USERS.get(username) == password:
            db = _load_db()
            user_data = _get_user_data(db, username)
            lum_thresh = sensitivity_to_threshold(user_data["profile"]["sensitivity"], tolerance)
        else:
            user_t = float(request.form.get("threshold") or 0.5)
            user_t = max(0.1, min(0.9, user_t))
            lum_thresh = sensitivity_to_threshold(user_t, tolerance)

        # Personalised dampening params from frontend (computed from sensitivity slider)
        dim_factor = float(request.form.get("dim_factor") or 0.65)
        sat_factor = float(request.form.get("sat_factor") or 0.60)
        blur_px    = int(float(request.form.get("blur_px") or 3))
        dim_factor = max(0.1, min(1.0, dim_factor))
        sat_factor = max(0.1, min(1.0, sat_factor))
        blur_px    = max(0, min(15, blur_px))

        if tolerance == "high":
            cfg    = ConfigHigh()
            cfg.luminance_segment_threshold = lum_thresh
            result = analyze_high(filepath, cfg)
            report_path, _ = write_outputs(result, REPORT_FOLDER)
            # Use personalised dampen params instead of preset
            dampen_video(filepath, report_path, safe_path,
                         dim_factor=dim_factor,
                         sat_factor=sat_factor,
                         blur_px=blur_px,
                         ramp_sec=0.12)
        else:
            cfg    = ConfigLow()
            cfg.luminance_segment_threshold = lum_thresh
            result = analyze_low(filepath, cfg)
            report_path, _ = write_outputs(result, REPORT_FOLDER)
            sanitize_video(filepath, report_path, safe_path)

        return jsonify({
            "triggers":           result.get("possible_triggers", 0),
            "timestamps":         result.get("harmful_segment_end_times_sec", []),
            "tolerance":          tolerance,
            "download_url":       f"/download/{safe_filename}",
            "adaptive_threshold": result.get("adaptive_threshold"),
            "effective_threshold": result.get("effective_threshold"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Download ────────────────────────────────────────────────────────────────
@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)




# ── USER DB + ADAPTIVE THRESHOLD MODEL ─────────────────────────────────────

import json as _json
from datetime import datetime, timezone, timedelta

SEIZURE_DB = 'seizure_store.json'

FAKE_USERS = {
    'alex':   'flickr123',
    'sarah':  'safe456',
    'jordan': 'log789',
}

# sensitivity: 0.1 = very strict (low threshold), 0.9 = relaxed (high threshold)
# luminance_segment_threshold = BASE_THRESH - (sensitivity_inverted * RANGE)
# low tolerance base: 25.0,  range 0–20  → sensitivity 0.1 → thresh=23, 0.5 → thresh=15, 0.9 → thresh=7
# high tolerance base: 15.0, range 0–12  → sensitivity 0.1 → thresh=14, 0.5 → thresh=9,  0.9 → thresh=3
BASE_LOW  = 25.0
BASE_HIGH = 15.0
RANGE_LOW  = 18.0
RANGE_HIGH = 11.0

def sensitivity_to_threshold(sensitivity, mode='low'):
    # sensitivity 0.1 (strict) → high threshold deduction → lower actual threshold → catches more
    inv = 1.0 - sensitivity   # 0.1 strict → inv=0.9 → max deduction
    if mode == 'high':
        return max(3.0, BASE_HIGH - inv * RANGE_HIGH)
    return max(5.0, BASE_LOW - inv * RANGE_LOW)

def _default_profile():
    return {
        'sensitivity': 0.5,
        'feedback_count': 0,
        'seizure_after_video': 0,
        'no_issue_after_video': 0,
    }

def _load_db():
    if not os.path.exists(SEIZURE_DB):
        return {}
    try:
        with open(SEIZURE_DB, 'r') as f:
            return _json.load(f)
    except Exception:
        return {}

def _save_db(data):
    with open(SEIZURE_DB, 'w') as f:
        _json.dump(data, f, indent=2)

def _get_user_data(db, username):
    if username not in db:
        db[username] = {'events': [], 'profile': _default_profile()}
    if 'profile' not in db[username]:
        db[username]['profile'] = _default_profile()
    if 'events' not in db[username]:
        db[username]['events'] = []
    return db[username]


@app.route('/login', methods=['POST'])
def login():
    body = request.get_json(silent=True) or {}
    username = (body.get('username') or '').strip().lower()
    password = body.get('password', '')
    if FAKE_USERS.get(username) == password:
        db = _load_db()
        user_data = _get_user_data(db, username)
        profile = user_data['profile']
        return jsonify({
            'ok': True,
            'username': username,
            'displayName': username.capitalize(),
            'sensitivity': profile['sensitivity'],
            'threshold_low':  round(sensitivity_to_threshold(profile['sensitivity'], 'low'), 2),
            'threshold_high': round(sensitivity_to_threshold(profile['sensitivity'], 'high'), 2),
            'feedback_count': profile['feedback_count'],
        })
    return jsonify({'ok': False, 'error': 'Invalid credentials'}), 401


@app.route('/log-seizure', methods=['POST'])
def log_seizure():
    body = request.get_json(silent=True) or {}
    username = (body.get('username') or '').strip().lower()
    password = body.get('password', '')
    note     = (body.get('note') or '').strip()
    if FAKE_USERS.get(username) != password:
        return jsonify({'error': 'Unauthorised'}), 401
    db = _load_db()
    user_data = _get_user_data(db, username)
    event = {'timestamp': datetime.now(timezone.utc).isoformat(), 'note': note}
    user_data['events'].insert(0, event)
    _save_db(db)
    return jsonify({'ok': True, 'event': event, 'total': len(user_data['events'])})


@app.route('/seizure-events', methods=['GET'])
def seizure_events():
    username = (request.args.get('username') or '').strip().lower()
    password = request.args.get('password', '')
    period   = request.args.get('period', 'all')
    if FAKE_USERS.get(username) != password:
        return jsonify({'error': 'Unauthorised'}), 401
    db = _load_db()
    user_data  = _get_user_data(db, username)
    all_events = user_data['events']
    cutoffs = {'week': 7, 'month': 30, '3months': 90}
    if period in cutoffs:
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=cutoffs[period])
        filtered = [e for e in all_events if datetime.fromisoformat(e['timestamp']) >= cutoff_dt]
    else:
        filtered = all_events
    return jsonify({'events': filtered, 'total_all': len(all_events)})


@app.route('/seizure-report', methods=['GET'])
def seizure_report():
    username = (request.args.get('username') or '').strip().lower()
    password = request.args.get('password', '')
    if FAKE_USERS.get(username) != password:
        return jsonify({'error': 'Unauthorised'}), 401
    db         = _load_db()
    user_data  = _get_user_data(db, username)
    all_events = user_data['events']
    profile    = user_data['profile']
    now        = datetime.now(timezone.utc)
    def count_period(days):
        cutoff = now - timedelta(days=days)
        return sum(1 for e in all_events if datetime.fromisoformat(e['timestamp']) >= cutoff)
    report = {
        'username': username, 'display_name': username.capitalize(),
        'generated_at': now.isoformat(), 'total_all_time': len(all_events),
        'last_7_days': count_period(7), 'last_30_days': count_period(30),
        'last_90_days': count_period(90),
        'avg_per_week_30d': round(count_period(30) / 4.3, 2),
        'last_event': all_events[0]['timestamp'] if all_events else None,
        'recent_events': all_events[:20],
        'sensitivity': profile['sensitivity'],
        'threshold_low':  round(sensitivity_to_threshold(profile['sensitivity'], 'low'), 2),
        'threshold_high': round(sensitivity_to_threshold(profile['sensitivity'], 'high'), 2),
        'feedback_count': profile['feedback_count'],
    }
    return jsonify(report)


@app.route('/update-sensitivity', methods=['POST'])
def update_sensitivity():
    """Allow frontend slider to directly set a user's sensitivity."""
    body = request.get_json(silent=True) or {}
    username = (body.get('username') or '').strip().lower()
    password = body.get('password', '')
    if FAKE_USERS.get(username) != password:
        return jsonify({'error': 'Unauthorised'}), 401
    try:
        new_sensitivity = float(body.get('sensitivity', 0.5))
        new_sensitivity = max(0.1, min(0.9, new_sensitivity))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid sensitivity value'}), 400

    db = _load_db()
    user_data = _get_user_data(db, username)
    user_data['profile']['sensitivity'] = round(new_sensitivity, 3)
    _save_db(db)

    return jsonify({
        'ok': True,
        'sensitivity': user_data['profile']['sensitivity'],
        'threshold_low':  round(sensitivity_to_threshold(new_sensitivity, 'low'), 2),
        'threshold_high': round(sensitivity_to_threshold(new_sensitivity, 'high'), 2),
    })


# ── ADAPTIVE MODEL: video feedback ──────────────────────────────────────────
@app.route('/feedback', methods=['POST'])
def video_feedback():
    body     = request.get_json(silent=True) or {}
    username = (body.get('username') or '').strip().lower()
    password = body.get('password', '')
    outcome  = body.get('outcome', '')   # 'seizure' | 'no_issue'

    if FAKE_USERS.get(username) != password:
        return jsonify({'error': 'Unauthorised'}), 401
    if outcome not in ('seizure', 'no_issue'):
        return jsonify({'error': 'outcome must be seizure or no_issue'}), 400

    db        = _load_db()
    user_data = _get_user_data(db, username)
    profile   = user_data['profile']

    profile['feedback_count'] += 1
    s = profile['sensitivity']

    if outcome == 'seizure':
        # Stricter: lower sensitivity (catches more) — bigger step
        profile['seizure_after_video'] += 1
        s = max(0.1, s - 0.08)
    else:
        # Relax only after 3 consecutive no-issue feedbacks — smaller step
        profile['no_issue_after_video'] += 1
        # Only relax if recent no-issue feedbacks outweigh seizure ones
        ratio = profile['no_issue_after_video'] / max(1, profile['seizure_after_video'] + profile['no_issue_after_video'])
        if ratio >= 0.75 and profile['no_issue_after_video'] >= 3:
            s = min(0.9, s + 0.03)

    profile['sensitivity'] = round(s, 3)
    _save_db(db)

    return jsonify({
        'ok': True,
        'sensitivity': profile['sensitivity'],
        'threshold_low':  round(sensitivity_to_threshold(profile['sensitivity'], 'low'), 2),
        'threshold_high': round(sensitivity_to_threshold(profile['sensitivity'], 'high'), 2),
        'feedback_count': profile['feedback_count'],
    })


@app.route('/user-profile', methods=['GET'])
def user_profile():
    username = (request.args.get('username') or '').strip().lower()
    password = request.args.get('password', '')
    if FAKE_USERS.get(username) != password:
        return jsonify({'error': 'Unauthorised'}), 401
    db        = _load_db()
    user_data = _get_user_data(db, username)
    profile   = user_data['profile']
    return jsonify({
        'sensitivity': profile['sensitivity'],
        'threshold_low':  round(sensitivity_to_threshold(profile['sensitivity'], 'low'), 2),
        'threshold_high': round(sensitivity_to_threshold(profile['sensitivity'], 'high'), 2),
        'feedback_count': profile['feedback_count'],
        'seizure_after_video': profile['seizure_after_video'],
        'no_issue_after_video': profile['no_issue_after_video'],
    })



# ── AI CHAT PROXY (Gemini) ───────────────────────────────────────────────────
from google import genai
from google.genai import types

CHAT_SYSTEM = (
    "You are a compassionate and knowledgeable epilepsy support assistant called Flickr AI. "
    "You help people with epilepsy and their caregivers by: explaining different types of seizures "
    "and epilepsy conditions in plain language, discussing common seizure triggers and how to avoid them, "
    "advising on what information to track and share with neurologists, providing emotional support and "
    "reassurance, explaining medical terms, and guiding users on when to seek urgent medical attention. "
    "You are warm, clear, and patient. You always remind users that you cannot provide medical diagnoses "
    "and that they should consult their neurologist or doctor for any medical decisions. When someone "
    "describes symptoms, help them understand what they might mean and what questions to ask their doctor "
    "- never diagnose. If someone seems to be in a medical emergency, urgently tell them to call emergency "
    "services. Keep responses concise and easy to read. Use short paragraphs."
)

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    messages = body.get("messages", [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set on server"}), 500

    clean = [m for m in messages if m.get("role") in ("user", "assistant")]

    # Build contents list for new SDK
    contents = []
    for m in clean:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=str(m["content"]))]))

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=CHAT_SYSTEM),
        )
        reply = response.text or "Sorry, I could not generate a response."
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)