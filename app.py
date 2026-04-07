from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import pandas as pd
import joblib
import json
import sqlite3
import numpy as np
from pathlib import Path

app = Flask(__name__)
app.secret_key = "super_secret_aeropredict_key"

BASE = Path(__file__).parent

# ---------------- Load Model & Metadata ----------------
try:
    model = joblib.load(BASE / "FlightDelayPredictionModel.pkl")
except:
    model = None
feature_cols = pd.read_csv(BASE / "model_feature_columns.csv", header=None)[0].tolist()
feature_cols = [str(c) for c in feature_cols if str(c) != '0']

with open(BASE / "model_metadata.json", "r") as f:
    meta = json.load(f)

top_k_maps = meta["top_k_maps"]
numeric_features = meta["num_features"]

# ---------------- Database Setup ----------------
def init_db():
    conn = sqlite3.connect(BASE / 'database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        name TEXT, 
        email TEXT UNIQUE, 
        password TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS ContactMessages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        name TEXT, 
        email TEXT, 
        message TEXT, 
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS PredictionLogs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        user_id INTEGER, 
        airline TEXT, 
        origin TEXT, 
        destination TEXT, 
        departure_time TEXT, 
        distance REAL, 
        predicted_label INTEGER, 
        delay_probability REAL, 
        ontime_probability REAL, 
        weather TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect(BASE / 'database.db')
    conn.row_factory = sqlite3.Row
    return conn

# ---------------- Decorators ----------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "danger")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin'):
            flash("Admin access required.", "danger")
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- Helper Functions ----------------
def time_to_minutes(x):
    try:
        x = int(x)
        return (x // 100) * 60 + (x % 100)
    except:
        return 0

def preprocess_input(data):
    row = {}

    row["Month"] = int(data.get("Month", 1))
    row["DayOfWeek"] = int(data.get("DayOfWeek", 1))
    row["Distance"] = float(data.get("Distance", 0))
    row["CRSDep_MIN"] = time_to_minutes(data.get("CRSDepTime", 0))

    df = pd.DataFrame([row])

    # categorical
    for col, top_list in top_k_maps.items():
        val = data.get(col, "Other")
        if val not in top_list:
            val = "Other"
        df[col] = val

    # one-hot encoding
    for col in top_k_maps.keys():
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    return df[feature_cols]

# ---------------- Public & Auth Routes ----------------
@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        conn = get_db_connection()
        conn.execute("INSERT INTO ContactMessages (name, email, message) VALUES (?, ?, ?)", (name, email, message))
        conn.commit()
        conn.close()
        flash("Message sent successfully!", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        
        hashed_pw = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO Users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_pw))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists.", "danger")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM Users WHERE email = ?", (email,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user["password"], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            flash(f"Welcome back, {user['name']}!", "success")
            # Clear any admin session
            session.pop('is_admin', None)
            return redirect(url_for("homepage"))
        else:
            flash("Invalid email or password.", "danger")
            
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("homepage"))

# ---------------- User Protected Routes ----------------
@app.route("/predict_form")
@login_required
def predict_form():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    form_data = {k: request.form.get(k) for k in request.form.keys()}
    
    try:
        X = preprocess_input(form_data)
        
        # ML Logic
        if model:
            proba = model.predict_proba(X)[0]
            prob_on_time = float(proba[0])
            prob_delayed = float(proba[1])
        else:
            prob_on_time = 0.6
            prob_delayed = 0.4

        # Extra metrics heuristic
        weather = form_data.get("Weather", "Clear")
        traffic = form_data.get("TrafficLevel", "Low")
        raw_prev_delay = form_data.get("PreviousDelay", "").strip()
        prev_delay = float(raw_prev_delay) if raw_prev_delay else 0.0

        prob_modifier = 0.0
        explanation_triggers = []

        if weather in ["Storm", "Fog"]:
            prob_modifier += 0.2
            explanation_triggers.append(f"adverse weather ({weather})")
        elif weather == "Rain":
            prob_modifier += 0.1
            explanation_triggers.append("rainy conditions")

        if prev_delay > 15:
            prob_modifier += 0.15
            explanation_triggers.append("previous flight delay cascade")

        if traffic == "High":
            prob_modifier += 0.1
            explanation_triggers.append("high airport traffic")
        
        prob_delayed = min(1.0, prob_delayed + prob_modifier)
        prob_on_time = 1.0 - prob_delayed
        label = 1 if prob_delayed > 0.5 else 0

        if label == 1 and explanation_triggers:
            explanation_string = "Delay likely due to: " + " + ".join(explanation_triggers) + "."
        elif label == 1:
            explanation_string = "Standard scheduled operational delay detected."
        elif explanation_triggers:
            explanation_string = "Expected on-time, but monitor: " + " & ".join(explanation_triggers) + "."
        else:
            explanation_string = "Optimal clearing conditions for on-time departure."

        prob_delayed_perc = round(prob_delayed * 100, 2)
        prob_ontime_perc = round(prob_on_time * 100, 2)

        result = {
            "predicted_label": label,
            "prob_delayed": prob_delayed_perc,
            "prob_on_time": prob_ontime_perc,
            "explanation": explanation_string
        }

        # Save to database
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO PredictionLogs 
            (user_id, airline, origin, destination, departure_time, distance, predicted_label, delay_probability, ontime_probability, weather)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], form_data.get('Reporting_Airline'), form_data.get('Origin'), form_data.get('Dest'),
              form_data.get('CRSDepTime'), form_data.get('Distance'), label, prob_delayed_perc, prob_ontime_perc, weather))
        conn.commit()
        conn.close()

        return render_template("result.html", result=result, input=form_data)
    except Exception as e:
        print("Prediction Error:", e)
        flash("Invalid input. Please ensure all fields are filled properly.", "danger")
        return redirect(url_for("predict_form"))

@app.route("/history")
@login_required
def history():
    conn = get_db_connection()
    logs = conn.execute("SELECT * FROM PredictionLogs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20", (session['user_id'],)).fetchall()
    conn.close()
    return render_template("history.html", history=logs)

# ---------------- Admin Module ----------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "admin123":
            session.clear()
            session['is_admin'] = True
            flash("Admin logged in successfully.", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid admin credentials.", "danger")
    return render_template("admin_login.html")

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    conn = get_db_connection()
    users_count = conn.execute("SELECT COUNT(*) as count FROM Users").fetchone()['count']
    messages_count = conn.execute("SELECT COUNT(*) as count FROM ContactMessages").fetchone()['count']
    predictions_count = conn.execute("SELECT COUNT(*) as count FROM PredictionLogs").fetchone()['count']
    conn.close()
    return render_template("admin_dashboard.html", u_count=users_count, m_count=messages_count, p_count=predictions_count)

@app.route("/admin/messages")
@admin_required
def admin_messages():
    conn = get_db_connection()
    msgs = conn.execute("SELECT * FROM ContactMessages ORDER BY timestamp DESC").fetchall()
    conn.close()
    return render_template("admin_messages.html", messages=msgs)

@app.route("/admin/messages/delete/<int:msg_id>")
@admin_required
def delete_message(msg_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM ContactMessages WHERE id = ?", (msg_id,))
    conn.commit()
    conn.close()
    flash("Message deleted.", "success")
    return redirect(url_for("admin_messages"))

@app.route("/admin/predictions")
@admin_required
def admin_predictions():
    conn = get_db_connection()
    preds = conn.execute('''
        SELECT p.*, u.name as user_name 
        FROM PredictionLogs p 
        LEFT JOIN Users u ON p.user_id = u.id 
        ORDER BY p.timestamp DESC LIMIT 50
    ''').fetchall()
    conn.close()
    return render_template("admin_predictions.html", predictions=preds)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
