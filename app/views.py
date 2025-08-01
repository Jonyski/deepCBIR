from flask import Flask, request, session, redirect, url_for, render_template, flash, send_from_directory, current_app
from werkzeug.utils import secure_filename
from .models import deepCBIR
import os

cbir = deepCBIR()
app = Flask(__name__)

@app.after_request
def add_header(response):
    response.cache_control.max_age = 300
    return response

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/retrieve", methods=["GET", "POST"])
def retrieve():
    query = request.files["file"]
    scope = int(request.form["scope"])

    query_img_path = "./app/tmp/query.jpg"
    query.save(query_img_path)

    retrieve_img_paths = cbir.retrieve_images(query_img_path, scope=scope, debug=False)
    cbir.create_plot([query_img_path])
    cbir.create_plot(retrieve_img_paths)

    return render_template("result.html")

@app.route('/uploads/<path:filename>')
def get_images(filename):
    base_dir = os.path.dirname(current_app.root_path) # This should get you to the root directory (deepCBIR-main)
    image_directory = os.path.join(base_dir, 'app', 'tmp')
    return send_from_directory(image_directory,
                                filename, as_attachment=True, cache_timeout=0)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r