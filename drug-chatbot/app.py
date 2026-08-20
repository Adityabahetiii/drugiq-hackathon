"""
Drug Information Q&A Chatbot with Citations & Role-Based Access
--------------------------------------------------------------
Flask backend. Run with: python app.py

Features:
  - Role-based access: Doctor (Prescription, ChromaDB, Knowledge Graph) vs. Patient (OTC only, OpenFDA)
  - Ask drug questions via REST API
  - Answers grounded only in official drug PDFs (Doctors) or OpenFDA OTC labels (Patients)
  - Citations with drug name + page number(s)
  - PDF upload & knowledge graph for healthcare professionals
  - Full conversation memory & contextual query rewriting
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import ssl
import uuid
import threading

# ── SSL Fix ────────────────────────────────────────────────────────
# Windows often has SSL certificate issues with Python. This fixes it
# by disabling SSL verification for HuggingFace model downloads.
# MUST run before ANY library imports.

os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Monkey-patch httpx to disable SSL verification (used by huggingface_hub)
import httpx
_original_client_init = httpx.Client.__init__
_original_async_client_init = httpx.AsyncClient.__init__

def _patched_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    _original_client_init(self, *args, **kwargs)

def _patched_async_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    _original_async_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_client_init
httpx.AsyncClient.__init__ = _patched_async_client_init

import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
# ── End SSL Fix ────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from backend.vector_store import (
    build_vector_store, get_vector_store, vector_store_exists,
    get_all_drug_names, add_pdf_to_store, get_drug_info,
    remove_file_from_store, get_indexed_files
)
from backend.otc_lookup import OTC_DATA
from backend.rag_engine import ask
from backend.knowledge_graph import build_knowledge_graph, load_cached_graph

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_base_dir, ".env"))

PDF_FOLDER = os.path.join(_base_dir, "data", "pdfs")
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")



os.makedirs(PDF_FOLDER, exist_ok=True)

# The UI is built via `npm run build` in ../drugiq-newfrontend.
FRONTEND_DIST = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "drugiq-newfrontend", "dist"
))

app = Flask(__name__, static_folder=os.path.join(FRONTEND_DIST, "assets"), static_url_path="/assets")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

db_state = {
    "collection": None,
    "ready": False
}
chat_sessions = {}  # session_id -> {"chat_history": [], "role": "patient"}

# Auto-load database on startup
if vector_store_exists():
    try:
        db_state["collection"] = get_vector_store()
        db_state["ready"] = True
        print("✅ Database auto-loaded successfully!")
    except Exception as e:
        print(f"⚠️  Could not auto-load database: {e}")

# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    if not os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        return (
            "Frontend build not found. Run `npm run build` in drugiq-newfrontend, "
            "then restart the server.",
            500,
        )
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.route('/figures/<path:filename>')
def serve_figures_root(filename):
    if os.path.exists(os.path.join(FIGURES_FOLDER, filename)):
        return send_from_directory(FIGURES_FOLDER, filename)
    figures_dist = os.path.join(FRONTEND_DIST, "figures")
    if os.path.exists(os.path.join(figures_dist, filename)):
        return send_from_directory(figures_dist, filename)
    return ("Figure not found", 404)


@app.route('/api/figures/<path:filename>')
def serve_figures_api(filename):
    if os.path.exists(os.path.join(FIGURES_FOLDER, filename)):
        return send_from_directory(FIGURES_FOLDER, filename)
    figures_dist = os.path.join(FRONTEND_DIST, "figures")
    if os.path.exists(os.path.join(figures_dist, filename)):
        return send_from_directory(figures_dist, filename)
    return ("Figure not found", 404)

# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route('/api/status')
def status():
    return jsonify({
        "db_ready": db_state["ready"],
        "groq_key_set": bool(GROQ_API_KEY),
        "pdf_count": len([f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")])
    })


@app.route('/api/drugs')
def list_drugs():
    role = request.args.get("role", "doctor").strip().lower()
    if role == "patient":
        otc_list = []
        for key, data in OTC_DATA.items():
            b_name = data.get("brand_name", key.capitalize())
            g_name = data.get("generic_name", key.upper())
            disp_name = f"{b_name} ({g_name})" if g_name.lower() not in b_name.lower() else b_name
            otc_list.append({
                "drug_name": disp_name,
                "generic_name": g_name,
                "brand_name": b_name,
                "source_file": f"FDA Drug Facts — {b_name}",
                "product_type": "OTC",
                "purpose": data.get("purpose", ""),
                "source_url": data.get("source_url", ""),
                "chunk_count": 1,
            })
        return jsonify({"drugs": otc_list, "files": [d["source_file"] for d in otc_list]})

    if not db_state["ready"] or not db_state["collection"]:
        if vector_store_exists():
            try:
                db_state["collection"] = get_vector_store()
                db_state["ready"] = True
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return jsonify({"drugs": [], "files": []})
        else:
            return jsonify({"drugs": [], "files": []})

    drug_info = get_drug_info(db_state["collection"])
    files = get_indexed_files(db_state["collection"])
    return jsonify({"drugs": drug_info, "files": files})


@app.route('/api/knowledge-graph')
def knowledge_graph():
    role = request.args.get("role", "patient").strip().lower()
    if role == "patient":
        return jsonify({"error": "Access restricted to healthcare professionals"}), 403

    if not db_state["ready"] or not db_state["collection"]:
        return jsonify({"error": "Database not loaded. Upload PDFs and build the database first."}), 400
    if not GROQ_API_KEY:
        return jsonify({"error": "No Groq API key set. Add it in Settings."}), 400
    cached = load_cached_graph()
    if cached:
        return jsonify(cached)
    try:
        graph = build_knowledge_graph(db_state["collection"], GROQ_API_KEY)
        return jsonify(graph)
    except Exception as e:
        return jsonify({"error": f"Error building knowledge graph: {str(e)}"}), 500


@app.route('/api/knowledge-graph/rebuild', methods=['POST'])
def rebuild_knowledge_graph():
    data = request.json or {}
    role = request.args.get("role") or data.get("role", "patient")
    if str(role).strip().lower() == "patient":
        return jsonify({"error": "Access restricted to healthcare professionals"}), 403

    if not db_state["ready"] or not db_state["collection"]:
        return jsonify({"error": "Database not loaded. Upload PDFs and build the database first."}), 400
    if not GROQ_API_KEY:
        return jsonify({"error": "No Groq API key set. Add it in Settings."}), 400
    try:
        graph = build_knowledge_graph(db_state["collection"], GROQ_API_KEY)
        return jsonify(graph)
    except Exception as e:
        return jsonify({"error": f"Error rebuilding knowledge graph: {str(e)}"}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    global GROQ_API_KEY
    data = request.json or {}
    question = data.get("question", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))
    api_key = data.get("api_key") or GROQ_API_KEY
    
    # Read role (default to "patient" if missing or invalid)
    role_input = data.get("role", "patient")
    role = "doctor" if role_input == "doctor" else "patient"

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not api_key:
        return jsonify({"error": "No Groq API key set. Add it in Settings."}), 400

    # For doctors, verify vector database is loaded
    if role == "doctor" and not db_state["ready"]:
        return jsonify({"error": "Database not loaded. Upload PDFs and build the database first."}), 400

    # Get or create session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"chat_history": [], "role": role}
    session = chat_sessions[session_id]
    session["role"] = role

    # Hydrate history from client if session in memory is empty
    client_history = data.get("chat_history")
    if not session["chat_history"] and client_history and isinstance(client_history, list):
        session["chat_history"] = [
            {"role": m.get("role", "user"), "content": m.get("content") or m.get("text", "")}
            for m in client_history if (m.get("content") or m.get("text"))
        ]

    # Get RAG / OTC answer based on role
    try:
        result = ask(
            question=question,
            collection=db_state["collection"],
            chat_history=session["chat_history"],
            groq_api_key=api_key,
            role=role,
            session=session,
        )
    except Exception as e:
        return jsonify({"error": f"Error generating answer: {str(e)}"}), 500

    # Update conversation history
    if result.get("query_type") != "overview_menu":
        session["chat_history"].append({"role": "user", "content": question})
        session["chat_history"].append({"role": "assistant", "content": result["answer"]})
    elif result.get("drug_name"):
        session["chat_history"].append({"role": "user", "content": question})
        session["chat_history"].append({
            "role": "assistant",
            "content": f"[Displayed overview options for {result['drug_name']}]"
        })

    return jsonify({
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "citation": result.get("citation"),
        "source_file": result.get("source_file"),
        "page_number": result.get("page_number"),
        "refused": result.get("refused", False),
        "query_type": result.get("query_type", "general"),
        "drug_name": result.get("drug_name"),
        "session_id": session_id,
        "role": role,
        "anomaly_count": session.get("anomaly_count", 0),
    })


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    role = request.form.get("role", "patient").strip().lower()
    if role == "patient":
        return jsonify({"error": "Uploading prescribing PDFs is restricted to healthcare professionals."}), 403

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(PDF_FOLDER, filename)
    file.save(filepath)

    # Index the uploaded PDF
    try:
        if db_state["ready"] and db_state["collection"]:
            collection = add_pdf_to_store(filepath, db_state["collection"])
        else:
            collection = build_vector_store(PDF_FOLDER)

        db_state["collection"] = collection
        db_state["ready"] = True

        # Find the drug name
        drug_info = get_drug_info(collection)
        drug_name = filename
        for info in drug_info:
            if info["source_file"] == filename:
                drug_name = info["drug_name"]
                break

        if GROQ_API_KEY:
            threading.Thread(
                target=_rebuild_graph_safely, args=(collection, GROQ_API_KEY), daemon=True
            ).start()

        return jsonify({
            "success": True,
            "message": f"{drug_name} indexed successfully!",
            "filename": filename,
            "drug_name": drug_name
        })
    except Exception as e:
        return jsonify({"error": f"Error indexing PDF: {str(e)}"}), 500


def _rebuild_graph_safely(collection, api_key):
    try:
        build_knowledge_graph(collection, api_key)
        print("✅ Knowledge graph rebuilt after upload")
    except Exception as e:
        print(f"⚠️  Knowledge graph rebuild after upload failed: {e}")


@app.route('/api/build', methods=['POST'])
def build_db():
    data = request.json or {}
    role = data.get("role", "patient")
    if role == "patient":
        return jsonify({"error": "Database building is restricted to healthcare professionals."}), 403

    try:
        collection = build_vector_store(PDF_FOLDER)
        db_state["collection"] = collection
        db_state["ready"] = True
        return jsonify({"success": True, "message": "Database rebuilt successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/drugs/<filename>', methods=['DELETE'])
def remove_drug(filename):
    role = request.args.get("role", "patient")
    if role == "patient":
        return jsonify({"error": "Unauthorized"}), 403

    if not db_state["ready"] or not db_state["collection"]:
        return jsonify({"error": "Database not loaded"}), 400
    try:
        remove_file_from_store(filename, db_state["collection"])
        filepath = os.path.join(PDF_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"success": True, "message": f"Removed {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


FIGURES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "figures")
os.makedirs(FIGURES_FOLDER, exist_ok=True)


@app.route('/api/pdf/<filename>')
def serve_pdf(filename):
    return send_from_directory(PDF_FOLDER, filename)


@app.route('/api/download/<filename>')
def download_pdf(filename):
    return send_from_directory(PDF_FOLDER, filename, as_attachment=True)


@app.route('/api/clear', methods=['POST'])
def clear_chat():
    data = request.json or {}
    session_id = data.get("session_id", "")
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return jsonify({"success": True})


@app.route('/api/set_key', methods=['POST'])
def set_key():
    global GROQ_API_KEY
    data = request.json or {}
    key = data.get("api_key", "")
    if key:
        GROQ_API_KEY = key
        return jsonify({"success": True})
    return jsonify({"error": "No key provided"}), 400


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print()
    print(" DrugIQ — Drug Information Assistant")
    print("=" * 42)
    print(f"   PDF Folder : {PDF_FOLDER}")
    print(f"   Database   : {'✅ Ready' if db_state['ready'] else '❌ Not loaded'}")
    print(f"   API Key    : {'✅ Set' if GROQ_API_KEY else '❌ Not set'}")
    print("=" * 42)

    print()
    app.run(debug=True, port=5002, host='0.0.0.0')
