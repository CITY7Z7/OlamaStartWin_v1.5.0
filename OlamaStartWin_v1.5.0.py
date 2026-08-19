import os
import re
import sys
import time
import json
import queue
import threading
import subprocess
import shutil
import urllib.request
import urllib.error
from pathlib import Path

import psutil
import customtkinter as ctk
from tkinter import messagebox   # Canvas удалён


# ============================================================
# CONFIGURATION
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Prefer Ollama from PATH; fall back to the standard per-user installation path.
OLLAMA_PATH = shutil.which("ollama") or str(
    Path.home() / r"AppData\Local\Programs\Ollama\ollama.exe"
)

OLLAMA_HOST = "http://127.0.0.1:11434"

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800


# ============================================================
# MODEL DESCRIPTIONS
# ============================================================

PROMPT_PRESETS = {
"General chat": "Write a short and clear answer to my request. Don't add unnecessary fluff.",
"Rewrite": "Rewrite the following text clearly, naturally, and correctly, preserving the original meaning:\n\n",
"Summarize": "Make a brief summary of the following text. Highlight only the key points:\n\n",
"Code review": "Analyze the following code. Find errors, potential problems, and suggest specific improvements:\n\n",
"Explain": "Explain it in simple terms, step by step, and without unnecessary complexity:\n\n",
}


DESCRIPTIONS = {
"qwen2.5-coder:0.5b":
"An ultra-lightweight model for on-the-fly code completion",

"qwen2.5-coder:1.5b":
"A fast coding assistant for low-end PCs and laptops",

"qwen2.5-coder:3b":
"A balanced mini-model for code generation",

"qwen2.5-coder:7b":
"An excellent AI programmer, smart error correction",

"qwen2.5-coder:14b":
"A powerful model for complex architecture and refactoring",

"qwen2.5:3b":
"A fast, universal chatbot for general tasks",

"qwen2.5:7b":
"A popular base model, excellent in Russian",

"qwen3.5:4b":
"An updated, lightweight model, improved Logic",

"qwen3:8b":
"A mid-sized flagship general-purpose bot for text processing",

"gemma2:2b":
"A lightweight and fast chatbot from Google",

"gemma2:9b-instruct-q4_K_M":
"Advanced instruction following and deep analysis",

"gemma4:12b":
"Google's multimodal model: logic, code, vision",

"dolphin-mistral:7b":
"A versatile and creative chatbot",

"wizard-vicuna-uncensored:13b":
"A classic large model",

"muse-glimmer:30b":
"A heavy model for multi-step tasks",

"SetneufPT/Qwopus3.5-9B-Coder_Q3_64k_8GB-GPU:latest":
"A coding model with context 64K"
}


# ============================================================
# HELPERS
# ============================================================

def format_bytes(value):
    try:
        value = float(value)
    except Exception:
        return str(value)

    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{value:.1f} PB"


def run_command(args, timeout=None):
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW   # <-- добавлено
        )
        return (result.returncode, result.stdout.strip(), result.stderr.strip())
    except subprocess.TimeoutExpired:
        return -1, "", "The operation has exceeded the waiting time..."
    except Exception as e:
        return -1, "", str(e)


def ollama_process_running():
    """
    Checks for the presence of the Ollama process.
    """

    try:
        for proc in psutil.process_iter(["name"]):
            name = proc.info.get("name")

            if name and "ollama" in name.lower():
                return True

    except Exception:
        pass

    return False


def ollama_api_available():
    """
    Checks the real Ollama API.
    """

    try:
        request = urllib.request.Request(
            f"{OLLAMA_HOST}/api/tags",
            method="GET"
        )

        with urllib.request.urlopen(request, timeout=1.5) as response:
            return response.status == 200

    except Exception:
        return False


def api_generate(model, prompt):
    """
    Starts generation via Ollama API.

    stream=false allows you to obtain summary statistics:
        eval_count
        eval_duration
        prompt_eval_count
        total_duration
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    start_time = time.perf_counter()

    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))

    wall_time = time.perf_counter() - start_time

    return result, wall_time


# ============================================================
# MAIN APPLICATION
# ============================================================

class OllamaManagerApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Ollama Control Manager")
        self.geometry("900x840")
        self.minsize(900, 620)

        self.current_selected_model = None
        self.loaded_models = []
        self.model_rows = {}
        self.model_selection_var = ctk.StringVar(value="")
        self.loaded_model_details = ""

        self.worker_queue = queue.Queue()
        self.operation_running = False

        # System monitor
        self.ram_history = []
        self.gpu_history = []
        self.cpu_history = []
        self.max_history = 60
        self.gpu_available = None
        self.monitor_running = True

        # Prompt presets live in memory so the whole application remains one file.
        self.prompt_presets = dict(PROMPT_PRESETS)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.build_interface()

        self.after(100, self.process_worker_queue)
        self.after(500, self.status_loop)
        self.after(700, self.system_monitor_loop)

        # Initial boot.
        self.after(800, self.refresh_models)


    # ========================================================
    # UI
    # ========================================================

    def build_interface(self):
        """Compact layout: controls left, models/graphs right, huge output at bottom."""

        # TOP BAR
        self.top_bar = ctk.CTkFrame(self, height=40, corner_radius=0)
        self.top_bar.pack(side="top", fill="x")
        self.top_bar.pack_propagate(False)

        self.logo_label = ctk.CTkLabel(
            self.top_bar,
            text="OLLAMA CONTROL",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.logo_label.pack(side="left", padx=10)

        self.status_label = ctk.CTkLabel(
            self.top_bar,
            text="CHECKING...",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.status_label.pack(side="right", padx=10)

        # MAIN AREA
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # LEFT: compact controls
        self.left_panel = ctk.CTkFrame(self.main_frame, width=175, corner_radius=7)
        self.left_panel.pack(side="left", fill="y", padx=(0, 6))
        self.left_panel.pack_propagate(False)

        ctk.CTkLabel(
            self.left_panel, text="SERVER",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=8, pady=(8, 4))

        button_common = {"height": 26, "corner_radius": 5, "font": ctk.CTkFont(size=11)}

        self.btn_start = ctk.CTkButton(
            self.left_panel, text="START", command=self.confirm_start_server,
            fg_color="#2ecc71", hover_color="#27ae60", **button_common
        )
        self.btn_start.pack(fill="x", padx=7, pady=2)

        self.btn_stop = ctk.CTkButton(
            self.left_panel, text="STOP", command=self.confirm_stop_server,
            fg_color="#e74c3c", hover_color="#c0392b", **button_common
        )
        self.btn_stop.pack(fill="x", padx=7, pady=2)

        self.btn_force_stop = ctk.CTkButton(
            self.left_panel, text="SOS", command=self.force_stop_ollama,
            fg_color="#8e44ad", hover_color="#71368a", **button_common
        )
        self.btn_force_stop.pack(fill="x", padx=7, pady=2)

        self.btn_restart = ctk.CTkButton(
            self.left_panel, text="Restart",
            command=self.confirm_restart_server, **button_common
        )
        self.btn_restart.pack(fill="x", padx=7, pady=2)

        ctk.CTkFrame(self.left_panel, height=1, fg_color="gray30").pack(
            fill="x", padx=8, pady=6
        )

        ctk.CTkLabel(
            self.left_panel, text="OLLAMA",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.btn_ps = ctk.CTkButton(
            self.left_panel, text="ollama ps",
            command=self.confirm_show_ps, **button_common
        )
        self.btn_ps.pack(fill="x", padx=7, pady=2)

        self.btn_all_models = ctk.CTkButton(
            self.left_panel, text="List of all models",
            command=self.confirm_show_all_models, **button_common
        )
        self.btn_all_models.pack(fill="x", padx=7, pady=2)

        self.btn_download = ctk.CTkButton(
            self.left_panel, text="Download the model",
            command=self.confirm_download_model, **button_common
        )
        self.btn_download.pack(fill="x", padx=7, pady=2)

        self.btn_refresh = ctk.CTkButton(
            self.left_panel, text="Refresh the list",
            command=self.confirm_refresh, **button_common
        )
        self.btn_refresh.pack(fill="x", padx=7, pady=2)

        self.loaded_model_label = ctk.CTkLabel(
            self.left_panel,
            text="Uploaded model:\n—",
            justify="left",
            anchor="w",
            text_color="#d7d7d7",
            font=ctk.CTkFont(size=10, weight="bold"),
            wraplength=155
        )
        self.loaded_model_label.pack(fill="x", padx=8, pady=(10, 4))

        self.server_info = ctk.CTkLabel(
            self.left_panel,
            text="API: checking...\nProcess: Checking...",
            justify="left",
            anchor="w",
            text_color="gray",
            font=ctk.CTkFont(size=9)
        )
        self.server_info.pack(fill="x", padx=8, pady=(2, 6))

        self.download_progress = ctk.CTkProgressBar(
            self.left_panel, height=6, mode="determinate"
        )
        self.download_progress.pack(fill="x", padx=8, pady=(2, 3))
        self.download_progress.set(0)

        self.download_progress_label = ctk.CTkLabel(
            self.left_panel,
            text="Ready",
            text_color="gray",
            anchor="w",
            font=ctk.CTkFont(size=9)
        )
        self.download_progress_label.pack(fill="x", padx=8)

        # RIGHT AREA
        self.right_panel = ctk.CTkFrame(self.main_frame, corner_radius=7)
        self.right_panel.pack(side="right", fill="both", expand=True)

        # MODEL LIST — compact rows
        model_header = ctk.CTkFrame(self.right_panel, fg_color="transparent", height=28)
        model_header.pack(fill="x", padx=7, pady=(5, 2))
        model_header.pack_propagate(False)

        ctk.CTkLabel(
            model_header,
            text="INSTALLED MODELS (Choose a model and test it)",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="left")

        self.models_frame = ctk.CTkScrollableFrame(
            self.right_panel, corner_radius=5
        )
        self.models_frame.pack(fill="both", expand=True, padx=6, pady=3)

        # MONITOR — always on the right side (no graphs, only numbers and bars)
        self.monitor_frame = ctk.CTkFrame(self.right_panel, corner_radius=6)
        self.monitor_frame.pack(fill="x", padx=6, pady=(3, 4))

        monitor_title = ctk.CTkFrame(self.monitor_frame, fg_color="transparent", height=20)
        monitor_title.pack(fill="x", padx=7, pady=(3, 0))
        monitor_title.pack_propagate(False)

        ctk.CTkLabel(
            monitor_title, text="SYSTEM",
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(side="left")

        ctk.CTkLabel(
            monitor_title, text="--- LIVE UPD---", text_color="#2ecc71",
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(side="right")

        monitor_content = ctk.CTkFrame(self.monitor_frame, fg_color="transparent")
        monitor_content.pack(fill="x", padx=5, pady=(1, 5))

        # Three panels for CPU, RAM, GPU – now just titles and bars
        self.cpu_panel = ctk.CTkFrame(monitor_content, corner_radius=5)
        self.cpu_panel.pack(side="left", fill="both", expand=True, padx=(0, 2))

        self.ram_panel = ctk.CTkFrame(monitor_content, corner_radius=5)
        self.ram_panel.pack(side="left", fill="both", expand=True, padx=2)

        self.gpu_panel = ctk.CTkFrame(monitor_content, corner_radius=5)
        self.gpu_panel.pack(side="left", fill="both", expand=True, padx=(2, 0))

        self.cpu_header = ctk.CTkLabel(
            self.cpu_panel, text="CPU —", anchor="w",
            font=ctk.CTkFont(size=9, weight="bold")
        )
        self.cpu_header.pack(fill="x", padx=6, pady=(4, 1))

        self.ram_header = ctk.CTkLabel(
            self.ram_panel, text="RAM —", anchor="w",
            font=ctk.CTkFont(size=9, weight="bold")
        )
        self.ram_header.pack(fill="x", padx=6, pady=(4, 1))

        self.gpu_header = ctk.CTkLabel(
            self.gpu_panel, text="GPU —", anchor="w",
            font=ctk.CTkFont(size=9, weight="bold")
        )
        self.gpu_header.pack(fill="x", padx=6, pady=(4, 1))

        self.cpu_bar = ctk.CTkProgressBar(self.cpu_panel, height=5)
        self.cpu_bar.pack(fill="x", padx=6, pady=(0, 2))
        self.cpu_bar.set(0)

        self.ram_bar = ctk.CTkProgressBar(self.ram_panel, height=5)
        self.ram_bar.pack(fill="x", padx=6, pady=(0, 2))
        self.ram_bar.set(0)

        self.gpu_bar = ctk.CTkProgressBar(self.gpu_panel, height=5)
        self.gpu_bar.pack(fill="x", padx=6, pady=(0, 2))
        self.gpu_bar.set(0)

        # Canvas graphics removed -------------------------------------------------------------------------

        # Selected model / actions
        self.selected_frame = ctk.CTkFrame(self.right_panel, corner_radius=5)
        self.selected_frame.pack(fill="x", padx=6, pady=(0, 5))

        self.selected_label = ctk.CTkLabel(
            self.selected_frame, text="Model not selected",
            anchor="w", font=ctk.CTkFont(size=10, weight="bold")
        )
        self.selected_label.pack(side="left", padx=7, pady=5)

        small_btn = {"height": 25, "corner_radius": 5, "font": ctk.CTkFont(size=10), "text_color": "black"}

        self.btn_info = ctk.CTkButton(
            self.selected_frame, text="Info", width=55,
            command=self.confirm_model_info, state="disabled", **small_btn
        )
        self.btn_info.pack(side="right", padx=2, pady=3)

        self.btn_delete = ctk.CTkButton(
            self.selected_frame, text="Delete from disk", width=60,
            fg_color="#c0392b", hover_color="#922b21",
            command=self.confirm_delete_model, state="disabled", **small_btn
        )
        self.btn_delete.pack(side="right", padx=2, pady=3)

        self.btn_unload = ctk.CTkButton(
            self.selected_frame, text="Unload from memory", width=70,
            fg_color="#00AAA7", hover_color="#00C4C1",
            command=self.confirm_unload_model, state="disabled", **small_btn
        )
        self.btn_unload.pack(side="right", padx=2, pady=3)

        self.btn_cmd = ctk.CTkButton(
            self.selected_frame, text="CMD", width=48,
            command=self.confirm_run_cmd, state="disabled", **small_btn
        )
        self.btn_cmd.pack(side="right", padx=2, pady=3)









        # BOTTOM TEST AREA — response dominates the available height.
        self.test_frame = ctk.CTkFrame(self, corner_radius=6)
        self.test_frame.pack(side="bottom", fill="both", expand=False, padx=6, pady=(0, 6))
        self.test_frame.configure(height=285)
        self.test_frame.pack_propagate(False)

        test_toolbar = ctk.CTkFrame(self.test_frame, fg_color="transparent", height=30)
        test_toolbar.pack(fill="x", padx=6, pady=(5, 2))
        test_toolbar.pack_propagate(False)

        ctk.CTkLabel(
            test_toolbar, text="GENERATION AND TEST",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="left", padx=(2, 5))

        self.preset_menu = ctk.CTkOptionMenu(
            test_toolbar, values=list(self.prompt_presets.keys()),
            command=self.apply_prompt_preset, width=125, height=24,
            font=ctk.CTkFont(size=10)
        )
        self.preset_menu.set(list(self.prompt_presets.keys())[0])
        self.preset_menu.pack(side="left", padx=2)
        
        # Apply example - doesn't work
        self.btn_apply_preset = ctk.CTkButton(
            test_toolbar, text="Apply example", width=70, height=24,
            command=lambda: self.apply_prompt_preset(self.preset_menu.get()),
            font=ctk.CTkFont(size=10)
        )
        self.btn_apply_preset.pack(side="left", padx=2)

        self.btn_test = ctk.CTkButton(
            test_toolbar, text="TEST", width=65, height=24,
            command=self.confirm_test_model, state="disabled",
            font=ctk.CTkFont(size=10, weight="bold")
        )
        self.btn_test.pack(side="left", padx=2)

        self.test_stats = ctk.CTkLabel(
            test_toolbar, text="Time — | Speed ​​— | Tokens —",
            text_color="gray", font=ctk.CTkFont(size=9)
        )
        self.test_stats.pack(side="right", padx=3)

        # Small prompt row; huge response area below.
        self.prompt_box = ctk.CTkTextbox(self.test_frame, height=52, font=ctk.CTkFont(size=11))
        self.prompt_box.pack(fill="x", padx=6, pady=(0, 4))
        self.prompt_box.insert("1.0", "Write a short text about your capabilities")

        self.response_box = ctk.CTkTextbox(
            self.test_frame, height=190, font=ctk.CTkFont(size=11)
        )
        self.response_box.pack(fill="both", expand=True, padx=6, pady=(0, 6))


    # ========================================================
    # CONFIRMATION
    # ========================================================

    def confirm(self, title, message):
        return messagebox.askyesno(
            title,
            message,
            parent=self
        )


    # ========================================================
    # SERVER ACTIONS
    # ========================================================

    def confirm_start_server(self):

        if not self.confirm(
            "Launch of Ollama",
            "Start Ollama server?"
        ):
            return

        self.start_server()


    def start_server(self):

        if ollama_api_available():
            self.show_info(
                "Ollama",
                "The Ollama server is already running."
            )
            return

        self.run_background(
            self._start_server_worker
        )


    def _start_server_worker(self):

        if not os.path.exists(OLLAMA_PATH):
            self.worker_queue.put(
                (
                    "error",
                    "Ollama not found.\n\n"
                    f"Путь:\n{OLLAMA_PATH}"
                )
            )
            return

        try:

            subprocess.Popen(
                [
                    OLLAMA_PATH,
                    "serve"
                ],
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            # We are waiting for the API to appear
            for _ in range(30):

                if ollama_api_available():

                    self.worker_queue.put(
                        (
                            "message",
                            "Ollama has been successfully launched."
                        )
                    )

                    return

                time.sleep(0.5)

            self.worker_queue.put(
                (
                    "error",
                    "Ollama was started but the API did not respond."
                )
            )

        except Exception as e:

            self.worker_queue.put(
                (
                    "error",
                    f"Launch error:\n{e}"
                )
            )


    def confirm_stop_server(self):

        if not self.confirm(
            "Ollama stop",
            "Stop Ollama server?\n\n"
            "All running models will also be stopped."
        ):
            return

        self.stop_server()


    def stop_server(self):

        self.run_background(
            self._stop_server_worker
        )


    def _stop_server_worker(self):

        stopped = False

        try:

            for proc in psutil.process_iter(
                ["pid", "name"]
            ):

                name = proc.info.get("name")

                if name and name.lower() == "ollama.exe":

                    try:
                        proc.terminate()
                        stopped = True
                    except Exception:
                        pass

            time.sleep(1)

            # If the process is still alive, we force it to terminate.
            for proc in psutil.process_iter(
                ["pid", "name"]
            ):

                name = proc.info.get("name")

                if name and name.lower() == "ollama.exe":

                    try:
                        proc.kill()
                    except Exception:
                        pass

            if stopped:

                self.worker_queue.put(
                    (
                        "message",
                        "Ollama stopped."
                    )
                )

            else:

                self.worker_queue.put(
                    (
                        "message",
                        "Ollama had already been stopped."
                    )
                )

        except Exception as e:

            self.worker_queue.put(
                (
                    "error",
                    f"Stop error:\n{e}"
                )
            )


    # ========================================================
    # FORCE STOP — TASKKILL
    # ========================================================

    def force_stop_ollama(self):

        if not self.confirm(
            "Forced stop",
            "Force kill all Ollama processes?\n\n"
            "The following will be stopped:\n"
            "• ollama.exe\n"
            "• ollama app.exe\n"
            "• llama-server.exe"
        ):
            return

        self.run_background(
            self._force_stop_ollama_worker
        )


    def _force_stop_ollama_worker(self):

        try:

            processes = [
                "ollama.exe",
                "ollama app.exe",
                "llama-server.exe"
            ]

            stopped = []

            for process_name in processes:

                result = subprocess.run(
                    [
                        "taskkill",
                        "/f",
                        "/im",
                        process_name
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )

                if result.returncode == 0:
                    stopped.append(process_name)

            self.worker_queue.put(
                (
                    "message",
                    "Forced stop completed.\n\n"
                    "Processes completed:\n"
                    + (
                        "\n".join(
                            f"• {name}"
                            for name in stopped
                        )
                        if stopped
                        else "• No active processes found."
                    )
                )
            )

        except Exception as e:

            self.worker_queue.put(
                (
                    "error",
                    f"Error in forced stop:\n{e}"
                )
            )


    def confirm_restart_server(self):

        if not self.confirm(
            "Ollama Restart",
            "Restart Ollama server?"
        ):
            return

        self.run_background(
            self._restart_server_worker
        )


    def _restart_server_worker(self):

        self._stop_server_worker()

        time.sleep(1)

        self._start_server_worker()


    # ========================================================
    # MODEL REFRESH
    # ========================================================

    def confirm_refresh(self):

        if not self.confirm(
            "Update",
            "Update the list of installed models?"
        ):
            return

        self.refresh_models()


    def refresh_models(self):

        if self.operation_running:
            return

        self.run_background(
            self._refresh_models_worker
        )


    def _refresh_models_worker(self):

        returncode, stdout, stderr = run_command(
            [
                OLLAMA_PATH,
                "list"
            ],
            timeout=20
        )

        if returncode != 0:

            self.worker_queue.put(
                (
                    "models_error",
                    stderr or "Failed to retrieve list of models."
                )
            )

            return

        models = self.parse_ollama_list(stdout)

        self.worker_queue.put(
            (
                "models",
                models
            )
        )


    def parse_ollama_list(self, output):

        models = []

        lines = output.splitlines()

        if len(lines) <= 1:
            return models

        for line in lines[1:]:

            line = line.strip()

            if not line:
                continue

            # NAME is usually at the beginning of the line.
            # The remaining columns are separated by spaces.
            match = re.match(
                r"^(\S+)\s+(\S+)\s+(.+?)\s{2,}(.+)$",
                line
            )

            if match:

                name = match.group(1)
                model_id = match.group(2)
                size = match.group(3).strip()
                modified = match.group(4).strip()

            else:

                parts = line.split()

                if not parts:
                    continue

                name = parts[0]
                model_id = parts[1] if len(parts) > 1 else ""
                size = parts[2] if len(parts) > 2 else ""
                modified = " ".join(parts[3:])

            models.append(
                {
                    "name": name,
                    "id": model_id,
                    "size": size,
                    "modified": modified,
                    "description": DESCRIPTIONS.get(
                        name,
                        "Local Ollama model"
                    )
                }
            )

        return models


    # ========================================================
    # MODEL UI
    # ========================================================

    def display_models(self, models):

        for widget in self.models_frame.winfo_children():
            widget.destroy()

        self.loaded_models = models
        self.model_rows = {}

        if not models:

            ctk.CTkLabel(
                self.models_frame,
                text="No installed models found.",
                text_color="gray"
            ).pack(
                pady=30
            )

            return

        for model in models:

            self.create_model_row(model)


    def create_model_row(self, model):
        frame = ctk.CTkFrame(self.models_frame, corner_radius=5)
        frame.pack(fill="x", pady=1)

        radio = ctk.CTkRadioButton(
            frame,
            text=model["name"],
            value=model["name"],
            variable=self.model_selection_var,
            command=lambda n=model["name"]: self.select_model(n),
            font=ctk.CTkFont(size=10),
            height=24,
            width=20
        )
        radio.pack(side="left", padx=(7, 4), pady=2)

        size_label = ctk.CTkLabel(
            frame,
            text=model["size"],
            text_color="gray",
            font=ctk.CTkFont(size=9)
        )
        size_label.pack(side="right", padx=7)

        self.model_rows[model["name"]] = {
            "frame": frame,
            "radio": radio
        }

    def select_model(self, model_name):

        self.current_selected_model = model_name
        self.model_selection_var.set(model_name)

        self.selected_label.configure(
            text=f"Выбрано: {model_name}"
        )

        self.btn_cmd.configure(
            state="normal"
        )

        self.btn_delete.configure(
            state="normal"
        )

        self.btn_unload.configure(
            state="normal"
        )

        self.btn_info.configure(
            state="normal"
        )

        self.btn_test.configure(
            state="normal"
        )


    # ========================================================
    # MODEL INFO
    # ========================================================

    def confirm_model_info(self):

        if not self.current_selected_model:
            return

        if not self.confirm(
            "Model Information",
            f"Get detailed information about:\n\n"
            f"{self.current_selected_model}?"
        ):
            return

        self.run_background(
            lambda: self._model_info_worker(
                self.current_selected_model
            )
        )


    def _model_info_worker(self, model):

        returncode, stdout, stderr = run_command(
            [
                OLLAMA_PATH,
                "show",
                model
            ],
            timeout=30
        )

        if returncode != 0:

            self.worker_queue.put(
                (
                    "error",
                    stderr or "Unable to retrieve information."
                )
            )

            return

        self.worker_queue.put(
            (
                "info",
                stdout
            )
        )


    # ========================================================
    # DELETE MODEL
    # ========================================================

    def confirm_delete_model(self):

        model = self.current_selected_model

        if not model:
            return

        if not self.confirm(
            "Deleting a model",
            f"Delete model?\n\n{model}\n\n"
            "This action cannot be undone."
        ):
            return

        self.run_background(
            lambda: self._delete_model_worker(model)
        )


    def _delete_model_worker(self, model):

        returncode, stdout, stderr = run_command(
            [
                OLLAMA_PATH,
                "rm",
                model
            ],
            timeout=600
        )

        if returncode != 0:

            self.worker_queue.put(
                (
                    "error",
                    stderr or f"Failed to delete {model}."
                )
            )

            return

        self.worker_queue.put(
            (
                "message",
                f"Model removed:\n{model}"
            )
        )

        self.worker_queue.put(
            (
                "refresh_after",
                None
            )
        )


    # ========================================================
    # DOWNLOAD MODEL
    # ========================================================

    def confirm_download_model(self):

        dialog = ctk.CTkInputDialog(
            text="Enter the Ollama model name:\n\nFor example:\nqwen2.5:3b",
            title="Скачать модель"
        )

        model = dialog.get_input()

        if not model:
            return

        model = model.strip()

        if not model:
            return

        if not self.confirm(
            "Downloading the model",
            f"Download model?\n\n{model}\n\n"
            "The operation may take a long time "
            "and require several gigabytes of disk space."
        ):
            return

        self.download_progress.set(0)

        self.download_progress_label.configure(
            text=f"Загрузка {model}... 0%"
        )

        self.run_background(
            lambda: self._download_model_worker(model)
        )


    def _download_model_worker(self, model):

        try:

            process = subprocess.Popen(
                [
                    OLLAMA_PATH,
                    "pull",
                    model
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            output = []

            for line in process.stdout:

                line = line.rstrip()

                if line:
                    output.append(line)

                self.worker_queue.put(
                    (
                        "progress",
                        line
                    )
                )

            process.wait()

            if process.returncode != 0:

                self.worker_queue.put(
                    (
                        "error",
                        "\n".join(output[-10:])
                        or "Error downloading model."
                    )
                )

                return

            self.worker_queue.put(
                (
                    "message",
                    f"Model downloaded successfully:\n{model}"
                )
            )

            self.worker_queue.put(
                (
                    "refresh_after",
                    None
                )
            )

        except Exception as e:

            self.worker_queue.put(
                (
                    "error",
                    f"Download error:\n{e}"
                )
            )


    # ========================================================
    # RUN MODEL IN CMD
    # ========================================================

    def confirm_run_cmd(self):

        model = self.current_selected_model

        if not model:
            return

        if not self.confirm(
            "Launching the model",
            f"Open CMD and run the model?\n\n{model}"
        ):
            return

        self.run_model_in_cmd(model)


    def run_model_in_cmd(self, model=None):

        model = model or self.current_selected_model

        if not model:
            return

        try:

            subprocess.Popen(
                [
                    "cmd.exe",
                    "/k",
                    f'"{OLLAMA_PATH}" run "{model}"'
                ]
            )

        except Exception as e:

            self.show_error(
                "Error",
                str(e)
            )


    # ========================================================
    # LOADED MODELS / UNLOAD
    # ========================================================

    def confirm_unload_model(self):

        model = self.current_selected_model

        if not model:
            return

        if not self.confirm(
            "Unloading the model",
            f"Unload model from RAM/VRAM?\n\n{model}"
        ):
            return

        self.run_background(
            lambda: self._unload_model_worker(model)
        )


    def _unload_model_worker(self, model):

        returncode, stdout, stderr = run_command(
            [
                OLLAMA_PATH,
                "stop",
                model
            ],
            timeout=60
        )

        if returncode != 0:

            self.worker_queue.put(
                (
                    "error",
                    stderr or f"Failed to upload {model}."
                )
            )

            return

        self.worker_queue.put(
            (
                "message",
                f"Model uploaded:\n{model}"
            )
        )

        self.worker_queue.put(("refresh_loaded", None))


    def _run_cli_text(self, args, timeout=15):
        return run_command([OLLAMA_PATH, *args], timeout=timeout)

    def confirm_show_ps(self):
        if not self.confirm("ollama ps", "Run the command:\n\nollama ps\n\nShow result?"):
            return
        self.run_background(self._show_ps_worker)

    def _show_ps_worker(self):
        returncode, stdout, stderr = self._run_cli_text(["ps"], timeout=10)
        if returncode != 0:
            self.worker_queue.put(("error", stderr or "ollama ps not executed."))
            return
        self.worker_queue.put(("ps_output", stdout or "NAME    ID    SIZE    PROCESSOR    UNTIL\n(no models loaded)"))

    def confirm_show_all_models(self):
        if not self.confirm(
            "List of all models",
            "Run the command:\n\nollama list\n\nShow full terminal output?"
        ):
            return
        self.run_background(self._show_all_models_worker)

    def _show_all_models_worker(self):
        returncode, stdout, stderr = self._run_cli_text(["list"], timeout=20)
        if returncode != 0:
            self.worker_queue.put(("error", stderr or "ollama list not executed."))
            return
        self.worker_queue.put(("all_models_output", stdout or "No models found."))

    def update_loaded_models(self, ps_output=None):
        if ps_output is None:
            if not ollama_api_available():
                self.loaded_models = []
                self.loaded_model_details = ""
                return
            returncode, ps_output, stderr = self._run_cli_text(["ps"], timeout=10)
            if returncode != 0:
                self.loaded_models = []
                self.loaded_model_details = ""
                return

        lines = [line.rstrip() for line in ps_output.splitlines() if line.strip()]
        loaded = []

        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    loaded.append(parts[0])

        self.loaded_models = loaded
        self.loaded_model_details = ps_output.strip()

        if loaded:
            text = "Uploaded model:\n" + "\n".join(loaded)
        else:
            text = "Uploaded model:\n—"

        self.loaded_model_label.configure(text=text)

    def refresh_loaded_models_ui(self):
        if not ollama_api_available():
            self.update_loaded_models("")
            return
        _, stdout, _ = self._run_cli_text(["ps"], timeout=5)
        self.update_loaded_models(stdout or "")


    # ========================================================
    # PROMPT PRESETS
    # ========================================================

    def apply_prompt_preset(self, preset_name=None):

        preset_name = preset_name or self.preset_menu.get()

        prompt = self.prompt_presets.get(preset_name)

        if prompt is None:
            return

        self.prompt_box.delete(
            "1.0",
            "end"
        )

        self.prompt_box.insert(
            "1.0",
            prompt
        )


    # ========================================================
    # SYSTEM MONITOR (без графиков)
    # ========================================================

    def get_gpu_load(self):
        """
        Return a universal Windows GPU utilization estimate (0-100%).

        Uses the built-in GPU Engine performance counter, so it is not tied
        to NVIDIA, AMD or Intel-specific tools. The busiest GPU engine is
        used as the overall load, matching how Windows commonly represents
        GPU utilization in Task Manager.
        """

        try:

            result = subprocess.run(
                [
                    "typeperf",
                    r"\GPU Engine(*)\Utilization Percentage",
                    "-sc",
                    "1",
                    "-y"
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=4,
                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0
                )
            )

            if result.returncode != 0:
                return None

            values = []

            for line in result.stdout.splitlines():

                if not line or line.startswith('"\\'):
                    continue

                # typeperf CSV lines contain a timestamp followed by counters.
                for part in line.split(",")[1:]:

                    part = part.strip().strip('"')

                    try:

                        value = float(
                            part.replace("%", "")
                        )

                        if 0 <= value <= 1000:
                            values.append(value)

                    except ValueError:
                        continue

            if not values:
                return None

            return max(
                0.0,
                min(
                    100.0,
                    max(values)
                )
            )

        except Exception:
            return None


    def system_monitor_loop(self):
        if not self.monitor_running:
            return

        def worker():
            try:
                cpu_load = psutil.cpu_percent(interval=0.05)
                ram = psutil.virtual_memory()
                gpu_load = self.get_gpu_load()

                self.after(
                    0,
                    lambda: self.update_system_monitor(
                        cpu_load,
                        ram.used,
                        ram.total,
                        gpu_load
                    )
                )
            except Exception:
                pass

        threading.Thread(
            target=worker,
            daemon=True,
            name="SystemMonitorWorker"
        ).start()

        self.after(1200, self.system_monitor_loop)

    def update_system_monitor(
        self,
        cpu_load,
        ram_used,
        ram_total,
        gpu_load
    ):
        cpu_percent = max(0.0, min(100.0, float(cpu_load or 0)))
        ram_percent = (
            (ram_used / ram_total) * 100
            if ram_total > 0 else 0
        )
        ram_percent = max(0.0, min(100.0, ram_percent))

        self.cpu_history.append(cpu_percent)
        self.cpu_history = self.cpu_history[-self.max_history:]

        self.ram_history.append(ram_percent)
        self.ram_history = self.ram_history[-self.max_history:]

        self.cpu_header.configure(text=f"CPU  {cpu_percent:.0f}%")
        self.cpu_bar.set(cpu_percent / 100)

        self.ram_header.configure(
            text=f"RAM  {ram_percent:.0f}%  {format_bytes(ram_used)} / {format_bytes(ram_total)}"
        )
        self.ram_bar.set(ram_percent / 100)

        if gpu_load is not None:
            self.gpu_available = True
            gpu_load = max(0.0, min(100.0, float(gpu_load)))
            self.gpu_history.append(gpu_load)
            self.gpu_history = self.gpu_history[-self.max_history:]

            self.gpu_header.configure(text=f"GPU  {gpu_load:.0f}%")
            self.gpu_bar.set(gpu_load / 100)
        else:
            self.gpu_available = False
            self.gpu_header.configure(text="GPU  —")
            self.gpu_bar.set(0)


    # ========================================================
    # MODEL TEST
    # ========================================================

    def confirm_test_model(self):

        model = self.current_selected_model

        if not model:
            return

        prompt = self.prompt_box.get(
            "1.0",
            "end"
        ).strip()

        if not prompt:

            self.show_error(
                "Model test",
                "Enter prompt."
            )

            return

        if not self.confirm(
            "Testing the model",
            f"Run model test?\n\n{model}\n\n"
            "The model can boot into RAM/VRAM."
        ):
            return

        self.response_box.delete(
            "1.0",
            "end"
        )

        self.response_box.insert(
            "end",
            "Generation...\n"
        )

        self.test_stats.configure(
            text="Time: running..."
        )

        self.run_background(
            lambda: self._test_model_worker(
                model,
                prompt
            )
        )


    def _test_model_worker(
        self,
        model,
        prompt
    ):

        try:

            result, wall_time = api_generate(
                model,
                prompt
            )

            response = result.get(
                "response",
                ""
            )

            eval_count = result.get(
                "eval_count",
                0
            )

            eval_duration = result.get(
                "eval_duration",
                0
            )

            prompt_eval_count = result.get(
                "prompt_eval_count",
                0
            )

            total_duration = result.get(
                "total_duration",
                0
            )

            # nanoseconds -> seconds
            eval_seconds = (
                eval_duration / 1_000_000_000
                if eval_duration
                else 0
            )

            if eval_seconds > 0 and eval_count:

                tokens_per_second = (
                    eval_count / eval_seconds
                )

            else:

                tokens_per_second = 0

            self.worker_queue.put(
                (
                    "test_result",
                    {
                        "response": response,
                        "eval_count": eval_count,
                        "prompt_eval_count": prompt_eval_count,
                        "wall_time": wall_time,
                        "eval_seconds": eval_seconds,
                        "tokens_per_second": tokens_per_second,
                        "total_duration": total_duration
                    }
                )
            )

        except urllib.error.HTTPError as e:

            try:

                error_body = e.read().decode(
                    "utf-8",
                    errors="replace"
                )

            except Exception:

                error_body = str(e)

            self.worker_queue.put(
                (
                    "error",
                    f"Ollama API:\n{error_body}"
                )
            )

        except Exception as e:

            self.worker_queue.put(
                (
                    "error",
                    f"Test error:\n{e}"
                )
            )


    # ========================================================
    # BACKGROUND WORKERS
    # ========================================================

    def run_background(self, function):

        if self.operation_running:
            return

        self.operation_running = True

        self.set_buttons_enabled(False)

        def worker_wrapper():

            try:

                function()

            finally:

                # UI is re-enabled only after the actual worker finishes.
                self.worker_queue.put(
                    (
                        "operation_done",
                        None
                    )
                )

        thread = threading.Thread(
            target=worker_wrapper,
            daemon=True,
            name="OllamaManagerWorker"
        )

        thread.start()


    def process_worker_queue(self):

        try:

            while True:

                action, data = self.worker_queue.get_nowait()

                if action == "models":

                    self.display_models(data)

                elif action == "models_error":

                    self.display_models([])

                elif action == "message":

                    if (
                        isinstance(data, str)
                        and data.startswith(
                            "Model downloaded successfully:"
                        )
                    ):

                        self.download_progress.set(1)

                        self.download_progress_label.configure(
                            text="Download complete — 100%"
                        )

                    self.show_info(
                        "Ollama",
                        data
                    )

                elif action == "error":

                    if (
                        isinstance(data, str)
                        and (
                            "downloads" in data.lower()
                            or "download" in data.lower()
                        )
                    ):

                        self.download_progress_label.configure(
                            text="Loading error"
                        )

                    self.show_error(
                        "Error",
                        data
                    )

                elif action == "info":

                    self.show_text_window(
                        "Model Information",
                        data,
                        monospace=False
                    )

                elif action == "ps_output":

                    self.update_loaded_models(data)
                    self.show_text_window("ollama ps", data, monospace=True)

                elif action == "all_models_output":

                    self.show_text_window("List of all models — ollama list", data, monospace=True)

                elif action == "refresh_after":

                    self.after(
                        300,
                        self.refresh_models
                    )

                elif action == "refresh_loaded":

                    self.after(50, self.refresh_loaded_models_ui)

                elif action == "progress":

                    if data:

                        line = str(data).strip()

                        match = re.search(
                            r"(\d{1,3})%",
                            line
                        )

                        if match:

                            percent = max(
                                0,
                                min(
                                    100,
                                    int(match.group(1))
                                )
                            )

                            self.download_progress.set(
                                percent / 100
                            )

                            self.download_progress_label.configure(
                                text=f"Loading... {percent}%"
                            )

                        else:

                            # Ollama sometimes reports status without interest.
                            self.download_progress_label.configure(
                                text=line[-70:]
                            )

                elif action == "operation_done":

                    self.operation_running = False

                    self.set_buttons_enabled(True)

                elif action == "test_result":

                    result = data

                    self.response_box.delete(
                        "1.0",
                        "end"
                    )

                    self.response_box.insert(
                        "1.0",
                        result["response"]
                    )

                    self.test_stats.configure(
                        text=(
                            f"Time: {result['wall_time']:.2f} с    "
                            f"Speed: "
                            f"{result['tokens_per_second']:.2f} tok/s    "
                            f"Tokens: {result['eval_count']}"
                        )
                    )

        except queue.Empty:
            pass

        finally:

            # Queue processing itself does not determine worker completion.
            # The worker sends an explicit operation_done event.
            self.after(
                100,
                self.process_worker_queue
            )


    # ========================================================
    # BUTTON STATE
    # ========================================================

    def set_buttons_enabled(self, enabled):

        state = "normal" if enabled else "disabled"

        for button in [
            self.btn_start,
            self.btn_stop,
            self.btn_force_stop,
            self.btn_restart,
            self.btn_ps,
            self.btn_all_models,
            self.btn_download,
            self.btn_refresh
        ]:

            button.configure(
                state=state
            )

        if self.current_selected_model:

            for button in [
                self.btn_cmd,
                self.btn_delete,
                self.btn_unload,
                self.btn_info,
                self.btn_test
            ]:

                button.configure(
                    state=state
                )


    # ========================================================
    # STATUS MONITOR
    # ========================================================

    def status_loop(self):

        def worker():

            api_running = ollama_api_available()
            process_running = ollama_process_running()

            ps_output = ""
            if api_running:
                _, ps_output, _ = self._run_cli_text(["ps"], timeout=5)

            self.after(
                0,
                lambda:
                self.update_status(
                    api_running,
                    process_running,
                    ps_output
                )
            )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

        self.after(
            2000,
            self.status_loop
        )


    def update_status(
        self,
        api_running,
        process_running,
        ps_output=""
    ):

        if api_running:

            self.status_label.configure(
                text="OLLAMA SERVER IS WORKING",
                text_color="#2ecc71"
            )

        elif process_running:

            self.status_label.configure(
                text="STARTING...",
                text_color="#f1c40f"
            )

        else:

            self.status_label.configure(
                text="STOPPED",
                text_color="#e74c3c"
            )

        self.server_info.configure(
            text=(
                f"API: {'ONLINE' if api_running else 'OFFLINE'}\n"
                f"Process: {'RUNNING' if process_running else 'STOPPED'}"
            )
        )

        if ps_output:
            self.update_loaded_models(ps_output)
        elif not api_running:
            self.update_loaded_models("")


    # ========================================================
    # DIALOGS
    # ========================================================

    def show_info(self, title, message):

        messagebox.showinfo(
            title,
            message,
            parent=self
        )


    def show_error(self, title, message):

        messagebox.showerror(
            title,
            message,
            parent=self
        )


    def show_text_window(self, title, text, monospace=False):

        window = ctk.CTkToplevel(self)
        window.title(title)
        window.geometry("900x860")
        window.minsize(620, 380)
        window.transient(self)

        textbox = ctk.CTkTextbox(
            window,
            font=ctk.CTkFont(
                family="Consolas" if monospace else "Segoe UI",
                size=12
            )
        )
        textbox.pack(fill="both", expand=True, padx=8, pady=8)
        textbox.insert("1.0", text)
        textbox.configure(state="enable")

        ctk.CTkButton(
            window,
            text="Close",
            height=28,
            command=window.destroy,
            font=ctk.CTkFont(size=10)
        ).pack(fill="x", padx=8, pady=(0, 8))


    # ========================================================
    # CLOSE
    # ========================================================

    def on_close(self):

        if self.confirm(
            "Exit",
            "Close Ollama Manager?"
        ):

            self.monitor_running = False
            self.destroy()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    app = OllamaManagerApp()

    app.mainloop()