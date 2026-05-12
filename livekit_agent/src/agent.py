import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import cli

ROOT = Path(__file__).resolve().parents[2]
# Ensure the project root (containing `src/`) is importable when running from this package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load LiveKit/Ollama/Piper env vars from project root.
load_dotenv(ROOT / ".env.local")

from src.voice_server import server  # noqa: E402


if __name__ == "__main__":
    import time
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from pathlib import Path

    # Lightweight health endpoint for external monitors (GET /healthz -> 200 OK).
    class _HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/healthz":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format: str, *args) -> None:
            return  # silence default logging

    def _start_health_server(port: int) -> None:
        try:
            httpd = HTTPServer(("0.0.0.0", port), _HealthHandler)
        except Exception as e:
            print(f"[AGENT] health server failed to start on {port}: {e!r}")
            return

        def _serve() -> None:
            try:
                httpd.serve_forever()
            except Exception as e:
                print(f"[AGENT] health server stopped: {e!r}")

        threading.Thread(target=_serve, daemon=True).start()
        print(f"[AGENT] health server listening on 0.0.0.0:{port} (GET /healthz)")

    try:
        health_port = int(os.getenv("AGENT_HEALTH_PORT", "9090"))
        _start_health_server(health_port)
    except Exception:
        pass

    backoff = 0
    crash_flag = Path(__file__).resolve().parents[2] / "data" / "crash_flag.txt"
    crash_flag.parent.mkdir(exist_ok=True)
    while True:
        try:
            cli.run_app(server)
            break  # normal exit
        except KeyboardInterrupt:
            break
        except Exception as e:
            try:
                crash_flag.write_text("1", encoding="utf-8")
            except Exception:
                pass
            backoff = min(backoff + 2, 10)
            print(f"[AGENT] run_app error: {e!r}; retrying in {backoff}s")
            time.sleep(backoff)
