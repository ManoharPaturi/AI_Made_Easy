"""AI Made Easy web server: REST + WebSocket API over ``core.api`` and the browser UI.

``create_app()`` builds the FastAPI application; ``aime web`` serves it with
uvicorn. Every endpoint lives under ``/api``; the built frontend (``static/``)
is served at ``/``. Projects are stored as JSON under ``$AIME_HOME/projects``;
runs, sweeps and the model registry use the same stores as the desktop app.

Access control: when ``AIME_WEB_TOKEN`` (or ``create_app(token=...)``) is set,
every request needs ``Authorization: Bearer <token>`` (WebSockets:
``?token=<token>``). Without it the server is meant for localhost only.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.background import BackgroundTask

from ai_made_easy import __version__
from ai_made_easy.core import api
from ai_made_easy.core.paths import subdir

STATIC = Path(__file__).parent / "static"
_NAME = re.compile(r"^[\w][\w .-]{0,99}$")


def _graph_body(payload: dict) -> dict:
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        raise HTTPException(400, "body must contain a 'graph' object")
    return graph


def _project_file(root: Path, name: str) -> Path:
    if not _NAME.match(name) or name.endswith("."):
        raise HTTPException(400, "project names may use letters, digits, spaces, '_', '-' and '.'")
    return root / f"{name}.json"


def _zip_dir(folder: Path) -> Path:
    archive = shutil.make_archive(str(folder), "zip", root_dir=folder)
    return Path(archive)


def _data_url(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def create_app(*, token: str | None = None, projects_dir: str | Path | None = None) -> FastAPI:
    token = (token if token is not None else os.environ.get("AIME_WEB_TOKEN")) or None
    projects = Path(projects_dir) if projects_dir else subdir("projects")
    projects.mkdir(parents=True, exist_ok=True)

    def authorize(request: Request) -> None:
        if token is None:
            return
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(supplied, token):
            raise HTTPException(401, "missing or invalid access token")

    app = FastAPI(title="AI Made Easy", version=__version__, docs_url="/api/docs",
                  openapi_url="/api/openapi.json", redoc_url=None)
    guard = [Depends(authorize)]

    @app.exception_handler(api.ApiError)
    async def _api_error(_request: Request, exc: api.ApiError):  # noqa: ANN202
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def _not_found(_request: Request, exc: KeyError):  # noqa: ANN202
        return JSONResponse({"detail": str(exc.args[0]) if exc.args else "not found"},
                            status_code=404)

    @app.exception_handler(ValueError)
    async def _bad_value(_request: Request, exc: ValueError):  # noqa: ANN202
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # ------------------------------------------------------------------ info
    @app.get("/api/info")
    def info() -> dict:
        from ai_made_easy.core.registry import get_registry

        return {"version": __version__, "auth": token is not None,
                "blocks": len(get_registry().all())}

    # ------------------------------------------------------------- catalog
    @app.get("/api/blocks", dependencies=guard)
    def blocks(category: str | None = None) -> dict:
        return api.list_blocks(category)

    @app.get("/api/families", dependencies=guard)
    def families() -> dict:
        return api.list_families()

    @app.get("/api/tasks", dependencies=guard)
    def tasks(family: str | None = None) -> dict:
        return api.list_tasks(family)

    @app.post("/api/describe", dependencies=guard)
    def describe(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.describe_design(_graph_body(payload))

    @app.get("/api/samples", dependencies=guard)
    def samples() -> dict:
        return api.list_samples()

    @app.get("/api/samples/{name}", dependencies=guard)
    def sample(name: str) -> dict:
        from ai_made_easy.core.migrate import migrate

        return migrate(api.read_sample(name))

    # --------------------------------------------------------------- graph
    @app.post("/api/validate", dependencies=guard)
    def validate(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.validate(_graph_body(payload))

    @app.post("/api/fix", dependencies=guard)
    def fix(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.apply_fix(_graph_body(payload), int(payload.get("index", -1)))

    @app.get("/api/targets", dependencies=guard)
    def targets() -> dict:
        return api.targets()

    @app.post("/api/generate", dependencies=guard)
    def generate(payload: dict = Body(...)) -> dict:  # noqa: B008
        try:
            code = api.generate(_graph_body(payload), str(payload.get("target", "pytorch_model")))
        except api.ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — codegen errors are user-facing
            raise HTTPException(400, str(exc)) from exc
        return {"code": code}

    @app.post("/api/summary", dependencies=guard)
    def summary(payload: dict = Body(...)) -> dict:  # noqa: B008
        try:
            return api.summarize(_graph_body(payload))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/expand", dependencies=guard)
    def expand(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.expand(_graph_body(payload), str(payload.get("node_id", "")))

    # ------------------------------------------------------------ projects
    @app.get("/api/projects", dependencies=guard)
    def list_projects() -> dict:
        rows = [{"name": p.stem, "modified": p.stat().st_mtime}
                for p in sorted(projects.glob("*.json"), key=lambda p: -p.stat().st_mtime)]
        return {"projects": rows}

    @app.get("/api/projects/{name}", dependencies=guard)
    def get_project(name: str) -> dict:
        from ai_made_easy.core.migrate import migrate

        file = _project_file(projects, name)
        if not file.exists():
            raise HTTPException(404, f"no project named {name!r}")
        return migrate(json.loads(file.read_text()))

    @app.put("/api/projects/{name}", dependencies=guard)
    def put_project(name: str, graph: dict = Body(...)) -> dict:  # noqa: B008
        from ai_made_easy.core.graph import Graph

        file = _project_file(projects, name)
        try:
            Graph.from_dict(graph)
        except Exception as exc:  # noqa: BLE001 — refuse unreadable projects
            raise HTTPException(400, f"not a valid project: {exc}") from exc
        graph = {**graph, "name": name}
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(graph, indent=1))
        tmp.replace(file)
        return {"saved": name}

    @app.delete("/api/projects/{name}", dependencies=guard)
    def delete_project(name: str) -> dict:
        file = _project_file(projects, name)
        if not file.exists():
            raise HTTPException(404, f"no project named {name!r}")
        file.unlink()
        return {"deleted": name}

    # ---------------------------------------------------------------- runs
    @app.post("/api/runs", dependencies=guard)
    def start_run(payload: dict = Body(...)) -> dict:  # noqa: B008
        try:
            return api.start_training(_graph_body(payload), str(payload.get("framework", "auto")),
                                      str(payload.get("project", "")),
                                      tags=payload.get("tags"))
        except api.ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — e.g. codegen errors
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/runs", dependencies=guard)
    def runs(project: str | None = None, parent: str | None = None) -> dict:
        return api.list_runs(project or None, parent or None)

    @app.post("/api/runs/compare", dependencies=guard)
    def compare(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.compare_runs([str(r) for r in payload.get("run_ids", [])])

    @app.get("/api/runs/{run_id}", dependencies=guard)
    def run(run_id: str) -> dict:
        return api.get_run(run_id)

    @app.get("/api/runs/{run_id}/metrics", dependencies=guard)
    def metrics(run_id: str) -> dict:
        return {"run_id": run_id, "epochs": api.manager().history.epochs(run_id)}

    @app.post("/api/runs/{run_id}/stop", dependencies=guard)
    def stop(run_id: str) -> dict:
        return api.stop_run(run_id)

    @app.patch("/api/runs/{run_id}", dependencies=guard)
    def update(run_id: str, payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.update_run(run_id, tags=payload.get("tags"), note=payload.get("note"))

    @app.delete("/api/runs/{run_id}", dependencies=guard)
    def delete_run(run_id: str) -> dict:
        return api.delete_run(run_id)

    @app.websocket("/api/runs/{run_id}/events")
    async def run_events(ws: WebSocket, run_id: str, token_q: str = Query("", alias="token")):
        if token is not None and not secrets.compare_digest(token_q, token):
            await ws.close(code=4401)
            return
        mgr = api.manager()
        try:
            mgr.history.get(run_id)
        except KeyError:
            await ws.close(code=4404)
            return
        await ws.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def listener(rid: str, event: dict) -> None:
            if rid == run_id:
                loop.call_soon_threadsafe(queue.put_nowait, event)

        mgr.subscribe(listener)
        try:
            for epoch in mgr.history.epochs(run_id):  # replay what happened so far
                await ws.send_json({"type": "epoch", **epoch})
            while True:
                record = mgr.history.get(run_id)
                if record.status != "running" and queue.empty():
                    await ws.send_json({"type": "status", "status": record.status,
                                        "final_metrics": record.final_metrics})
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=2.0)
                except TimeoutError:
                    if not mgr.is_live(run_id):
                        await asyncio.sleep(0.3)  # let the history catch up
                    continue
                await ws.send_json(event)
                if event.get("type") == "done":
                    await asyncio.sleep(0.3)  # finalize writes the status after "done"
        except WebSocketDisconnect:
            pass
        finally:
            mgr.unsubscribe(listener)
            try:
                await ws.close()
            except RuntimeError:
                pass

    # -------------------------------------------------------------- sweeps
    @app.post("/api/sweeps/params", dependencies=guard)
    def sweep_params(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.sweepable_params(_graph_body(payload))

    @app.post("/api/sweeps", dependencies=guard)
    def start_sweep(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.start_sweep(_graph_body(payload), dict(payload.get("spec") or {}),
                               str(payload.get("project", "")))

    @app.get("/api/sweeps", dependencies=guard)
    def sweeps(project: str | None = None) -> dict:
        return api.list_sweeps(project or None)

    @app.get("/api/sweeps/{sweep_id}", dependencies=guard)
    def sweep(sweep_id: str) -> dict:
        data = api.get_sweep(sweep_id)
        data.pop("graph", None)
        return data

    @app.post("/api/sweeps/{sweep_id}/stop", dependencies=guard)
    def stop_sweep(sweep_id: str) -> dict:
        return api.stop_sweep(sweep_id)

    @app.get("/api/sweeps/{sweep_id}/best", dependencies=guard)
    def sweep_best(sweep_id: str) -> dict:
        return {"graph": api.sweep_best_graph(sweep_id)}

    # -------------------------------------------------------------- deploy
    def _formats(framework: str) -> list[dict]:
        return api.deploy_formats().get(framework, [])

    @app.get("/api/runs/{run_id}/formats", dependencies=guard)
    def run_formats(run_id: str) -> dict:
        return {"formats": _formats(api.manager().history.get(run_id).framework)}

    def _package_response(build) -> FileResponse:  # noqa: ANN001
        work = Path(tempfile.mkdtemp(prefix="aime_package_"))
        try:
            result = build(str(work / "package"))
            archive = _zip_dir(Path(result["path"]))
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
        name = f"{result['metadata'].get('name', 'model')}-serving.zip"
        return FileResponse(archive, filename=name, media_type="application/zip",
                            background=BackgroundTask(shutil.rmtree, work, True))

    def _split(formats: str) -> list[str]:
        return [f.strip() for f in formats.split(",") if f.strip()]

    @app.get("/api/runs/{run_id}/deploy", dependencies=guard)
    def deploy_run(run_id: str, formats: str = "") -> FileResponse:
        return _package_response(lambda out: api.deploy_run(run_id, out, _split(formats)))

    @app.get("/api/models", dependencies=guard)
    def models() -> dict:
        return api.list_models()

    @app.post("/api/models", dependencies=guard)
    def register(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.register_model(str(payload.get("run_id", "")), str(payload.get("name", "")),
                                  str(payload.get("description", "")))

    @app.patch("/api/models/{name}/{version}", dependencies=guard)
    def set_stage(name: str, version: int, payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.set_model_stage(name, version, str(payload.get("stage", "")))

    @app.delete("/api/models/{name}/{version}", dependencies=guard)
    def delete_model(name: str, version: int) -> dict:
        return api.delete_model(name, version)

    @app.get("/api/models/{name}/{version}/deploy", dependencies=guard)
    def deploy_model(name: str, version: str, formats: str = "") -> FileResponse:
        return _package_response(lambda out: api.deploy_model(name, version, out,
                                                              _split(formats)))

    # -------------------------------------------------------------- import
    @app.post("/api/import", dependencies=guard)
    async def import_model(kind: str = Form(...), file: UploadFile = File(...),  # noqa: B008
                           attr: str = Form(""), input_shape: str = Form(""),
                           dtype: str = Form("float32"), kwargs: str = Form("{}")) -> dict:
        work = Path(tempfile.mkdtemp(prefix="aime_upload_"))
        try:
            target = work / Path(file.filename or "model").name
            target.write_bytes(await file.read())
            shape = [int(d) for d in re.split(r"[,x\s]+", input_shape.strip("()[] ")) if d] or None
            return await asyncio.to_thread(
                api.import_model, kind, str(target), attr, shape, dtype,
                json.loads(kwargs or "{}"))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # ---------------------------------------------------------------- data
    @app.post("/api/data/profile", dependencies=guard)
    def data_profile(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.profile_data(_graph_body(payload), payload.get("node_id") or None)

    @app.post("/api/data/issues", dependencies=guard)
    def data_issues(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.data_issues(_graph_body(payload))

    @app.post("/api/data/split", dependencies=guard)
    def data_split(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.split_preview(_graph_body(payload))

    @app.post("/api/data/augment", dependencies=guard)
    def data_augment(payload: dict = Body(...)) -> dict:  # noqa: B008
        work = tempfile.mkdtemp(prefix="aime_augment_")
        try:
            result: dict[str, Any] = api.augmentation_preview(
                _graph_body(payload), work, images=int(payload.get("images", 4)),
                variants=int(payload.get("variants", 6)))
            for image in result["images"]:  # inline the PNGs: the folder is temporary
                image["original"] = _data_url(image["original"])
                image["eval"] = _data_url(image["eval"])
                image["train"] = [_data_url(p) for p in image["train"]]
                image["source"] = Path(image["source"]).parent.name + "/" + \
                    Path(image["source"]).name
            return result
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # ------------------------------------------------------------ frontend
    @app.get("/api/{path:path}", include_in_schema=False)
    def api_404(path: str) -> None:
        raise HTTPException(404, f"no endpoint /api/{path}")

    index = STATIC / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):  # noqa: ANN202
        if index.exists():
            candidate = (STATIC / path).resolve()
            if path and candidate.is_file() and STATIC.resolve() in candidate.parents:
                # file names under assets/ carry a content hash: cache them for good
                immutable = candidate.parent.name == "assets"
                return FileResponse(candidate, headers={"Cache-Control": (
                    "public, max-age=31536000, immutable" if immutable else "no-cache")})
            # never cache the page itself, or an upgrade leaves it pointing at old assets
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>AI Made Easy API</h1><p>The web UI is not built. Run "
                            "<code>npm run build</code> in <code>web/</code>, or use the API "
                            "at <a href='/api/docs'>/api/docs</a>.</p>")

    return app


def main(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("AIME_WEB_TOKEN"):
        print("warning: serving on a public interface without AIME_WEB_TOKEN — anyone who can "
              "reach this port can train models and read your files")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
