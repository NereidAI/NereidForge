# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Notebook server routes: workbench CRUD + Jupyter proxy.

CRUD lives under /api/v1/notebook-servers and requires a NF login. The proxy
is mounted at the bare prefix /notebook-servers/{name}/... and forwards to
the notebook's in-cluster service (streamed HTTP + websocket pass-through).
Jupyter's base_url equals that bare prefix, so no path/body rewriting is
needed and the injected JUPYTER_TOKEN stays server-side.
"""
from __future__ import annotations

from typing import Optional

import httpx
from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.helpers import error
from app.clients.k8s import K8sNotebookError
from app.core.auth import require_auth
from app.services.notebooks import (
    NotebookServerService,
    NotebookUnavailable,
)

router = APIRouter()

# Hop-by-hop headers must not be forwarded by a proxy.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
# Tornado enforces host consistency and rejects cross-origin POSTs; the proxy
# is effectively same-origin from Jupyter's point of view.
STRIP_REQ_HEADERS = HOP_BY_HOP | {"origin", "referer"}


class _NsHolder:
    """Process-wide lazy singleton for the notebook service."""

    svc: Optional[NotebookServerService] = None


def _nb_svc() -> NotebookServerService:
    """Return the shared NotebookServerService instance."""
    if _NsHolder.svc is None:
        _NsHolder.svc = NotebookServerService()
    return _NsHolder.svc


def _tenant(request: Request) -> str:
    """Read the tenant header used for namespace resolution."""
    return request.headers.get("X-Tenant-Id", "default") or "default"


class NotebookServerCreate(BaseModel):
    """Create-notebook request body."""

    name: str
    image: str = ""
    cpu: str = "500m"
    memory: str = "1Gi"
    gpus: int = 0
    gpu_type: str = "nvidia"
    pvc_size: str = "10Gi"
    jupyter_token: str = "nereidforge"


@router.get("/notebook-servers", dependencies=[Depends(require_auth)])
async def list_notebook_servers(request: Request):
    """List interactive notebook servers."""
    try:
        items = _nb_svc().list(_tenant(request))
    except NotebookUnavailable as e:
        raise error(501, str(e))
    except K8sNotebookError as e:
        raise error(502, str(e)[:200])
    return {"items": items, "total": len(items)}


@router.post("/notebook-servers", dependencies=[Depends(require_auth)])
async def create_notebook_server(req: NotebookServerCreate, request: Request):
    """Create a notebook server (Notebook CR + workspace PVC)."""
    svc = _nb_svc()
    try:
        return svc.create(
            _tenant(request),
            req.name,
            req.image,
            req.cpu,
            req.memory,
            req.gpus,
            req.gpu_type,
            req.pvc_size,
            req.jupyter_token,
            request.headers.get("X-Username", "anonymous"),
        )
    except NotebookUnavailable as e:
        raise error(501, str(e))
    except ValueError as e:
        raise error(400, str(e))
    except K8sNotebookError as e:
        raise error(502, str(e)[:200])


@router.post("/notebook-servers/{name}/stop", dependencies=[Depends(require_auth)])
async def stop_notebook_server(name: str, request: Request):
    """Pause a notebook (pod parks on sleep; PVC/state kept)."""
    try:
        ok = _nb_svc().stop(_tenant(request), name)
    except NotebookUnavailable as e:
        raise error(501, str(e))
    except K8sNotebookError as e:
        raise error(502, str(e)[:200])
    if not ok:
        raise error(404, "notebook not found")
    return {"stopped": True}


@router.post("/notebook-servers/{name}/start", dependencies=[Depends(require_auth)])
async def start_notebook_server(name: str, request: Request):
    """Resume a paused notebook."""
    try:
        ok = _nb_svc().start(_tenant(request), name)
    except NotebookUnavailable as e:
        raise error(501, str(e))
    except (K8sNotebookError, ValueError) as e:
        raise error(409, str(e)[:200])
    if not ok:
        raise error(404, "notebook not found")
    return {"started": True}


@router.delete("/notebook-servers/{name}", dependencies=[Depends(require_auth)])
async def delete_notebook_server(name: str, request: Request):
    """Delete a notebook server (PVC kept unless ?pvc=true)."""
    pvc = request.query_params.get("pvc", "false").lower() in ("1", "true", "yes")
    try:
        ok = _nb_svc().delete(_tenant(request), name, delete_pvc=pvc)
    except NotebookUnavailable as e:
        raise error(501, str(e))
    except K8sNotebookError as e:
        raise error(502, str(e)[:200])
    if not ok:
        raise error(404, "notebook not found")
    return {"deleted": True}


@router.api_route(
    "/notebook-servers/{name}/{rest_of_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def proxy_notebook_http(name: str, rest_of_path: str, request: Request):
    """Stream-reverse-proxy HTTP into the notebook's Jupyter server.

    The browser path is Jupyter's base_url + resource path and both are
    identical, so the path is forwarded verbatim and redirects/cookies need
    no rewriting.
    """
    try:
        base, token = _nb_svc().proxy_target(_tenant(request), name)
    except NotebookUnavailable as e:
        raise error(501, str(e))
    except K8sNotebookError:
        # Missing CR or cluster error: 404 lets the UI show "not starting?".
        raise error(404, "notebook not found")

    fwd_path = f"/notebook-servers/{name}/{rest_of_path}" if rest_of_path else f"/notebook-servers/{name}/"
    url = f"{base}{fwd_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in STRIP_REQ_HEADERS
    }
    if token:
        # Harmless with token auth disabled, used if an image enforces it.
        headers.setdefault("Authorization", f"Bearer {token}")

    body = await request.body()
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0))
    try:
        upstream_req = client.build_request(request.method, url, headers=headers, content=body)
        upstream = await client.send(upstream_req, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        raise error(503, "notebook not reachable (still starting?)")

    async def _pump():
        # Stream raw (already-encoded) bytes straight through to the browser,
        # closing the upstream client once the body has been consumed.
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    # StreamingResponse requires a header mapping (duplicate keys collapse,
    # last value wins); hop-by-hop framing headers are stripped.
    resp_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP
    }
    return StreamingResponse(
        _pump(),
        status_code=upstream.status_code,
        headers=resp_headers,
    )


@router.websocket("/notebook-servers/{name}/{rest_of_path:path}")
async def proxy_notebook_ws(websocket: WebSocket, name: str, rest_of_path: str):
    """WebSocket pass-through (Jupyter kernel channels, terminals)."""
    await websocket.accept()
    try:
        base, token = _nb_svc().proxy_target(
            websocket.headers.get("X-Tenant-Id", "default") or "default", name
        )
    except (K8sNotebookError, NotebookUnavailable):
        await websocket.close(code=1011)
        return

    import websockets

    fwd_path = f"/notebook-servers/{name}/{rest_of_path}" if rest_of_path else f"/notebook-servers/{name}/"
    sep = "&" if "?" in fwd_path else "?"
    ws_url = f"{base.replace('http://', 'ws://')}{fwd_path}{sep}token={token or ''}"

    try:
        upstream = await websockets.connect(ws_url, max_size=2**22, ping_interval=20)
    except Exception:
        await websocket.close(code=1011)
        return

    async def pump_up() -> None:
        # Browser -> Jupyter.
        try:
            while True:
                data = await websocket.receive()
                if data.get("type") == "websocket.disconnect":
                    break
                if data.get("text") is not None:
                    await upstream.send(data["text"])
                elif data.get("bytes") is not None:
                    await upstream.send(data["bytes"])
        except Exception:
            pass
        finally:
            try:
                await upstream.close()
            except Exception:
                pass

    async def pump_down() -> None:
        # Jupyter -> browser.
        try:
            async for msg in upstream:
                if isinstance(msg, str):
                    await websocket.send_text(msg)
                else:
                    await websocket.send_bytes(msg)
        except Exception:
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    import asyncio

    try:
        await asyncio.gather(pump_up(), pump_down())
    except WebSocketDisconnect:
        return
