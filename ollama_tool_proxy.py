#!/usr/bin/env python3
"""
Proxy local de traducao: ARTEX (OpenAI /v1/chat/completions, nao-stream)
  -> Ollama nativo /api/chat (que parseia <tool_call> em tool_calls estruturado)
  -> de volta para JSON OpenAI chat.completion com tool_calls corretos.

Corrige o /v1 do Ollama que falha em emitir tool_calls estruturado p/ qwen2.5,
mantendo tudo local. Aponte o base_url do profile ollama do ARTEX para ca.
"""
import json, time, uuid, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA = "http://127.0.0.1:11434/api/chat"
LISTEN = ("127.0.0.1", 11435)
UPSTREAM_TIMEOUT = 900  # modelos grandes sao lentos

def log(*a):
    print("[proxy]", *a, file=sys.stderr, flush=True)

def to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                out.append(p.get("text") or p.get("content") or "")
            else:
                out.append(str(p))
        return "".join(out)
    return str(content)

def oai_to_ollama_messages(messages):
    out = []
    id2name = {}
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except Exception:
                        args = {"_raw": args}
                tcs.append({"function": {"name": fn.get("name", ""), "arguments": args or {}}})
                if tc.get("id"):
                    id2name[tc["id"]] = fn.get("name", "")
            out.append({"role": "assistant",
                        "content": to_text(m.get("content")),
                        "tool_calls": tcs})
        elif role == "tool":
            msg = {"role": "tool", "content": to_text(m.get("content"))}
            name = m.get("name") or id2name.get(m.get("tool_call_id"), "")
            if name:
                msg["tool_name"] = name
            out.append(msg)
        else:
            out.append({"role": role, "content": to_text(m.get("content"))})
    return out

def parse_text_toolcall(content):
    if not content:
        return None
    import re
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.S)
    blob = m.group(1) if m else None
    if not blob:
        m2 = re.search(r'(\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})', content, re.S)
        blob = m2.group(1) if m2 else None
    if not blob:
        return None
    try:
        obj = json.loads(blob)
        if "name" in obj:
            return [{"function": {"name": obj["name"], "arguments": obj.get("arguments", {})}}]
    except Exception:
        return None
    return None

def ollama_to_oai(resp, model):
    msg = resp.get("message", {}) or {}
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        fb = parse_text_toolcall(content)
        if fb:
            tool_calls = fb
            content = ""
    oai_tcs = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        oai_tcs.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": args},
        })
    message = {"role": "assistant", "content": content if content else None}
    if oai_tcs:
        message["tool_calls"] = oai_tcs
        finish = "tool_calls"
    else:
        message["content"] = content
        finish = "stop"
    pin = int(resp.get("prompt_eval_count") or 0)
    pout = int(resp.get("eval_count") or 0)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout},
    }

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad json: {e}"})
            return
        model = req.get("model", "qwen14k")
        options = {"temperature": req.get("temperature", 0)}
        if req.get("top_p") is not None:
            options["top_p"] = req["top_p"]
        options["num_ctx"] = 24576  # casa com o Modelfile
        ollama_req = {
            "model": model,
            "messages": oai_to_ollama_messages(req.get("messages", [])),
            "stream": False,
            "options": options,
            # Desliga o raciocinio do qwen3 (modo thinking): corta as cadeias longas
            # de <think> que deixavam cada chamada lenta. Para tarefas de execucao
            # (disparar curl, ler resposta) o ganho de velocidade compensa muito.
            "think": False,
        }
        if req.get("tools"):
            ollama_req["tools"] = req["tools"]
        data = json.dumps(ollama_req).encode()
        try:
            r = urllib.request.Request(OLLAMA, data=data,
                                       headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=UPSTREAM_TIMEOUT) as resp:
                raw = json.loads(resp.read())
        except Exception as e:
            log("upstream error:", repr(e))
            self._send(502, {"error": f"ollama upstream: {e}"})
            return
        out = ollama_to_oai(raw, model)
        ntc = len(out["choices"][0]["message"].get("tool_calls") or [])
        log(f"{model} -> finish={out['choices'][0]['finish_reason']} tool_calls={ntc} "
            f"in={out['usage']['prompt_tokens']} out={out['usage']['completion_tokens']}")
        self._send(200, out)

if __name__ == "__main__":
    srv = ThreadingHTTPServer(LISTEN, Handler)
    log(f"listening on http://{LISTEN[0]}:{LISTEN[1]}  ->  {OLLAMA}")
    srv.serve_forever()
