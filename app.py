import json, os, re
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.getenv("PFMEA_DATA", os.path.join(BASE_DIR, "pfmea-data.json"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")

app = FastAPI(title="PFMEA AIAG-VDA Practical Advisor v3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GuidedRequest(BaseModel):
    failure_effect: Optional[str] = ""
    occurrence_evidence: Optional[str] = ""
    detection_control: Optional[str] = ""
    current_S: Optional[int] = None
    current_O: Optional[int] = None
    current_D: Optional[int] = None
    question: Optional[str] = ""

class AdviceRequest(BaseModel):
    case_text: str

def load_db() -> Dict[str, List[Dict[str, Any]]]:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

DB = load_db()
AP_MAP: Dict[str, str] = {}
for r in DB.get("ap table", []):
    try:
        S, O, D = int(r.get("S")), int(r.get("O")), int(r.get("D"))
        AP_MAP[f"{S}-{O}-{D}"] = str(r.get("AP", "")).strip()
    except Exception:
        pass

def ap_value(S: Optional[int], O: Optional[int], D: Optional[int]) -> Tuple[Optional[str], Optional[int]]:
    if all(isinstance(x, int) and 1 <= x <= 10 for x in [S, O, D]):
        return AP_MAP.get(f"{S}-{O}-{D}", "NA"), S * O * D
    return None, None

def clean(t: Optional[str]) -> str:
    return (t or "").strip().lower()

def has_any(t: str, words: List[str]) -> bool:
    return any(w in t for w in words)

def years_in_text(t: str) -> int:
    m = re.search(r"(\d+)\s*(años|ano|years|year)", t)
    return int(m.group(1)) if m else 0

def source_row(table: str, score: Optional[int]) -> Dict[str, str]:
    if not score:
        return {}
    rows = DB.get(table, [])
    score_s = str(score)
    for i, r in enumerate(rows):
        txt = json.dumps(r, ensure_ascii=False).lower()
        # intenta localizar fila por score exacto en las columnas reales
        if f'"{score_s}"' in txt or f": {score}" in txt:
            short = txt[:260].replace("\\n", " ")
            return {"source": table, "ref": f"row {i}", "snippet": short}
    return {"source": table, "ref": f"score {score}", "snippet": "score taken from approved table family"}

def manual_score(v: Optional[int]) -> Optional[int]:
    return v if isinstance(v, int) and 1 <= v <= 10 else None

# ---- Rating engines: practical plant logic. They do not replace the table; they choose the closest table family. ----
def rate_severity(text: str) -> Dict[str, Any]:
    t = clean(text)
    if not t:
        return {"value": None, "why": "Falta el efecto del fallo.", "why_not_lower": "Severity no se baja por buen control; depende del efecto para cliente/usuario/regulación.", "evidence_to_reduce": "Define qué ocurre si la pieza mala escapa: seguridad, pérdida de función, montaje, scrap interno.", "questions": ["¿Qué pasa si el fallo llega al cliente o al usuario final?"]}

    no_safety = has_any(t, ["sin riesgo", "no seguridad", "sin riesgo de seguridad", "no safety"])
    if has_any(t, ["seguridad", "safety", "lesión", "lesion", "reglament", "legal", "homolog", "freno no funciona", "brake failure"]) and not no_safety:
        return {"value": 10, "why": "El texto menciona seguridad/regulación. En PFMEA eso se trata como severidad máxima o muy alta.", "why_not_lower": "No se puede bajar S por tener buena detección. Si el efecto es seguridad/regulatorio, la consecuencia manda.", "evidence_to_reduce": "Solo podría bajar si el equipo confirma que el efecto real no afecta seguridad/regulación.", "questions": []}
    if has_any(t, ["pérdida de función", "perdida de funcion", "loss of function", "función", "funcion", "vehículo", "vehiculo", "freno", "brake", "cliente", "customer"]) and no_safety:
        return {"value": 8, "why": "Hay pérdida de función en vehículo/cliente, pero indicas que no hay riesgo de seguridad.", "why_not_lower": "No lo bajo más porque una pérdida de función para cliente es consecuencia seria aunque no sea safety.", "evidence_to_reduce": "Demostrar que el efecto queda limitado a molestia menor, retrabajo o impacto interno sin pérdida funcional.", "questions": []}
    if has_any(t, ["pérdida de función", "perdida de funcion", "loss of function", "función", "funcion", "vehículo", "vehiculo", "freno", "brake", "cliente", "customer"]):
        return {"value": 9, "why": "Hay pérdida de función para cliente/vehículo y no queda claro si existe o no riesgo de seguridad.", "why_not_lower": "No bajo a S8 o menor hasta confirmar explícitamente que no hay safety/regulación.", "evidence_to_reduce": "Confirmación técnica/documentada de que el fallo no compromete seguridad, normativa ni función principal.", "questions": ["¿Puedes confirmar si el efecto tiene riesgo de seguridad o regulación?"]}
    if has_any(t, ["montaje", "assembly", "parada línea", "parada linea", "line stop", "no monta"]):
        return {"value": 7, "why": "El efecto parece impacto de montaje/proceso cliente, no safety.", "why_not_lower": "No bajo más si puede parar línea o impedir montaje.", "evidence_to_reduce": "Evidencia de que solo genera retrabajo menor sin parar línea ni afectar cliente final.", "questions": []}
    if has_any(t, ["scrap", "rechazo interno", "interno", "retrabajo", "selección", "seleccion"]):
        return {"value": 5, "why": "El efecto descrito parece interno: scrap/retrabajo/selección.", "why_not_lower": "No bajo más sin saber si puede escapar a cliente.", "evidence_to_reduce": "Confirmar contención interna robusta y sin impacto cliente.", "questions": []}
    return {"value": None, "why": "El efecto no está descrito con suficiente claridad.", "why_not_lower": "Sin efecto claro, no hay base para asignar Severity.", "evidence_to_reduce": "Describe consecuencia concreta, no el defecto: qué le pasa al cliente/vehículo/proceso.", "questions": ["¿El fallo afecta seguridad, regulación, función del vehículo, montaje cliente o solo scrap interno?"]}

def rate_occurrence(text: str) -> Dict[str, Any]:
    t = clean(text)
    if not t:
        return {"value": None, "why": "Falta evidencia histórica del proceso.", "why_not_lower": "Occurrence no se baja por deseo; necesita histórico, PPM, reclamaciones, capacidad o estabilidad.", "evidence_to_reduce": "Años de producción, PPM, reclamaciones, incidencias internas, Cpk/Ppk y cambios recientes.", "questions": ["¿Cuántos años lleva en producción y qué PPM/reclamaciones/incidencias tiene?"]}

    y = years_in_text(t)
    no_claims = has_any(t, ["sin reclam", "cero reclam", "0 reclam", "no claims", "zero claims", "ninguna reclam"])
    ppm0 = has_any(t, ["ppm = 0", "ppm=0", "ppm 0", "cero ppm"])
    stable = has_any(t, ["estable", "stable", "validado", "validated", "capaz", "capacidad", "cpk", "ppk"])
    new = has_any(t, ["nuevo", "new", "sop", "sin histórico", "sin historico", "no histórico", "no historico", "lanzamiento"])
    incidents = has_any(t, ["incidencia", "desviación", "desviacion", "scrap", "retraba", "retrabajo", "problema", "reclam", "ppm alto", "varias"])

    if new and not y:
        return {"value": 8, "why": "Proceso nuevo o sin histórico: la ocurrencia no puede considerarse baja.", "why_not_lower": "No hay serie histórica que demuestre estabilidad.", "evidence_to_reduce": "Producción real con datos: PPM, scrap, capacidad y ausencia de incidencias durante un periodo suficiente.", "questions": []}
    if incidents and not no_claims:
        return {"value": 7, "why": "Hay incidencias/desviaciones/reclamaciones; eso sube la ocurrencia.", "why_not_lower": "No bajo O mientras existan problemas recientes o sin cierre eficaz.", "evidence_to_reduce": "Acciones cerradas, tendencia estable, PPM reducido y capacidad demostrada.", "questions": []}
    if (y >= 5 and (no_claims or ppm0) and stable):
        return {"value": 2, "why": "Histórico largo, sin reclamaciones/PPM cero y proceso estable.", "why_not_lower": "No bajo a O1 salvo evidencia extremadamente robusta y repetible según tabla aprobada.", "evidence_to_reduce": "Añadir capacidad sólida, tendencia PPM, lessons learned y ausencia de cambios de proceso/material.", "questions": []}
    if (y >= 2 and no_claims and stable):
        return {"value": 3, "why": "Hay varios años de producción estable sin reclamaciones.", "why_not_lower": "No bajo más si faltan PPM/capacidad detallada o evidencia de largo plazo superior.", "evidence_to_reduce": "PPM cuantificado, Cpk/Ppk, estabilidad por familia y confirmación sin cambios recientes.", "questions": []}
    if (y >= 1 and no_claims):
        return {"value": 4, "why": "Hay al menos un año sin reclamaciones, pero la evidencia aún es limitada.", "why_not_lower": "No bajo más sin estabilidad/capacidad/PPM documentados.", "evidence_to_reduce": "Añadir Cpk/Ppk, PPM, scrap interno y tendencia de más tiempo.", "questions": []}
    if no_claims:
        return {"value": 5, "why": "Dices que no hay reclamaciones, pero falta tiempo, PPM o capacidad.", "why_not_lower": "Sin duración ni datos de proceso, no se justifica O bajo.", "evidence_to_reduce": "Indica años, volumen, PPM, scrap interno y capacidad.", "questions": ["¿Durante cuánto tiempo/volumen no ha habido reclamaciones? ¿Tienes PPM o Cpk/Ppk?"]}
    return {"value": None, "why": "La evidencia de ocurrencia no es suficiente para escoger O.", "why_not_lower": "Sin histórico no se puede asumir baja ocurrencia.", "evidence_to_reduce": "Aporta años, volumen, PPM, reclamaciones, scrap/incidencias y capacidad.", "questions": ["¿Hay reclamaciones, PPM, incidencias internas o datos de capacidad?"]}

def rate_detection(text: str) -> Dict[str, Any]:
    t = clean(text)
    if not t:
        return {"value": None, "why": "Falta método de detección/control.", "why_not_lower": "Detection requiere método, frecuencia, MSA y reacción.", "evidence_to_reduce": "Describe control: 100%/muestreo, automático/manual, frecuencia, MSA y plan de reacción.", "questions": ["¿Qué control existe, con qué frecuencia, y hay MSA/reacción definida?"]}

    validated = has_any(t, ["validado", "validated", "msa", "r&r", "grr", "trazabilidad", "traceability", "registros"])
    auto = has_any(t, ["automático", "automatic", "cámara", "camera", "vision", "sensor", "torque monitoring", "monitoring"])
    reject = has_any(t, ["rechazo automático", "rechazo automatico", "auto reject", "bloqueo", "interlock", "parada", "stop"])
    hundred = has_any(t, ["100%", "100 %", "todas", "all parts", "cada pieza"])
    poke = has_any(t, ["poka", "yoke", "ptc", "previene", "prevents", "imposible montar"])
    micro = has_any(t, ["micrómetro", "micrometer", "micrometro", "calibre", "gauge"])
    cmm = has_any(t, ["cmm", "tridimensional"])
    visual = has_any(t, ["visual", "operario", "operator", "humano"])
    no_evidence = has_any(t, ["sin evidencia", "no evidencia", "sin estándar", "sin estandar", "sin msa", "no msa"])
    daily = has_any(t, ["diaria", "diario", "daily", "una vez al día", "1 vez al día"])
    every2h = has_any(t, ["2h", "2 h", "2 horas", "cada 2"])
    every4h = has_any(t, ["4h", "4 h", "4 horas", "cada 4"])

    if poke and validated:
        return {"value": 1, "why": "El control parece preventivo/poka-yoke validado: evita o bloquea el fallo antes del escape.", "why_not_lower": "D1 ya es el nivel más bajo.", "evidence_to_reduce": "Mantener validación, prueba de fallo, reacción y auditorías periódicas.", "questions": []}
    if hundred and auto and reject and validated:
        return {"value": 2, "why": "Control 100% automático, con rechazo/bloqueo, validado y trazable.", "why_not_lower": "No bajo a D1 porque detecta/rechaza; no necesariamente previene la generación del defecto.", "evidence_to_reduce": "Para D1 debería ser prevención o poka-yoke que impida producir/enviar la pieza mala.", "questions": []}
    if hundred and auto and validated:
        return {"value": 3, "why": "Control automático 100% validado, pero no queda claro si bloquea/rechaza automáticamente.", "why_not_lower": "No bajo a D2 sin evidencia de rechazo automático y reacción robusta.", "evidence_to_reduce": "Demostrar rechazo automático, bloqueo de lote, trazabilidad y prueba de fallos.", "questions": []}
    if auto and validated:
        return {"value": 4, "why": "Hay detección automática validada, pero no está claro que sea 100% o con bloqueo robusto.", "why_not_lower": "No bajo más sin confirmar cobertura 100%, rechazo automático y reacción.", "evidence_to_reduce": "Confirmar 100%, estudios de falsa aceptación/rechazo y plan de reacción.", "questions": []}
    if micro and every2h and validated:
        return {"value": 6, "why": "Micrómetro cada 2h con MSA válido: buen control manual, pero es muestreo, no 100%.", "why_not_lower": "No bajo más porque pueden escaparse piezas entre controles y depende de disciplina de reacción.", "evidence_to_reduce": "Aumentar frecuencia, control 100%, automatizar, SPC con reacción clara o poka-yoke.", "questions": []}
    if micro and validated:
        return {"value": 7, "why": "Control manual con instrumento y MSA, pero falta frecuencia o cobertura clara.", "why_not_lower": "No bajo más sin frecuencia, plan de reacción y evidencia de eficacia.", "evidence_to_reduce": "Indicar frecuencia, reacción, registros, GRR y cobertura del defecto.", "questions": ["¿Cada cuánto se mide y qué reacción hay si aparece una pieza fuera de especificación?"]}
    if cmm and daily:
        return {"value": 7, "why": "CMM diaria: medición fiable, pero frecuencia baja para detectar escapes entre controles.", "why_not_lower": "No bajo más porque no es control en línea ni 100%.", "evidence_to_reduce": "Aumentar frecuencia, usar SPC, control en línea o 100% automático.", "questions": []}
    if visual and (every4h or no_evidence):
        return {"value": 9, "why": "Control visual manual y débil; además falta estándar/evidencia.", "why_not_lower": "No bajo más porque depende del operario y la detectabilidad no está demostrada.", "evidence_to_reduce": "Estandarizar defecto visual, formar, validar detección, aumentar frecuencia o automatizar.", "questions": []}
    if visual:
        return {"value": 8, "why": "Control visual/manual: detectabilidad limitada y dependiente de persona.", "why_not_lower": "No bajo más sin validación de eficacia, estándar visual y reacción.", "evidence_to_reduce": "Añadir patrón visual, auditoría de eficacia, frecuencia, reacción y/o cámara.", "questions": []}
    if no_evidence:
        return {"value": 10, "why": "No hay método/evidencia fiable de detección.", "why_not_lower": "Sin evidencia no se puede justificar mejor Detection.", "evidence_to_reduce": "Definir método, frecuencia, MSA, reacción y registros.", "questions": []}
    return {"value": None, "why": "El control no se puede clasificar con seguridad.", "why_not_lower": "Detection necesita método, frecuencia, cobertura, MSA y reacción.", "evidence_to_reduce": "Indica si es manual/automático, 100%/muestreo, frecuencia, MSA y reacción.", "questions": ["¿El control es 100% o por muestreo? ¿Automático o manual? ¿Tiene MSA y rechazo/reacción?"]}

def build_response(req: GuidedRequest) -> Dict[str, Any]:
    s = rate_severity(req.failure_effect or "")
    o = rate_occurrence(req.occurrence_evidence or "")
    d = rate_detection(req.detection_control or "")

    S = manual_score(req.current_S) or s["value"]
    O = manual_score(req.current_O) or o["value"]
    D = manual_score(req.current_D) or d["value"]

    AP, RPN = ap_value(S, O, D)
    missing = []
    if S is None: missing.extend(s["questions"])
    if O is None: missing.extend(o["questions"])
    if D is None: missing.extend(d["questions"])
    missing = missing[:3]

    status = "complete" if AP is not None else ("partial" if any(x is not None for x in [S, O, D]) else "not_enough_evidence")
    plain_parts = []
    if S is not None: plain_parts.append(f"S{S}: {s['why'] if not manual_score(req.current_S) else 'valor fijado manualmente por el usuario.'}")
    if O is not None: plain_parts.append(f"O{O}: {o['why'] if not manual_score(req.current_O) else 'valor fijado manualmente por el usuario.'}")
    if D is not None: plain_parts.append(f"D{D}: {d['why'] if not manual_score(req.current_D) else 'valor fijado manualmente por el usuario.'}")
    if not plain_parts:
        plain_parts.append("Not enough evidence in the approved sources")

    why_not = []
    if S is not None: why_not.append("S: " + s["why_not_lower"])
    if O is not None: why_not.append("O: " + o["why_not_lower"])
    if D is not None: why_not.append("D: " + d["why_not_lower"])

    evidence = []
    if S is not None: evidence.append("S: " + s["evidence_to_reduce"])
    if O is not None: evidence.append("O: " + o["evidence_to_reduce"])
    if D is not None: evidence.append("D: " + d["evidence_to_reduce"])

    sources = []
    if S is not None:
        src = source_row("severity fmea", S); sources.append({"rating":"S", "source":src.get("source","severity fmea"), "ref":src.get("ref",f"S{S}"), "reason":src.get("snippet","")})
    if O is not None:
        src = source_row("occurrency", O); sources.append({"rating":"O", "source":src.get("source","occurrency"), "ref":src.get("ref",f"O{O}"), "reason":src.get("snippet","")})
    if D is not None:
        src = source_row("detection", D); sources.append({"rating":"D", "source":src.get("source","detection"), "ref":src.get("ref",f"D{D}"), "reason":src.get("snippet","")})
    if AP is not None:
        sources.append({"rating":"AP", "source":"ap table", "ref":f"S{S}-O{O}-D{D}", "reason":"AP comes from uploaded AP table. RPN is informative only."})

    return {
        "status": status,
        "S": S, "O": O, "D": D,
        "AP": AP, "RPN": RPN,
        "confidence": "high" if status == "complete" and len(missing)==0 else "medium" if any(x is not None for x in [S,O,D]) else "low",
        "plain_explanation": "\n".join(plain_parts),
        "why_not_lower": "\n".join(why_not) if why_not else "No se puede bajar nada sin evidencia documentada.",
        "evidence_to_reduce": "\n".join(evidence) if evidence else "Describe efecto, histórico y control actual.",
        "missing_questions": missing,
        "sources": sources,
        "note": "RPN is informative only. AP comes from uploaded table. Ratings are suggestions; PFMEA owner must approve."
    }

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/api/health")
def health():
    return {"ok": True, "ai_enabled": bool(OPENAI_API_KEY), "sources": list(DB.keys()), "mode": "guided-v3-rules-plus-ai-ready"}

@app.get("/api/data")
def data():
    return DB

@app.get("/api/calc")
def calc(S: int, O: int, D: int):
    AP, RPN = ap_value(S, O, D)
    if AP is None:
        raise HTTPException(status_code=400, detail="S/O/D must be 1..10")
    return {"S": S, "O": O, "D": D, "AP": AP, "RPN": RPN, "note": "RPN is informative only. AP comes from uploaded AP table."}

@app.post("/api/guided-advice")
def guided_advice(req: GuidedRequest):
    try:
        return build_response(req)
    except Exception as e:
        return {
            "status": "not_enough_evidence",
            "S": None, "O": None, "D": None, "AP": None, "RPN": None,
            "plain_explanation": "Not enough evidence in the approved sources",
            "why_not_lower": "El backend no pudo justificar rating con seguridad.",
            "evidence_to_reduce": "Revisa que los datos estén separados en los 3 campos: efecto, ocurrencia y detección.",
            "missing_questions": ["¿Puedes separar el caso en efecto del fallo, histórico del proceso y control actual?"],
            "sources": [],
            "technical_error": str(e)[:300]
        }

@app.post("/api/pfmea-advice")
def pfmea_advice(req: AdviceRequest):
    text = req.case_text or ""
    t = clean(text)
    guided = GuidedRequest(question=text)
    # Reparto básico para compatibilidad con la caja antigua.
    if has_any(t, ["seguridad", "safety", "función", "funcion", "vehículo", "vehiculo", "cliente", "freno", "brake", "scrap", "montaje"]):
        guided.failure_effect = text
    if has_any(t, ["años", "years", "ppm", "reclam", "claims", "estable", "capacidad", "cpk", "nuevo", "sop", "incidencia"]):
        guided.occurrence_evidence = text
    if has_any(t, ["control", "micrómetro", "micrometro", "micrometer", "cmm", "cámara", "camera", "visual", "msa", "100", "torque", "ptc", "poka"]):
        guided.detection_control = text
    return build_response(guided)
