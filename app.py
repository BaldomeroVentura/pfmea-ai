import json, os, re, traceback
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.getenv('PFMEA_DATA', os.path.join(BASE_DIR, 'pfmea-data.json'))
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()

app = FastAPI(title='PFMEA guided advisor')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=False, allow_methods=['*'], allow_headers=['*'])

class GuidedRequest(BaseModel):
    failure_effect: Optional[str] = ''
    occurrence_evidence: Optional[str] = ''
    detection_control: Optional[str] = ''
    current_S: Optional[int] = None
    current_O: Optional[int] = None
    current_D: Optional[int] = None
    question: Optional[str] = ''

class AdviceRequest(BaseModel):
    case_text: str

def safe_json(data, status_code=200):
    return JSONResponse(content=data, status_code=status_code)

def load_db():
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {'_load_error': [{'error': str(e)}]}

DB = load_db()

def source_rows(name_contains):
    out = []
    for key, rows in DB.items():
        if name_contains.lower() in key.lower() and isinstance(rows, list):
            out.extend((key, i, r) for i, r in enumerate(rows))
    return out

AP_MAP = {}
for key, rows in DB.items():
    if 'ap' in key.lower() and 'table' in key.lower() and isinstance(rows, list):
        for r in rows:
            try:
                S, O, D = int(r.get('S')), int(r.get('O')), int(r.get('D'))
                AP_MAP[f'{S}-{O}-{D}'] = str(r.get('AP')).strip()
            except Exception:
                pass

def ap_value(S, O, D):
    if all(isinstance(x, int) and 1 <= x <= 10 for x in [S, O, D]):
        return AP_MAP.get(f'{S}-{O}-{D}', 'NA'), S * O * D
    return None, None

def words(text):
    return set(w for w in re.findall(r'[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ%]+', (text or '').lower()) if len(w) > 2)

def retrieve(query, source_hint, k=4):
    q = words(query)
    if not q:
        return []
    candidates = source_rows(source_hint)
    scored = []
    for source, idx, row in candidates:
        txt = json.dumps(row, ensure_ascii=False)
        score = len(q & words(txt))
        # practical boosts
        low = (query or '').lower(); tl = txt.lower()
        for b in ['micrómetro','micrometer','cmm','cámara','camera','100','visual','msa','reclam','claims','años','years','ppm','seguridad','safety','función','function','freno','brake']:
            if b in low and b in tl:
                score += 4
        if score:
            scored.append((score, {'source': source, 'row': idx, 'text': txt[:1600]}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]

def fallback(req: GuidedRequest, reason=''):
    S = req.current_S if isinstance(req.current_S, int) and 1 <= req.current_S <= 10 else None
    O = req.current_O if isinstance(req.current_O, int) and 1 <= req.current_O <= 10 else None
    D = req.current_D if isinstance(req.current_D, int) and 1 <= req.current_D <= 10 else None
    missing = []
    if not S and not (req.failure_effect or '').strip():
        missing.append('Describe el efecto del fallo: seguridad, función, cliente, regulación o solo scrap interno.')
    if not O and not (req.occurrence_evidence or '').strip():
        missing.append('Describe el histórico: años, reclamaciones, PPM, incidencias, capacidad o proceso nuevo.')
    if not D and not (req.detection_control or '').strip():
        missing.append('Describe el control: método, frecuencia, automático/manual, MSA y reacción.')
    AP, RPN = ap_value(S, O, D)
    return {
        'status': 'partial' if (S or O or D or missing) else 'not_enough_evidence',
        'S': S, 'O': O, 'D': D, 'AP': AP, 'RPN': RPN,
        'confidence': 'low',
        'plain_explanation': 'No calculo S/O/D si no hay evidencia separada. Primero necesito completar los 4 pasos.' + (f' Error técnico: {reason}' if reason else ''),
        'why_not_lower': 'No se puede bajar ninguna puntuación sin evidencia documentada en las fuentes aprobadas.',
        'evidence_to_reduce': 'Aporta evidencia concreta: histórico, MSA, frecuencia, automatización, reacción y trazabilidad.',
        'missing_questions': missing[:3],
        'sources': []
    }

def dump_model(req):
    try:
        return req.model_dump()
    except Exception:
        return req.dict()

def ai_advice(req: GuidedRequest, contexts):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    system = '''You are a practical PFMEA AIAG-VDA plant coach.
Use ONLY the supplied approved context. Never invent ratings.
If the user gives only Detection information, suggest only D and ask for S and O.
If the user gives only Severity information, suggest only S and ask for O and D.
If the user gives only Occurrence information, suggest only O and ask for S and D.
Calculate AP/RPN only when S, O and D are all known.
Explain in simple plant language, not theory.
Return ONLY valid JSON with this schema:
{"status":"complete|partial|not_enough_evidence","S":null,"O":null,"D":null,"confidence":"high|medium|low","plain_explanation":"","why_not_lower":"","evidence_to_reduce":"","missing_questions":[],"sources":[]}
If evidence is weak: leave rating null and ask one clear question.
'''
    payload = {
        'user_inputs': dump_model(req),
        'approved_context': contexts,
        'plant_rule': 'No AP unless S,O,D are known. RPN informative only.'
    }
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={'type': 'json_object'},
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(payload, ensure_ascii=False)}]
    )
    out = json.loads(resp.choices[0].message.content)
    for k in ['S','O','D']:
        manual = getattr(req, f'current_{k}')
        if isinstance(manual, int) and 1 <= manual <= 10:
            out[k] = manual
        elif not (isinstance(out.get(k), int) and 1 <= out.get(k) <= 10):
            out[k] = None
    AP, RPN = ap_value(out.get('S'), out.get('O'), out.get('D'))
    out['AP'], out['RPN'] = AP, RPN
    if AP is None and out.get('status') == 'complete':
        out['status'] = 'partial'
    return out

@app.exception_handler(Exception)
async def all_errors(request, exc):
    # Important: frontend always receives JSON, never plain "Internal Server Error".
    return safe_json({'status':'technical_error','error':str(exc),'plain_explanation':'Error técnico en backend. Mira Render logs para detalle. No se ha calculado nada.'}, 200)

@app.get('/')
def home():
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

@app.get('/api/health')
def health():
    return {'ok': True, 'ai_enabled': bool(OPENAI_API_KEY), 'model': OPENAI_MODEL, 'sources': list(DB.keys()), 'ap_entries': len(AP_MAP)}

@app.get('/api/data')
def data():
    return DB

@app.get('/api/calc')
def calc(S:int, O:int, D:int):
    AP, RPN = ap_value(S,O,D)
    return {'S':S,'O':O,'D':D,'AP':AP,'RPN':RPN,'note':'RPN is informative only. AP comes from uploaded AP table.'}

@app.post('/api/guided-advice')
def guided(req: GuidedRequest):
    try:
        contexts = {
            'Severity': retrieve(req.failure_effect or req.question or '', 'severity', 5),
            'Occurrence': retrieve(req.occurrence_evidence or req.question or '', 'occurr', 5),
            'Detection': retrieve(req.detection_control or req.question or '', 'detect', 5),
        }
        if not OPENAI_API_KEY:
            return fallback(req, 'OPENAI_API_KEY no configurada')
        return ai_advice(req, contexts)
    except Exception as e:
        print('GUIDED ERROR:', traceback.format_exc())
        return fallback(req, str(e))

@app.post('/api/pfmea-advice')
def old(req: AdviceRequest):
    text = req.case_text or ''
    return guided(GuidedRequest(question=text))
