import json, os, re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.getenv('PFMEA_DATA', os.path.join(BASE_DIR, 'pfmea-data.json'))
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()
ALLOWED_ORIGIN = os.getenv('ALLOWED_ORIGIN', '*')

app = FastAPI(title='PFMEA AIAG-VDA Practical Advisor')
app.add_middleware(CORSMiddleware, allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != '*' else ['*'], allow_credentials=False, allow_methods=['*'], allow_headers=['*'])

class GuidedRequest(BaseModel):
    failure_effect: Optional[str] = ''       # para Severity
    occurrence_evidence: Optional[str] = ''  # para Occurrence
    detection_control: Optional[str] = ''    # para Detection
    current_S: Optional[int] = None
    current_O: Optional[int] = None
    current_D: Optional[int] = None
    question: Optional[str] = ''

class AdviceRequest(BaseModel):
    case_text: str


def load_db() -> Dict[str, List[Dict[str, Any]]]:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

DB = load_db()
AP_MAP = {f"{int(r.get('S'))}-{int(r.get('O'))}-{int(r.get('D'))}": str(r.get('AP')).strip() for r in DB.get('ap table', []) if str(r.get('S','')).isdigit()}

def ap_value(S: Optional[int], O: Optional[int], D: Optional[int]):
    if all(isinstance(x, int) and 1 <= x <= 10 for x in [S,O,D]):
        return AP_MAP.get(f'{S}-{O}-{D}', 'NA'), S*O*D
    return None, None

def norm(text: str) -> List[str]:
    return [w for w in re.findall(r'[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ]+', (text or '').lower()) if len(w) > 2]

def row_text(source: str, row: Dict[str, Any]) -> str:
    return source + '\n' + '\n'.join(f'{k}: {v}' for k, v in row.items() if str(v).strip())

DOCUMENTS=[]
for source, rows in DB.items():
    for i,row in enumerate(rows):
        txt=row_text(source,row)
        DOCUMENTS.append({'source':source,'idx':i,'text':txt,'row':row,'tokens':set(norm(txt))})

def retrieve(query: str, allowed_sources: List[str], k: int = 8):
    q=set(norm(query)); scored=[]
    ql=(query or '').lower()
    boosts=['micrometer','micrómetro','cmm','camera','cámara','ptc','torque','visual','msa','grr','r&r','kappa','capability','capacidad','cpk','ppk','poka','yoke','100%','daily','diaria','cada','hours','horas','claims','reclamaciones','ppm','years','años','scrap','safety','seguridad','regulation','reglament']
    for d in DOCUMENTS:
        if d['source'] not in allowed_sources: continue
        tl=d['text'].lower()
        score=len(q & d['tokens']) + 4*sum(1 for t in boosts if t in ql and t in tl)
        if score: scored.append((score,d))
    scored.sort(key=lambda x:x[0], reverse=True)
    return [d for _,d in scored[:k]]

def docs_context(docs):
    return '\n\n---\n\n'.join(f"SOURCE={d['source']} ROW={d['idx']}\n{d['text'][:2200]}" for d in docs)

def fallback_partial(req: GuidedRequest):
    # No inventa. Si no hay IA, solo devuelve preguntas y calcula con valores manuales.
    S=req.current_S if isinstance(req.current_S,int) else None
    O=req.current_O if isinstance(req.current_O,int) else None
    D=req.current_D if isinstance(req.current_D,int) else None
    missing=[]
    if not S and not req.failure_effect: missing.append('Describe el efecto del fallo para poder valorar Severity. Ejemplo: ¿afecta a seguridad, función de freno, montaje o solo scrap interno?')
    if not O and not req.occurrence_evidence: missing.append('Describe evidencia de ocurrencia: años en producción, reclamaciones, PPM, incidencias internas, capacidad, cambios de proceso.')
    if not D and not req.detection_control: missing.append('Describe el control de detección: método, frecuencia, 100%, automático/manual, MSA y plan de reacción.')
    AP,RPN=ap_value(S,O,D)
    return {'mode':'partial','not_enough_evidence': bool(missing), 'S':S,'O':O,'D':D,'AP':AP,'RPN':RPN,
            'plain_explanation':'Sin IA activa: solo puedo calcular con S/O/D introducidos manualmente y pedir la evidencia que falta.',
            'why_not_lower':'No se baja ninguna puntuación sin evidencia documentada en tablas aprobadas.',
            'missing_questions':missing or [], 'sources':[]}

def call_ai(req: GuidedRequest, contexts: Dict[str, List[Dict[str,Any]]]):
    from openai import OpenAI
    client=OpenAI(api_key=OPENAI_API_KEY)
    approved = {
        'Severity_context': docs_context(contexts.get('S',[])),
        'Occurrence_context': docs_context(contexts.get('O',[])),
        'Detection_context': docs_context(contexts.get('D',[])),
    }
    system = '''You are a practical PFMEA AIAG-VDA coach for an automotive plant.
CRITICAL RULES:
1) Use ONLY the approved context provided. Do not use general knowledge.
2) Do NOT invent S/O/D. If the user only gives detection information, only suggest D and ask for S/O evidence.
3) AP and RPN can only be calculated when S, O and D are all known.
4) Explain in very simple Feynman style: "I choose this because... not lower because... to lower it you need...".
5) Be Socratic: ask the minimum next questions needed. Max 3 questions.
6) Severity needs failure effect/customer/end-user/regulatory impact. Occurrence needs process history/frequency/capability/claims. Detection needs method/frequency/automatic or manual/MSA/reaction plan.
7) Return only valid JSON.
JSON schema:
{"status":"complete|partial|not_enough_evidence", "S":number|null, "O":number|null, "D":number|null, "confidence":"high|medium|low", "plain_explanation":string, "why_not_lower":string, "evidence_to_reduce":string, "missing_questions":[string], "sources":[{"rating":"S|O|D", "source":string, "ref":string, "reason":string}]}
If approved sources do not support a rating, set that rating null and ask a question. If almost nothing matches, status="not_enough_evidence" and plain_explanation="Not enough evidence in the approved sources".'''
    user = json.dumps({
        'user_inputs': req.model_dump(),
        'approved_context': approved,
        'instruction': 'Propose only the ratings that are justified. Do not calculate AP unless backend has S,O,D.'
    }, ensure_ascii=False)
    resp=client.chat.completions.create(model=OPENAI_MODEL, temperature=0, response_format={'type':'json_object'}, messages=[{'role':'system','content':system},{'role':'user','content':user}])
    try:
        out=json.loads(resp.choices[0].message.content)
    except Exception:
        return {'status':'not_enough_evidence','plain_explanation':'Not enough evidence in the approved sources','missing_questions':['Reformula el caso separando efecto, histórico y control de detección.']}
    # Valores manuales tienen prioridad si el usuario ya los fijó
    for key in ['S','O','D']:
        manual=getattr(req, f'current_{key}')
        if isinstance(manual,int) and 1<=manual<=10:
            out[key]=manual
    S,O,D=out.get('S'),out.get('O'),out.get('D')
    for key,val in [('S',S),('O',O),('D',D)]:
        if not (isinstance(val,int) and 1<=val<=10): out[key]=None
    AP,RPN=ap_value(out.get('S'), out.get('O'), out.get('D'))
    out['AP']=AP; out['RPN']=RPN
    if AP is None:
        out['status']='partial' if any(out.get(x) for x in ['S','O','D']) else 'not_enough_evidence'
    else:
        out['status']='complete'
    return out

@app.get('/')
def home(): return FileResponse(os.path.join(BASE_DIR,'index.html'))

@app.get('/api/health')
def health(): return {'ok':True,'ai_enabled':bool(OPENAI_API_KEY),'sources':list(DB.keys()), 'mode':'guided-v2'}

@app.get('/api/data')
def data(): return DB

@app.get('/api/calc')
def calc(S:int,O:int,D:int):
    AP,RPN=ap_value(S,O,D)
    if AP is None: raise HTTPException(status_code=400, detail='S/O/D must be 1..10')
    return {'S':S,'O':O,'D':D,'AP':AP,'RPN':RPN,'note':'RPN is informative only. AP comes from uploaded AP table.'}

@app.post('/api/guided-advice')
def guided_advice(req: GuidedRequest):
    # Recuperación separada: evita que una frase de Detection contamine Severity/Occurrence.
    s_query = (req.failure_effect or '') + ' ' + (req.question or '')
    o_query = (req.occurrence_evidence or '') + ' ' + (req.question or '')
    d_query = (req.detection_control or '') + ' ' + (req.question or '')
    contexts = {
        'S': retrieve(s_query, ['severity fmea'], 6) if s_query.strip() else [],
        'O': retrieve(o_query, ['occurrency'], 6) if o_query.strip() else [],
        'D': retrieve(d_query, ['detection'], 6) if d_query.strip() else [],
    }
    if not OPENAI_API_KEY:
        return fallback_partial(req)
    return call_ai(req, contexts)

# Compatibilidad con el endpoint antiguo: manda todo como pregunta libre, pero ya no inventa completo.
@app.post('/api/pfmea-advice')
def pfmea_advice(req: AdviceRequest):
    text=req.case_text.strip()
    if len(text)<5: raise HTTPException(status_code=400, detail='case_text too short')
    guided=GuidedRequest(question=text, failure_effect='', occurrence_evidence='', detection_control='')
    # Si el texto contiene pistas claras, las copiamos a los tres campos pero el prompt obliga a preguntar si falta evidencia.
    guided.failure_effect=text if any(w in text.lower() for w in ['safety','seguridad','cliente','customer','función','function','reglament','freno','brake']) else ''
    guided.occurrence_evidence=text if any(w in text.lower() for w in ['años','years','ppm','reclam','claims','capacidad','cpk','estable','nuevo','sop']) else ''
    guided.detection_control=text if any(w in text.lower() for w in ['micrómetro','micrometer','cmm','cámara','camera','visual','100','msa','control','torque','ptc']) else ''
    return guided_advice(guided)
