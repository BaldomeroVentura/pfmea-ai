import json, os, re
from typing import Any, Dict, List
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

app = FastAPI(title='PFMEA AIAG-VDA Advisor')
app.add_middleware(CORSMiddleware, allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != '*' else ['*'], allow_credentials=False, allow_methods=['*'], allow_headers=['*'])
class AdviceRequest(BaseModel):
    case_text: str

def load_db() -> Dict[str, List[Dict[str, Any]]]:
    with open(DATA_FILE, 'r', encoding='utf-8') as f: return json.load(f)
DB = load_db()
AP_MAP = {f"{int(r.get('S'))}-{int(r.get('O'))}-{int(r.get('D'))}": str(r.get('AP')).strip() for r in DB.get('ap table', []) if str(r.get('S','')).isdigit()}

def norm(text: str) -> List[str]:
    return [w for w in re.findall(r'[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ]+', text.lower()) if len(w) > 2]
def row_text(source: str, row: Dict[str, Any]) -> str:
    return source + '\n' + '\n'.join(f'{k}: {v}' for k, v in row.items())
DOCUMENTS=[]
for source, rows in DB.items():
    for i,row in enumerate(rows):
        txt=row_text(source,row); DOCUMENTS.append({'source':source,'idx':i,'text':txt,'row':row,'tokens':set(norm(txt))})

def retrieve(query: str, k: int = 24):
    q=set(norm(query)); scored=[]; terms=['micrometer','micrómetro','cmm','camera','cámara','ptc','torque','demonstrated','msa','visual','kappa','grr','cgk','cpk','poka']
    ql=query.lower()
    for d in DOCUMENTS:
        tl=d['text'].lower(); score=len(q & d['tokens']) + 3*sum(1 for t in terms if t in tl and t in ql)
        if score: scored.append((score,d))
    scored.sort(key=lambda x:x[0], reverse=True)
    return [d for _,d in scored[:k]]

def extract_score(row, letter):
    for k,v in row.items():
        key=str(k).upper().strip()
        if key.startswith(letter.upper()+' ') or key == letter.upper() or letter.upper() in key[:4]:
            m=re.search(r'\b(10|[1-9])\b', str(v))
            if m: return int(m.group(1))
    m=re.search(letter+r'\s*[=:]?\s*(10|[1-9])', json.dumps(row,ensure_ascii=False), re.I)
    return int(m.group(1)) if m else None

def local_answer(docs):
    sev=next((d for d in docs if d['source']=='severity fmea'),None); occ=next((d for d in docs if d['source']=='occurrency'),None); det=next((d for d in docs if d['source']=='detection'),None)
    if not (sev and occ and det): return {'not_enough_evidence': True, 'message':'Not enough evidence in the approved sources'}
    S,O,D=extract_score(sev['row'],'S'),extract_score(occ['row'],'O'),extract_score(det['row'],'D')
    if not (S and O and D): return {'not_enough_evidence': True, 'message':'Not enough evidence in the approved sources'}
    return {'not_enough_evidence':False,'S':S,'O':O,'D':D,'AP':AP_MAP.get(f'{S}-{O}-{D}','NA'),'RPN':S*O*D,'explanation':'Fallback sin OpenAI: coincidencia local contra tablas aprobadas. Revisar manualmente antes de liberar PFMEA.','why_not_lower':'No bajar S/O/D sin evidencia explícita en las fuentes aprobadas.','evidence_missing':'Para bajar O/D: histórico demostrado, MSA válido, capacidad, cero escapes, reacción validada, bloqueo automático o poka-yoke según aplique.','sources':[{'source':d['source'],'ref':f"row {d['idx']}",'excerpt':d['text'][:600]} for d in [sev,occ,det]]}

@app.get('/')
def home(): return FileResponse(os.path.join(BASE_DIR,'index.html'))
@app.get('/api/health')
def health(): return {'ok':True,'ai_enabled':bool(OPENAI_API_KEY),'sources':list(DB.keys())}
@app.post('/api/pfmea-advice')
def pfmea_advice(req: AdviceRequest):
    case=req.case_text.strip()
    if len(case)<10: raise HTTPException(status_code=400, detail='case_text too short')
    docs=retrieve(case)
    if len(docs)<3: return {'not_enough_evidence': True, 'message':'Not enough evidence in the approved sources'}
    if not OPENAI_API_KEY: return local_answer(docs)
    context='\n\n---\n\n'.join(f"SOURCE={d['source']} ROW={d['idx']}\n{d['text'][:2500]}" for d in docs)
    from openai import OpenAI
    client=OpenAI(api_key=OPENAI_API_KEY)
    system='''You are a practical PFMEA AIAG-VDA assistant for automotive manufacturing. Use ONLY the approved context provided. Do not use general knowledge. Return only valid JSON. If the context does not justify S, O and D, return {"not_enough_evidence": true, "message":"Not enough evidence in the approved sources"}. AP is assigned by backend from approved AP table. RPN is informative only. Never invent evidence. Explain why ratings cannot be lower and what evidence would be required. JSON schema: {"not_enough_evidence": boolean, "S": number|null, "O": number|null, "D": number|null, "explanation": string, "why_not_lower": string, "evidence_missing": string, "sources": [{"source": string, "ref": string, "excerpt": string}]}'''
    resp=client.chat.completions.create(model=OPENAI_MODEL,temperature=0,response_format={'type':'json_object'},messages=[{'role':'system','content':system},{'role':'user','content':f'CASE:\n{case}\n\nAPPROVED CONTEXT:\n{context}'}])
    try: out=json.loads(resp.choices[0].message.content)
    except Exception: return {'not_enough_evidence': True, 'message':'Not enough evidence in the approved sources'}
    if out.get('not_enough_evidence'):
        out['message']='Not enough evidence in the approved sources'; return out
    S,O,D=out.get('S'),out.get('O'),out.get('D')
    if not all(isinstance(x,int) and 1<=x<=10 for x in [S,O,D]): return {'not_enough_evidence': True, 'message':'Not enough evidence in the approved sources'}
    out['AP']=AP_MAP.get(f'{S}-{O}-{D}','NA'); out['RPN']=S*O*D; out['not_enough_evidence']=False
    return out
