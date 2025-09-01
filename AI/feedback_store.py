
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict,Any,List,Optional

class FeedbackStore:
    def __init__(self,path:Path):
        self.path=Path(path)

    def append(self,rec:Dict[str,Any]):
        self.path.parent.mkdir(parents=True,exist_ok=True)

        if "ts" not in rec:
            try:
                import time as _t
                rec["ts"]=_t.time()
            except Exception:
                pass

        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    def load_as_qa_pairs(self,q_key:str="question",a_key:str="answer")->List[Dict[str,str]]:
        pairs:List[Dict[str,str]]=[]
        if not self.path.exists():
            return pairs

        with open(self.path,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue

                try:
                    obj=json.loads(line)
                except Exception:
                    continue

                q=(obj.get('query') or "").strip()
                model_a=(obj.get("model_answer") or "").strip()
                corrected=(obj.get("correct_answer") or "").strip()
                label=(obj.get("label") or "").strip().lower()

                if not q:
                    continue
                if label=="up" and model_a:
                    pairs.append({q_key:q,a_key:model_a})
                elif label=="down" and corrected:
                    pairs.append({q_key:q,a_key:corrected})

        return pairs