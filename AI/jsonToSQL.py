import json
from db import QAPair,Session
from pathlib import Path


data_path=Path(__file__).resolve().parents[1]/'Data'/"training_ready.jsonl"

KEY_CANDIDATES = [
    ("prompt", "completion"),
    ("question", "answer"),
    ("Question", "Answer"),
    ("input", "output"),
    ("instruction", "response"),
    ("user", "assistant"),
    ("q", "a"),
]
def main():
    session = Session()
    added=0
    with open(data_path, 'r', encoding='utf-8',errors='replace') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            item=json.loads(line)
            q=a=None
            for qk,ak in KEY_CANDIDATES:
                if qk in item and ak in item:
                    q=(item[qk] or "").strip()
                    a=(item[ak] or "").strip()
                    break

            if q and a:
                session.add(QAPair(question=q,answer=a))
                added+=1
    session.commit()

if __name__ == '__main__':
    main()