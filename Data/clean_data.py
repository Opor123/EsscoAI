import json
import re
from pathlib import Path

input_file = Path("essco_raw_data.json")
output_file = Path("Essco_product_cleaned_data.json")

SKIP_PHRASES = [
    "read more", "click here", "download", "certification",
    "iso", "copyright", "home", "upsups"
]

PRODUCT_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Z0-9]+[-/]?)*[A-Z]*\d+[A-Z0-9-]*\b|\b[A-Z]{2,}(?:-[A-Z0-9]+){1,}\b",
    re.IGNORECASE
)

WS_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://", re.I)

THAI_PREFIXES = (
    "สินค้าและบริการ ",
    "สินค้า ",
)
PLACEHOLDER_TITLES = {"สินค้าและบริการ", "สินค้า"}



def normalize_ws(text:str)->str:
    return WS_RE.sub(' ',text).strip()

def strip_prefixes(text:str)->str:
    for p in THAI_PREFIXES:
        if text.startswith(p):
            return text[len(p):]
    return text

def is_junk_line(text:str)->bool:
    t=text.strip().lower()
    if not t:
        return True
    if t in SKIP_PHRASES:
        return True

    if URL_RE.search(t) and len(t)<80:
        return True

    if len(t)<2:
        return True
    return False

def is_product_url(url:str)->bool:
    u=(url or '').lower()
    return 'product_id' in u

def clean_text_list(texts):
    if not isinstance(texts,list):
        return []

    seen=set()
    cleaned=[]
    for raw in texts:
        if not isinstance(raw,str):
            continue
        raw = normalize_ws(raw)

        parts=re.split(r"[•·ㆍ]|(?:\s-\s)|(?:\s\|\s)|(?:\s{2,})", raw)
        for p in parts:
            p=normalize_ws(p)
            if is_junk_line(p):
                continue
            key=p.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(p)

    return cleaned

def extract_products(raw):
    training_data=[]

    for category, info in (raw or {}).items():
        if not isinstance(info,dict) or 'product' not in info:
            continue


        for prod in info['product']:
            if not isinstance(prod,dict):
                continue

            url=normalize_ws(prod.get('url',''))
            if not is_product_url(url):
                continue

            title_raw=normalize_ws(prod.get('title',''))

            title=strip_prefixes(title_raw)

            if title in PLACEHOLDER_TITLES:
                continue

            model_matches=PRODUCT_PATTERN.findall(title)

            digital_candidate=[m for m in model_matches if any(ch.isdigit for ch in m)]

            if digital_candidate:
                model_name=sorted(digital_candidate,key=len, reverse=True)[0].strip('-_/')
            elif model_matches:
                model_name=sorted(model_matches,key=len,reverse=True)[0].strip('-_/')
            else:
                model_name=title

            details_src=[]
            for key in ('details','technical_details','context'):
                val=prod.get(key)
                if isinstance(val,list):
                    details_src.extend(val)

            cleaned_details=clean_text_list(details_src)

            if not (model_name or url):
                continue

            training_data.append({
                "category": category,
                "product_name": model_name,
                "details": cleaned_details,
                "url": url
            })

    seen=set()
    unique=[]
    for item in training_data:
        key=(item['url'] or "").lower() or item['product_name'].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique

def main():
    with open(input_file,'r',encoding='utf-8') as f:
        raw=json.load(f)

    ai=extract_products(raw)

    with open(output_file,'w',encoding='utf-8') as f:
        json.dump(ai,f,indent=2,ensure_ascii=False)

    print(f"✅ AI training dataset saved to {output_file}")
    print(f"📦 Total products: {len(ai)}")

if __name__ == "__main__":
    main()