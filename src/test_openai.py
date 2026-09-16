"""OpenAI anahtarini ve model secimini test eder. Calistir: py -3 src/test_openai.py"""
import os, sys, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
key = os.getenv("OPENAI_API_KEY", "")
if not key.startswith("sk-"):
    sys.exit("OPENAI_API_KEY bulunamadi. .env dosyasina yazin (.env.example'a bakin).")

client = OpenAI(api_key=key)
print("Hesabin erisebildigi ucuz modeller:")
names = sorted(m.id for m in client.models.list())
for n in names:
    if any(t in n for t in ("mini", "nano")) and "audio" not in n and "realtime" not in n:
        print("  -", n)

model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
t0 = time.time()
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Tek kelimeyle cevap ver: calisiyor musun?"}],
    max_tokens=10,
)
print(f"\nModel {model} cevap verdi ({time.time()-t0:.1f}s): {r.choices[0].message.content!r}")
print("Token kullanimi:", r.usage.prompt_tokens, "giris /", r.usage.completion_tokens, "cikis")
