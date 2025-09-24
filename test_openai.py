from openai import OpenAI

client = OpenAI()

try:
    resp = client.models.list()
    print([m.id for m in resp.data[:5]])
except Exception as e:
    print("❌ Błąd połączenia:", e)
