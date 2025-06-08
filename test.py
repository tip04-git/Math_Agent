import requests
resp = requests.post("http://localhost:5001/ask", json={"query": "sum of first 5 prime numbers"})
print(resp.json())
