# ProjectMLops

How to run
1. .venv\Scripts\activate
2. pip install -r requirement.txt
3. pip install -r requirement.backend.txt
4. open : docker compose up --build
close : docker compose down

run backend
uvicorn backend.main:app --reload


web: http://34.21.147.39:8080/