# Результат 3. REST API (`api/main.py`)

Повна специфікація OpenAPI: `docs/openapi.json` (також `http://localhost:8000/docs` після запуску).

## Запуск

```bash
# у .env: API_KEY_PUBLIC, API_KEY_ANALYST (>=16 символів), RATE_LIMIT_PER_MIN (за замовчуванням 10)
uvicorn api.main:app --port 8000          # з кореня репозиторію, ОДИН процес (сесії й ліміти — в пам'яті)
python src/frontend.py                     # необов'язково: веб-інтерфейс на :7860
```

## Ендпоінти

| Метод | Шлях | Ключ | Опис |
|---|---|---|---|
| GET | `/health` | не потрібен | стан сервера, доступність графа, кількість чанків |
| POST | `/ask` | `X-API-Key` | питання → відповідь, джерела, підграф, токени, затримка |
| GET | `/sources/{id}` | `X-API-Key` | повний текст чанка за `chunk_id` (з перевіркою прав) |

`POST /ask`, тіло (Pydantic, зайві поля заборонені):

| Поле | Тип | Правила |
|---|---|---|
| `prompt` | string | 3–1000 символів, лише рядок |
| `user_id` | UUID, необов'язково | сесія діалогу; без нього створюється нова (повертається у відповіді) |
| `use_graph` | bool, за замовч. `true` | `false` — режим «лише текст» |

Відповідь: `answer`, `sources[]` (лише цитовані й перевірені чанки), `graph{used,mode,entities,facts}`, `usage{tokens_in,tokens_out}`, `latency_ms`, `citations_valid`, `invalid_citations`, `role`, `user_id`.

## Автентифікація і ролі

Роль визначається **лише** ключем на сервері; клієнт не може її передати (поле `role` у тілі → 422). Ключі порівнюються `secrets.compare_digest`. Сесія діалогу прив'язана до пари (роль, `user_id`), тож чужу історію іншим ключем не отримати.

| Роль | Ключ (змінна `.env`) | Дозволені `sensitivity` |
|---|---|---|
| `public` | `API_KEY_PUBLIC` | `public` |
| `analyst` | `API_KEY_ANALYST` | `public`, `restricted` |

Права застосовуються **до** ранжування (pre-filter у гібридному пошуку), а в `/ask` додатково перевіряються після генерації (друга лінія захисту), у `/sources/{id}` — повертає 403.

## Ліміти й коди помилок

Ковзне вікно 60 с на ключ: `RATE_LIMIT_PER_MIN` запитів (за замовч. 10), далі 429 із заголовком `Retry-After`.

| Код | Коли |
|---|---|
| 200 | успіх |
| 401 | немає або невірний `X-API-Key` (`WWW-Authenticate: ApiKey`) |
| 403 | роль не має доступу до джерела (`/sources/{id}`) |
| 404 | джерела з таким id немає |
| 422 | невалідний вхід: короткий/довгий/порожній `prompt`, зайві поля (`role`), не-UUID `user_id` |
| 429 | перевищено ліміт, див. `Retry-After` |
| 500 | збій генерації або порушення політики доступу |

## Приклади curl

```bash
export PUB=$API_KEY_PUBLIC ANA=$API_KEY_ANALYST

curl -s localhost:8000/health

# роль public: питання про норму ПДР
curl -s -X POST localhost:8000/ask -H "X-API-Key: $PUB" -H "Content-Type: application/json" \
     -d '{"prompt": "Яка максимальна швидкість у населених пунктах?", "use_graph": false}'

# те саме питання про restricted-джерело: public -> «не маю інформації», analyst -> відповідь із джерелами
curl -s -X POST localhost:8000/ask -H "X-API-Key: $PUB" -H "Content-Type: application/json" \
     -d '{"prompt": "Скільки світлофорів у Львові і яка частка працює онлайн?"}'
curl -s -X POST localhost:8000/ask -H "X-API-Key: $ANA" -H "Content-Type: application/json" \
     -d '{"prompt": "Скільки світлофорів у Львові і яка частка працює онлайн?"}'

# повний текст джерела (# у URL кодується як %23)
curl -s -H "X-API-Key: $ANA" "localhost:8000/sources/lviv_svitloforni_rezhymy%23str001"
curl -s -o /dev/null -w "%{http_code}\n" -H "X-API-Key: $PUB" "localhost:8000/sources/lviv_svitloforni_rezhymy%23str001"   # 403

# помилки
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/ask -d '{"prompt":"x"}' -H "Content-Type: application/json"            # 401
curl -s -X POST localhost:8000/ask -H "X-API-Key: $PUB" -H "Content-Type: application/json" -d '{"prompt":"привіт","role":"analyst"}'  # 422
for i in $(seq 12); do curl -s -o /dev/null -w "%{http_code} " -H "X-API-Key: $PUB" "localhost:8000/sources/nope"; done                # ... 429
```

## Перевірка

`pytest src/test_api.py` (заглушка моделі): без ключа → 401; дві ролі бачать різні джерела; `/sources` → 403/404; 422 на 7 видах невалідного входу; 429 + `Retry-After`; сесія прив'язана до ролі. Журнал запитів: `logs/api_log.jsonl`.

## Відомі обмеження

- Підграф Neo4j не фільтрується за `sensitivity` (граф ЛР 9 синтетичний); у production його треба обмежувати за роллю.
- Ліміти й сесії — в пам'яті одного процесу (не масштабується на кілька воркерів).
