# ЛР 11. RAG-чат-бот із графом знань (Domain Knowledge Hub)

Вхід: корпус і індекс з ЛР 10 (`data/`), граф Neo4j з ЛР 9, золотий набір `eval/gold.jsonl`.

## Структура (артефакти за завданням)

| Результат | Артефакти |
|---|---|
| 1. RAG із цитуванням | `src/rag.py`, `src/llm_pipe.py`, `prompts/rag_v1.md`, журнал `logs/rag_log.jsonl` |
| 2. Підграф | `src/graph_context.py`, `docs/graph_vs_text.md` (генерується), `eval/graph_questions.jsonl` |
| 3. REST API | `api/main.py`, `docs/api.md`, `docs/openapi.json` |
| 4. Тести й метрики | `eval/rag_eval.jsonl`, `src/eval_rag.py`, `docs/rag_eval.md` (генерується) |
| 5. Оптимізація | `docs/rag_optimization.md` (генерується `compare`) |
| LLM-експерт | `llm_log/*.jsonl`, `claims/claims.csv` |

## Запуск з нуля

```bash
python -m venv .venv && source .venv/bin/activate          # Python 3.10+ (тести перевірено на 3.12)
pip install -r requirements.txt
cp .env.example .env                                        # заповнити Neo4j, API_KEY_PUBLIC/ANALYST (secrets.token_urlsafe(24))
# Neo4j з ЛР 9 має працювати (без нього бот працює лише з текстом)

pytest src                                                  # 32 тести без моделі й Neo4j
python src/build_eval_set.py                                # (пере)збирає eval/rag_eval.jsonl і claims/claims.csv
python src/rag.py --demo                                    # Результат 1: 6 демо-запитів (потрібен LLM_API_KEY, див. .env.example)
uvicorn api.main:app --port 8000                            # Результат 3 (один процес)
python src/frontend.py                                      # веб-інтерфейс (Gradio) на :7860
```

Детермінізм: генерація `do_sample=False` (temperature/seed не застосовуються), вибірка для ручної перевірки — seed 42.

## Результати 2, 4, 5 (потребують GPU/модель — цифри з'являються після запуску)

```bash
# Результат 4
python src/eval_rag.py run --tag base --judge api           # JUDGE_API_KEY у .env; restricted-контексти судді НЕ йдуть без --judge-restricted
python src/eval_rag.py review-export --tag base --n 12      # заповніть human_score (0/1) в eval/manual_review.csv
python src/eval_rag.py kappa --tag base
python src/eval_rag.py report --tag base                    # -> docs/rag_eval.md

# Результат 2
python src/graph_context.py "що зачепить ДТП INC-317"       # перевірте, що ваші ID є в графі; за потреби змініть eval/graph_questions.jsonl
python src/eval_rag.py graph-ab                             # -> docs/graph_vs_text.md, eval/graph_ab_manual.csv
#   заповніть eval/graph_ab_manual.csv (0/1/2), потім: python src/eval_rag.py graph-ab --report-only

# Результат 5 (змінюємо ОДИН фактор — k; інші параметри ті самі)
RAG_TOP_K=3 python src/eval_rag.py run --tag k3
python src/eval_rag.py compare --a base --b k3 --factor "k: 5 -> 3 фрагменти" \
   --hypothesis "менше шуму й токенів для моделі: faithfulness не впаде, затримка зменшиться"
```

Інші параметри для експериментів: `RAG_PROMPT=rag_v2` (новий файл у `prompts/`), `RAG_STRICT_CITATIONS=0`.

## Протокол «LLM-експерт»

- `llm_log/<дата>.jsonl` — пишеться автоматично під час `run`/`graph-ab`: дата, модель, параметри, версія підказки, сирий вихід, поле `human_decision` (`pending` → змініть після перегляду).
- `claims/claims.csv` — 34 твердження з evidence корпусу; рівень V1 і статус `verified` виставлено автоперевіркою (дослівний підрядок у чанку). Для звіту додайте твердження, які пропонувала LLM, і порахуйте частку verified/rejected.
- Декларація використання ШІ (інструмент, призначення, приклад запиту, як перевірено) — у звіті.

## Безпека

`.env` і API-ключі не комітяться (`.gitignore`). Конфіденційні (restricted) контексти не передаються зовнішньому судді за замовчуванням.
