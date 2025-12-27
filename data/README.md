Datasets are JSONL.

Each line:

```json
{
  "id": "ex_000001",
  "prompt": "Instruction + input text ...",
  "completion": "{\"key\":\"value\"}",
  "schema": { "key": "str" }
}
```

- `prompt` should clearly demand **ONLY valid JSON** and specify exact keys/types.
- `completion` must be the *exact* target JSON text (no markdown fences, no trailing commentary).
- `schema` is used for the behavioral metric (format success rate).

