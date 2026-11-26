#!/usr/bin/env bash
# Envía un prompt desde archivo a un endpoint compatible con OpenAI y mide el tiempo.
# Uso:   ./test_modelo.sh prompt_completo.txt [modelo] [max_tokens]
# Vars:  URL, EFFORT  (ej.: EFFORT=low URL=http://127.0.0.1:18000/v1/chat/completions ./test_modelo.sh ...)
set -euo pipefail

PROMPT_FILE="${1:?Uso: $0 prompt.txt [modelo] [max_tokens]}"
MODEL="${2:-inkling-2-mini}"
MAX_TOKENS="${3:-8000}"
URL="${URL:-http://127.0.0.1:18000/v1/chat/completions}"
EFFORT="${EFFORT:-none}"

OUT_DIR="resultados"; mkdir -p "$OUT_DIR"
BASE="$OUT_DIR/${MODEL}_$(basename "${PROMPT_FILE%.*}")_$(date +%Y%m%d_%H%M%S)"

# jq escapa correctamente saltos de línea y comillas del prompt
BODY=$(jq -n --arg model "$MODEL" --argjson max "$MAX_TOKENS" --arg effort "$EFFORT" \
  --rawfile prompt "$PROMPT_FILE" \
  '{model:$model, max_tokens:$max, reasoning_effort:$effort,
    messages:[{role:"user", content:$prompt}]}')

echo "→ Modelo: $MODEL | effort: $EFFORT | max_tokens: $MAX_TOKENS"

# curl mide el tiempo total de la petición
ELAPSED=$(curl -sS --max-time 1800 "$URL" \
  -H 'Content-Type: application/json' \
  -d "$BODY" -o "$BASE.json" -w '%{time_total}')

if jq -e '.error' "$BASE.json" >/dev/null 2>&1; then
  echo "Error del servidor:"; jq '.error' "$BASE.json"; exit 1
fi

jq -r '.choices[0].message.content // ""' "$BASE.json" > "$BASE.md"

PT=$(jq '.usage.prompt_tokens // 0' "$BASE.json")
CT=$(jq '.usage.completion_tokens // 0' "$BASE.json")
FIN=$(jq -r '.choices[0].finish_reason // "?"' "$BASE.json")
TPS=$(awk -v c="$CT" -v t="$ELAPSED" 'BEGIN{ if (t>0) printf "%.1f", c/t; else print "0" }')

echo "---------------------------------------------"
printf "Tiempo total:      %.2f s\n" "$ELAPSED"
echo   "Tokens entrada:    $PT"
echo   "Tokens salida:     $CT"
echo   "Velocidad:         $TPS tokens/s"
echo   "Finish reason:     $FIN   (si es 'length', sube max_tokens)"
echo   "Respuesta:         $BASE.md"
echo   "JSON completo:     $BASE.json"
